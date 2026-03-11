# wav2vec2 Forced Aligner — CoreML Export Plan

## Goal

Replace Whisper's DTW-based word alignment (91ms MAE) with wav2vec2 CTC forced alignment (~30-50ms MAE) using CoreML for inference.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID PIPELINE                               │
│                                                                  │
│  Audio ──► Whisper (MLX) ──► Text                                │
│    │                          │                                  │
│    │                          ▼                                  │
│    └──► wav2vec2 (CoreML) ──► Frame Probs ──► CTC Align ──► Words│
└─────────────────────────────────────────────────────────────────┘
```

**Benefits of CoreML:**
- Runs on Neural Engine (ANE), freeing GPU for Whisper
- Simpler export path (HuggingFace → ONNX → CoreML)
- Faster time-to-validation

**Tradeoffs:**
- Fixed input shapes (need padding strategy)
- Less control over optimization
- Separate runtime from MLX

---

## Phase 1: Model Export (4 hours)

### 1.1 Select Model Variant

```python
# Options (WhisperX uses language-specific models):
MODELS = {
    "en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",  # 317M, best quality
    "en-base": "facebook/wav2vec2-base-960h",               # 95M, faster
    "multilingual": "facebook/wav2vec2-xlsr-53-espeak-cv-ft" # Phoneme-based
}
```

**Recommendation:** Start with `wav2vec2-base-960h` for validation, upgrade to large if needed.

### 1.2 Export to ONNX

```python
# scripts/export_wav2vec2_onnx.py
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import onnx

MODEL_ID = "facebook/wav2vec2-base-960h"
MAX_AUDIO_SAMPLES = 16000 * 30  # 30 seconds at 16kHz

def export():
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()
    
    # Dummy input (batch=1, samples=max_length)
    dummy_input = torch.randn(1, MAX_AUDIO_SAMPLES)
    
    # Export with dynamic axes for variable length
    torch.onnx.export(
        model,
        dummy_input,
        "wav2vec2-base.onnx",
        input_names=["audio"],
        output_names=["logits"],
        dynamic_axes={
            "audio": {1: "samples"},
            "logits": {1: "frames"}
        },
        opset_version=14
    )
    
    # Verify
    onnx_model = onnx.load("wav2vec2-base.onnx")
    onnx.checker.check_model(onnx_model)
    print(f"Exported: {onnx_model.graph.input[0]}")

if __name__ == "__main__":
    export()
```

### 1.3 Convert ONNX to CoreML

```python
# scripts/convert_wav2vec2_coreml.py
import coremltools as ct
import onnx

def convert():
    onnx_model = onnx.load("wav2vec2-base.onnx")
    
    # Convert with flexible shape
    mlmodel = ct.converters.onnx.convert(
        onnx_model,
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.ALL,  # CPU + GPU + ANE
        inputs=[
            ct.TensorType(
                name="audio",
                shape=ct.Shape(shape=(1, ct.RangeDim(16000, 16000 * 30))),
                dtype=float
            )
        ]
    )
    
    # Optimize for ANE
    mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(
        mlmodel, nbits=16  # FP16 for ANE
    )
    
    mlmodel.save("wav2vec2-base.mlpackage")
    print("Saved wav2vec2-base.mlpackage")

if __name__ == "__main__":
    convert()
```

### 1.4 Validate Export

```python
# scripts/validate_wav2vec2_coreml.py
import coremltools as ct
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC

def validate():
    # Load both models
    torch_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    coreml_model = ct.models.MLModel("wav2vec2-base.mlpackage")
    
    # Test input (5 seconds of audio)
    audio = np.random.randn(1, 16000 * 5).astype(np.float32)
    
    # PyTorch inference
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(audio)).logits.numpy()
    
    # CoreML inference
    coreml_out = coreml_model.predict({"audio": audio})["logits"]
    
    # Compare
    diff = np.abs(torch_out - coreml_out).max()
    print(f"Max difference: {diff:.6f}")
    assert diff < 0.01, "Output mismatch!"
    print("✓ Validation passed")

if __name__ == "__main__":
    validate()
```

**Deliverables:**
- `wav2vec2-base.mlpackage` (~200 MB)
- `vocab.json` (character vocabulary)
- Validation script confirming < 0.01 max diff

---

## Phase 2: Swift Integration (4 hours)

### 2.1 CoreML Wrapper

```swift
// Sources/Wav2Vec2Aligner/Wav2Vec2Model.swift
import CoreML
import Foundation

public class Wav2Vec2Model {
    private let model: MLModel
    private let vocab: [Int: String]
    private let blankId: Int = 0
    
    public init(modelPath: URL, vocabPath: URL) throws {
        // Load CoreML model
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Use ANE when available
        self.model = try MLModel(contentsOf: modelPath, configuration: config)
        
        // Load vocabulary
        let vocabData = try Data(contentsOf: vocabPath)
        let vocabDict = try JSONDecoder().decode([String: Int].self, from: vocabData)
        self.vocab = Dictionary(uniqueKeysWithValues: vocabDict.map { ($0.value, $0.key) })
    }
    
    /// Get frame-level log probabilities for audio
    /// - Parameter audio: 16kHz mono audio samples
    /// - Returns: Log probabilities [frames, vocab_size]
    public func getFrameLogProbs(_ audio: [Float]) throws -> [[Float]] {
        // Create MLMultiArray input
        let inputArray = try MLMultiArray(shape: [1, NSNumber(value: audio.count)], dataType: .float32)
        let ptr = inputArray.dataPointer.bindMemory(to: Float.self, capacity: audio.count)
        for (i, sample) in audio.enumerated() {
            ptr[i] = sample
        }
        
        // Run inference
        let input = try MLDictionaryFeatureProvider(dictionary: ["audio": inputArray])
        let output = try model.prediction(from: input)
        
        // Extract logits
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw Wav2Vec2Error.inferenceFailure
        }
        
        // Convert to log probabilities (log_softmax)
        return logSoftmax(logits)
    }
    
    private func logSoftmax(_ logits: MLMultiArray) -> [[Float]] {
        let frames = logits.shape[1].intValue
        let vocabSize = logits.shape[2].intValue
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: frames * vocabSize)
        
        var result: [[Float]] = []
        for t in 0..<frames {
            var row = [Float](repeating: 0, count: vocabSize)
            var maxVal: Float = -.infinity
            
            // Find max for numerical stability
            for v in 0..<vocabSize {
                let val = ptr[t * vocabSize + v]
                if val > maxVal { maxVal = val }
            }
            
            // Compute log_softmax
            var sumExp: Float = 0
            for v in 0..<vocabSize {
                sumExp += exp(ptr[t * vocabSize + v] - maxVal)
            }
            let logSumExp = maxVal + log(sumExp)
            
            for v in 0..<vocabSize {
                row[v] = ptr[t * vocabSize + v] - logSumExp
            }
            result.append(row)
        }
        return result
    }
}

enum Wav2Vec2Error: Error {
    case inferenceFailure
    case alignmentFailure
}
```

### 2.2 CTC Forced Alignment

```swift
// Sources/Wav2Vec2Aligner/CTCForcedAlign.swift
import Foundation

public struct AlignedToken {
    public let token: String
    public let tokenId: Int
    public let startFrame: Int
    public let endFrame: Int
    
    public var startTime: Float { Float(startFrame) * 0.02 }  // 20ms per frame
    public var endTime: Float { Float(endFrame) * 0.02 }
}

public struct CTCForcedAligner {
    private let blankId: Int
    
    public init(blankId: Int = 0) {
        self.blankId = blankId
    }
    
    /// Align known text to frame-level probabilities using CTC forced alignment
    /// - Parameters:
    ///   - logProbs: Frame-level log probabilities [T, V]
    ///   - tokens: Known text as token IDs (without blanks)
    /// - Returns: Aligned tokens with frame boundaries
    public func align(logProbs: [[Float]], tokens: [Int]) -> [AlignedToken] {
        guard !tokens.isEmpty, !logProbs.isEmpty else { return [] }
        
        let T = logProbs.count      // Number of frames
        let V = logProbs[0].count   // Vocab size
        
        // Build target sequence with blanks: _t1_t2_t3_
        var targets = [blankId]
        for t in tokens {
            targets.append(t)
            targets.append(blankId)
        }
        let S = targets.count
        
        // Check feasibility: need at least as many frames as targets
        guard T >= S else { return [] }
        
        // DP tables
        var alpha = [[Float]](repeating: [Float](repeating: -.infinity, count: S), count: T)
        var backptr = [[Int]](repeating: [Int](repeating: -1, count: S), count: T)
        
        // Initialize first frame
        alpha[0][0] = logProbs[0][targets[0]]
        if S > 1 {
            alpha[0][1] = logProbs[0][targets[1]]
        }
        
        // Forward pass
        for t in 1..<T {
            for s in 0..<S {
                let emit = logProbs[t][targets[s]]
                var candidates: [(score: Float, prev: Int)] = []
                
                // Stay in same state
                if alpha[t-1][s] > -.infinity {
                    candidates.append((alpha[t-1][s], s))
                }
                
                // Transition from previous state
                if s > 0 && alpha[t-1][s-1] > -.infinity {
                    candidates.append((alpha[t-1][s-1], s-1))
                }
                
                // Skip blank (non-blank to different non-blank)
                if s > 1 && targets[s] != blankId && targets[s] != targets[s-2] {
                    if alpha[t-1][s-2] > -.infinity {
                        candidates.append((alpha[t-1][s-2], s-2))
                    }
                }
                
                // Select best
                if let best = candidates.max(by: { $0.score < $1.score }) {
                    alpha[t][s] = best.score + emit
                    backptr[t][s] = best.prev
                }
            }
        }
        
        // Find best final state (must end in last token or final blank)
        var bestFinalState = S - 1
        if S > 1 && alpha[T-1][S-2] > alpha[T-1][S-1] {
            bestFinalState = S - 2
        }
        
        // Backtrace
        var path: [(frame: Int, state: Int)] = []
        var s = bestFinalState
        for t in stride(from: T-1, through: 0, by: -1) {
            path.append((t, s))
            if t > 0 {
                s = backptr[t][s]
            }
        }
        path.reverse()
        
        // Convert path to token alignments (skip blanks)
        var alignedTokens: [AlignedToken] = []
        var currentToken: Int? = nil
        var startFrame = 0
        
        for (frame, state) in path {
            let tokenId = targets[state]
            
            if tokenId != blankId {
                if currentToken == nil {
                    // Start new token
                    currentToken = tokenId
                    startFrame = frame
                } else if tokenId != currentToken {
                    // End previous token, start new one
                    alignedTokens.append(AlignedToken(
                        token: "", // Fill in later with vocab lookup
                        tokenId: currentToken!,
                        startFrame: startFrame,
                        endFrame: frame - 1
                    ))
                    currentToken = tokenId
                    startFrame = frame
                }
            } else if currentToken != nil {
                // Blank after token - might end the token
                // (Keep going until we see a different token)
            }
        }
        
        // Don't forget the last token
        if let lastToken = currentToken {
            alignedTokens.append(AlignedToken(
                token: "",
                tokenId: lastToken,
                startFrame: startFrame,
                endFrame: T - 1
            ))
        }
        
        return alignedTokens
    }
}
```

### 2.3 Text Tokenization

```swift
// Sources/Wav2Vec2Aligner/Wav2Vec2Tokenizer.swift
import Foundation

public class Wav2Vec2Tokenizer {
    private let charToId: [Character: Int]
    private let idToChar: [Int: Character]
    private let blankId: Int = 0
    private let padId: Int
    private let unkId: Int
    
    public init(vocabPath: URL) throws {
        let data = try Data(contentsOf: vocabPath)
        let vocab = try JSONDecoder().decode([String: Int].self, from: data)
        
        var c2i: [Character: Int] = [:]
        var i2c: [Int: Character] = [:]
        
        for (char, id) in vocab {
            if char.count == 1 {
                c2i[char.first!] = id
                i2c[id] = char.first!
            }
        }
        
        self.charToId = c2i
        self.idToChar = i2c
        self.padId = vocab["<pad>"] ?? 0
        self.unkId = vocab["<unk>"] ?? 1
    }
    
    /// Tokenize text to character IDs
    public func encode(_ text: String) -> [Int] {
        let normalized = text.uppercased()  // wav2vec2 uses uppercase
        return normalized.map { charToId[$0] ?? unkId }
    }
    
    /// Convert token IDs back to text
    public func decode(_ ids: [Int]) -> String {
        return String(ids.compactMap { idToChar[$0] })
    }
}
```

### 2.4 High-Level API

```swift
// Sources/Wav2Vec2Aligner/Wav2Vec2Aligner.swift
import Foundation

public struct WordAlignment {
    public let word: String
    public let startTime: Float
    public let endTime: Float
}

public class Wav2Vec2Aligner {
    private let model: Wav2Vec2Model
    private let tokenizer: Wav2Vec2Tokenizer
    private let ctcAligner: CTCForcedAligner
    
    public init(modelPath: URL, vocabPath: URL) throws {
        self.model = try Wav2Vec2Model(modelPath: modelPath, vocabPath: vocabPath)
        self.tokenizer = try Wav2Vec2Tokenizer(vocabPath: vocabPath)
        self.ctcAligner = CTCForcedAligner()
    }
    
    /// Align known text to audio, returning word-level timestamps
    /// - Parameters:
    ///   - audio: 16kHz mono audio samples
    ///   - text: Known transcription text
    /// - Returns: Word-level alignments
    public func align(audio: [Float], text: String) throws -> [WordAlignment] {
        // 1. Get frame-level probabilities from wav2vec2
        let logProbs = try model.getFrameLogProbs(audio)
        
        // 2. Tokenize text to characters
        let tokens = tokenizer.encode(text)
        
        // 3. Run CTC forced alignment
        let charAlignments = ctcAligner.align(logProbs: logProbs, tokens: tokens)
        
        // 4. Group characters into words
        return groupIntoWords(charAlignments: charAlignments, text: text)
    }
    
    private func groupIntoWords(charAlignments: [AlignedToken], text: String) -> [WordAlignment] {
        let words = text.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
        var wordAlignments: [WordAlignment] = []
        
        var charIndex = 0
        for word in words {
            let wordLength = word.count
            guard charIndex + wordLength <= charAlignments.count else { break }
            
            let startTime = charAlignments[charIndex].startTime
            let endTime = charAlignments[charIndex + wordLength - 1].endTime
            
            wordAlignments.append(WordAlignment(
                word: word,
                startTime: startTime,
                endTime: endTime
            ))
            
            charIndex += wordLength
            // Skip space character if present in alignment
            if charIndex < charAlignments.count {
                charIndex += 1
            }
        }
        
        return wordAlignments
    }
}
```

**Deliverables:**
- `Wav2Vec2Model.swift` — CoreML wrapper
- `CTCForcedAlign.swift` — Alignment algorithm
- `Wav2Vec2Tokenizer.swift` — Text tokenization
- `Wav2Vec2Aligner.swift` — High-level API

---

## Phase 3: Integration & Benchmarking (4 hours)

### 3.1 Add to Benchmark CLI

```swift
// benchmarks/cli/WhisperBenchmarkCLI.swift (additions)

case "wav2vec2-align":
    try await runWav2Vec2AlignCompare(audioFile: audioFile, baselineFile: baselineFile)

static func runWav2Vec2AlignCompare(audioFile: String, baselineFile: String?) async throws {
    // 1. Transcribe with Whisper (no word timestamps)
    let whisperResult = try await whisperEngine.transcribeBatched(audioURL, timestamps: .none)
    
    // 2. Align with wav2vec2
    let aligner = try Wav2Vec2Aligner(
        modelPath: URL(fileURLWithPath: "models/wav2vec2-base.mlpackage"),
        vocabPath: URL(fileURLWithPath: "models/vocab.json")
    )
    let wordAlignments = try aligner.align(audio: samples, text: whisperResult.text)
    
    // 3. Compare with Python baseline
    // ... (same comparison logic as other benchmarks)
}
```

### 3.2 Expected Results

| Metric | DTW (current) | wav2vec2 (expected) |
|--------|---------------|---------------------|
| Start MAE | 91 ms | 30-50 ms |
| End MAE | 105 ms | 40-60 ms |
| Match Rate | 82% | 85-90% |
| Alignment RTF | 0.024 | 0.005-0.010 |

### 3.3 Validation Checklist

- [ ] CoreML model loads without errors
- [ ] Frame probabilities match PyTorch reference (< 0.01 diff)
- [ ] CTC alignment produces monotonic timestamps
- [ ] Word grouping handles punctuation correctly
- [ ] Performance meets RTF target (< 0.02)
- [ ] MAE improves over DTW baseline

---

## Phase 4: Optimization (Optional, 2 hours)

### 4.1 Quantization

```python
# INT8 quantization for faster ANE inference
mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(
    mlmodel, 
    nbits=8,
    quantization_mode="linear"
)
```

### 4.2 Chunked Processing

For audio > 30s, process in overlapping chunks:

```swift
func alignLongAudio(audio: [Float], text: String, chunkSize: Int = 16000 * 30) -> [WordAlignment] {
    let overlap = 16000 * 2  // 2 second overlap
    var allAlignments: [WordAlignment] = []
    
    var offset = 0
    while offset < audio.count {
        let end = min(offset + chunkSize, audio.count)
        let chunk = Array(audio[offset..<end])
        
        // Align chunk
        let chunkAlignments = try aligner.align(audio: chunk, text: chunkText)
        
        // Adjust timestamps and merge
        for var alignment in chunkAlignments {
            alignment.startTime += Float(offset) / 16000
            alignment.endTime += Float(offset) / 16000
            allAlignments.append(alignment)
        }
        
        offset += chunkSize - overlap
    }
    
    return mergeOverlappingAlignments(allAlignments)
}
```

---

## Files to Create

```
mlx-swift-audio/
├── scripts/
│   ├── export_wav2vec2_onnx.py
│   ├── convert_wav2vec2_coreml.py
│   └── validate_wav2vec2_coreml.py
├── models/
│   ├── wav2vec2-base.mlpackage/
│   └── vocab.json
├── Sources/Wav2Vec2Aligner/
│   ├── Wav2Vec2Model.swift
│   ├── Wav2Vec2Tokenizer.swift
│   ├── CTCForcedAlign.swift
│   └── Wav2Vec2Aligner.swift
└── benchmarks/cli/
    └── WhisperBenchmarkCLI.swift (additions)
```

---

## Success Criteria

1. **Accuracy**: Start MAE < 60ms (vs 91ms current)
2. **Speed**: Alignment RTF < 0.02 (vs 0.024 current)
3. **Reliability**: > 85% word match rate
4. **Integration**: Clean API matching current `addWordTimestamps` signature

---

## Implementation Status (March 2026)

### Completed ✅
- Model conversion scripts (`scripts/convert_wav2vec2_coreml.py`, `export_wav2vec2_onnx.py`, `validate_wav2vec2_coreml.py`)
- Swift implementation (`package/Wav2Vec2Aligner/`):
  - `Wav2Vec2Model.swift` - CoreML wrapper
  - `CTCForcedAlign.swift` - CTC alignment algorithm
  - `Wav2Vec2Tokenizer.swift` - Character tokenization
  - `Wav2Vec2Aligner.swift` - High-level API
- `Wav2Vec2AlignerTests.swift` - Unit tests (all pass)
- Critical bug fix: MLMultiArray stride-based indexing in logSoftmax

### Blocked ❌

#### Issue 1: Metal Context Segfault
- **Segfault (exit 139)** during static initialization when importing Wav2Vec2Aligner
- Crash occurs before main() executes, during module load phase
- Unit tests work via xcodebuild, but standalone benchmark executable crashes
- Root cause: CoreML and MLX Metal context conflict during static initialization

#### Issue 2: CoreML Model Numerical Corruption (March 11, 2026)
- **ONNX → PyTorch validation**: ✅ PASSED (max diff: 0.000042)
- **CoreML model validation**: ❌ FAILED (numerical corruption)

**CoreML vs PyTorch output comparison (1 second test audio):**

| Metric | CoreML | PyTorch (Reference) | Error Factor |
|--------|--------|-------------------|--------------|
| Shape | [1, 49, 32] | [1, 49, 32] | ✅ Correct |
| Min | -42,012,168 | -16.71 | 2.5M× wrong |
| Max | 0.000043 | 4.03 | Completely wrong |
| Mean | -2,587,863 | -2.62 | 1M× wrong |
| Std | 10,023,029 | 4.69 | 2M× wrong |

**Root cause**: ONNX → CoreML conversion produces numerically incorrect outputs. Possible causes:
- coremltools version incompatibility (tested with Python 3.14 + coremltools 8.3.0)
- Model operations not correctly translated to CoreML
- Conversion settings or quantization issues

**Impact**: Subprocess CoreML workaround (to avoid Metal conflict) is not viable because the CoreML model itself produces incorrect results.

### Next Options
1. **Accept DTW**: Use DTW for alignment (91ms MAE meets < 100ms target) ✅ **RECOMMENDED**
2. **Fix CoreML conversion**: Requires debugging coremltools, may need older Python/coremltools version
3. **Debug MLX model**: Continue numerical debugging of MLX port (see `WAV2VEC2_MLX_PLAN.md`)
4. **Hybrid approach**: Whisper timestamps + DTW refinement (already working well)

---

## Test Scripts Added

- `scripts/test_coreml_model.swift` - Swift validation script for CoreML model
- `scripts/validate_wav2vec2_coreml.py` - Python validation script (requires working coremltools)
