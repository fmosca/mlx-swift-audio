# wav2vec2 Forced Aligner — MLX Native Port Plan

## Goal

Implement wav2vec2 forced alignment natively in MLX Swift, achieving:
- ~30-50ms MAE (vs 91ms DTW current)
- Unified MLX runtime (no CoreML dependency)
- Full control over optimization (quantization, batching, compilation)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ALL-MLX PIPELINE                              │
│                                                                  │
│  Audio ──► Whisper (MLX) ──► Text                                │
│    │                          │                                  │
│    │                          ▼                                  │
│    └──► wav2vec2 (MLX) ──► Frame Probs ──► CTC Align ──► Words   │
│                                                                  │
│         Both models share GPU memory, no runtime switching       │
└─────────────────────────────────────────────────────────────────┘
```

**Benefits of MLX Native:**
- Single runtime, shared GPU memory
- MLX `compile()` for fused operations
- Custom quantization (4-bit, 8-bit)
- Batched alignment across segments
- Full Swift type safety

**Tradeoffs:**
- More implementation effort
- Must manually port architecture
- Weight conversion required

---

## Phase 1: Architecture Implementation (2 days)

### 1.1 Model Architecture

wav2vec2 consists of:
1. **Feature Extractor**: 7 Conv1d layers (CNN)
2. **Feature Projection**: Linear layer
3. **Transformer Encoder**: 12 layers (base) or 24 layers (large)
4. **CTC Head**: Linear projection to vocabulary

```swift
// package/STT/Wav2Vec2/Wav2Vec2Config.swift
import Foundation

public struct Wav2Vec2Config: Codable {
    // Feature extractor
    let convDim: [Int]           // [512, 512, 512, 512, 512, 512, 512]
    let convKernel: [Int]        // [10, 3, 3, 3, 3, 2, 2]
    let convStride: [Int]        // [5, 2, 2, 2, 2, 2, 2]
    
    // Transformer
    let hiddenSize: Int          // 768 (base) or 1024 (large)
    let numHiddenLayers: Int     // 12 (base) or 24 (large)
    let numAttentionHeads: Int   // 12 (base) or 16 (large)
    let intermediateSize: Int    // 3072 (base) or 4096 (large)
    let hiddenDropout: Float     // 0.1
    let attentionDropout: Float  // 0.1
    
    // CTC
    let vocabSize: Int           // 32 for characters
    let padTokenId: Int          // 0
    let blankTokenId: Int        // 0
    
    public static let base = Wav2Vec2Config(
        convDim: [512, 512, 512, 512, 512, 512, 512],
        convKernel: [10, 3, 3, 3, 3, 2, 2],
        convStride: [5, 2, 2, 2, 2, 2, 2],
        hiddenSize: 768,
        numHiddenLayers: 12,
        numAttentionHeads: 12,
        intermediateSize: 3072,
        hiddenDropout: 0.1,
        attentionDropout: 0.1,
        vocabSize: 32,
        padTokenId: 0,
        blankTokenId: 0
    )
    
    public static let large = Wav2Vec2Config(
        convDim: [512, 512, 512, 512, 512, 512, 512],
        convKernel: [10, 3, 3, 3, 3, 2, 2],
        convStride: [5, 2, 2, 2, 2, 2, 2],
        hiddenSize: 1024,
        numHiddenLayers: 24,
        numAttentionHeads: 16,
        intermediateSize: 4096,
        hiddenDropout: 0.1,
        attentionDropout: 0.1,
        vocabSize: 32,
        padTokenId: 0,
        blankTokenId: 0
    )
}
```

### 1.2 Feature Extractor (CNN)

```swift
// package/STT/Wav2Vec2/Wav2Vec2FeatureExtractor.swift
import MLX
import MLXNN

/// wav2vec2 CNN feature extractor
/// Converts raw audio waveform to latent representations
public class Wav2Vec2FeatureExtractor: Module {
    let convLayers: [Conv1dLayerNorm]
    
    public init(config: Wav2Vec2Config) {
        var layers: [Conv1dLayerNorm] = []
        
        // First layer: audio -> first conv dim
        layers.append(Conv1dLayerNorm(
            inChannels: 1,
            outChannels: config.convDim[0],
            kernelSize: config.convKernel[0],
            stride: config.convStride[0],
            isFirstLayer: true
        ))
        
        // Remaining layers
        for i in 1..<config.convDim.count {
            layers.append(Conv1dLayerNorm(
                inChannels: config.convDim[i - 1],
                outChannels: config.convDim[i],
                kernelSize: config.convKernel[i],
                stride: config.convStride[i],
                isFirstLayer: false
            ))
        }
        
        self.convLayers = layers
        super.init()
    }
    
    public func callAsFunction(_ audio: MLXArray) -> MLXArray {
        // audio: [batch, samples] -> [batch, 1, samples]
        var x = audio.expandedDimensions(axis: 1)
        
        for layer in convLayers {
            x = layer(x)
        }
        
        // [batch, channels, frames] -> [batch, frames, channels]
        return x.transposed(0, 2, 1)
    }
}

/// Single conv layer with group normalization
class Conv1dLayerNorm: Module {
    let conv: Conv1d
    let layerNorm: LayerNorm?
    let isFirstLayer: Bool
    
    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, isFirstLayer: Bool) {
        self.conv = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0
        )
        
        // First layer uses group norm, rest use layer norm
        if isFirstLayer {
            self.layerNorm = nil  // Group norm handled separately
        } else {
            self.layerNorm = LayerNorm(dimensions: outChannels)
        }
        
        self.isFirstLayer = isFirstLayer
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = conv(x)
        
        if isFirstLayer {
            // Group normalization (groups = outChannels for wav2vec2)
            out = groupNorm(out, numGroups: out.shape[1])
        } else if let ln = layerNorm {
            // Layer norm over channel dimension
            // [batch, channels, frames] -> transpose -> norm -> transpose back
            out = out.transposed(0, 2, 1)
            out = ln(out)
            out = out.transposed(0, 2, 1)
        }
        
        return gelu(out)
    }
}

/// Group normalization
func groupNorm(_ x: MLXArray, numGroups: Int, eps: Float = 1e-5) -> MLXArray {
    let (batch, channels, frames) = (x.shape[0], x.shape[1], x.shape[2])
    let groupSize = channels / numGroups
    
    // Reshape: [batch, groups, groupSize, frames]
    var reshaped = x.reshaped([batch, numGroups, groupSize, frames])
    
    // Normalize over (groupSize, frames)
    let mean = MLX.mean(reshaped, axes: [2, 3], keepDims: true)
    let variance = MLX.variance(reshaped, axes: [2, 3], keepDims: true)
    reshaped = (reshaped - mean) / MLX.sqrt(variance + eps)
    
    // Reshape back
    return reshaped.reshaped([batch, channels, frames])
}
```

### 1.3 Transformer Encoder

```swift
// package/STT/Wav2Vec2/Wav2Vec2Encoder.swift
import MLX
import MLXNN

/// wav2vec2 transformer encoder
public class Wav2Vec2Encoder: Module {
    let featureProjection: Linear
    let posEmbedding: Wav2Vec2PositionalEncoding
    let layers: [Wav2Vec2EncoderLayer]
    let layerNorm: LayerNorm
    let config: Wav2Vec2Config
    
    public init(config: Wav2Vec2Config) {
        self.config = config
        
        // Project CNN features to hidden size
        self.featureProjection = Linear(config.convDim.last!, config.hiddenSize)
        
        // Convolutional positional encoding (wav2vec2 specific)
        self.posEmbedding = Wav2Vec2PositionalEncoding(hiddenSize: config.hiddenSize)
        
        // Transformer layers
        self.layers = (0..<config.numHiddenLayers).map { _ in
            Wav2Vec2EncoderLayer(config: config)
        }
        
        self.layerNorm = LayerNorm(dimensions: config.hiddenSize)
        
        super.init()
    }
    
    public func callAsFunction(_ features: MLXArray) -> MLXArray {
        // features: [batch, frames, convDim]
        var x = featureProjection(features)
        
        // Add positional encoding
        x = x + posEmbedding(x)
        
        // Transformer layers
        for layer in layers {
            x = layer(x)
        }
        
        return layerNorm(x)
    }
}

/// Single transformer encoder layer
class Wav2Vec2EncoderLayer: Module {
    let selfAttn: MultiHeadAttention
    let selfAttnLayerNorm: LayerNorm
    let feedForward: Wav2Vec2FeedForward
    let ffLayerNorm: LayerNorm
    
    init(config: Wav2Vec2Config) {
        self.selfAttn = MultiHeadAttention(
            dims: config.hiddenSize,
            numHeads: config.numAttentionHeads
        )
        self.selfAttnLayerNorm = LayerNorm(dimensions: config.hiddenSize)
        self.feedForward = Wav2Vec2FeedForward(config: config)
        self.ffLayerNorm = LayerNorm(dimensions: config.hiddenSize)
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Self-attention with residual
        var out = selfAttnLayerNorm(x)
        out = selfAttn(out, out, out)
        out = x + out
        
        // Feed-forward with residual
        let ffOut = ffLayerNorm(out)
        out = out + feedForward(ffOut)
        
        return out
    }
}

/// Feed-forward network
class Wav2Vec2FeedForward: Module {
    let fc1: Linear
    let fc2: Linear
    
    init(config: Wav2Vec2Config) {
        self.fc1 = Linear(config.hiddenSize, config.intermediateSize)
        self.fc2 = Linear(config.intermediateSize, config.hiddenSize)
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = fc1(x)
        out = gelu(out)
        return fc2(out)
    }
}

/// Convolutional positional encoding (wav2vec2 specific)
class Wav2Vec2PositionalEncoding: Module {
    let conv: Conv1d
    
    init(hiddenSize: Int, kernelSize: Int = 128, groups: Int = 16) {
        // Grouped 1D convolution for positional encoding
        self.conv = Conv1d(
            inputChannels: hiddenSize,
            outputChannels: hiddenSize,
            kernelSize: kernelSize,
            stride: 1,
            padding: kernelSize / 2,
            groups: groups
        )
        super.init()
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, frames, hidden]
        // Conv1d expects [batch, channels, length]
        var out = x.transposed(0, 2, 1)
        out = conv(out)
        // Remove extra frame from padding
        out = out[0..., 0..., 0..<x.shape[1]]
        out = gelu(out)
        return out.transposed(0, 2, 1)
    }
}
```

### 1.4 Full Model with CTC Head

```swift
// package/STT/Wav2Vec2/Wav2Vec2ForCTC.swift
import MLX
import MLXNN

/// Complete wav2vec2 model for CTC-based forced alignment
public class Wav2Vec2ForCTC: Module {
    let featureExtractor: Wav2Vec2FeatureExtractor
    let encoder: Wav2Vec2Encoder
    let ctcHead: Linear
    let config: Wav2Vec2Config
    
    public init(config: Wav2Vec2Config) {
        self.config = config
        self.featureExtractor = Wav2Vec2FeatureExtractor(config: config)
        self.encoder = Wav2Vec2Encoder(config: config)
        self.ctcHead = Linear(config.hiddenSize, config.vocabSize)
        super.init()
    }
    
    /// Forward pass returning log probabilities
    /// - Parameter audio: Raw audio [batch, samples] at 16kHz
    /// - Returns: Log probabilities [batch, frames, vocabSize]
    public func callAsFunction(_ audio: MLXArray) -> MLXArray {
        // CNN feature extraction
        let features = featureExtractor(audio)
        
        // Transformer encoding
        let encoded = encoder(features)
        
        // CTC projection
        let logits = ctcHead(encoded)
        
        // Log softmax for CTC
        return MLX.logSoftmax(logits, axis: -1)
    }
    
    /// Calculate output frame count for given input samples
    public func outputFrames(forSamples samples: Int) -> Int {
        var frames = samples
        for (kernel, stride) in zip(config.convKernel, config.convStride) {
            frames = (frames - kernel) / stride + 1
        }
        return frames
    }
}
```

**Deliverables:**
- `Wav2Vec2Config.swift`
- `Wav2Vec2FeatureExtractor.swift`
- `Wav2Vec2Encoder.swift`
- `Wav2Vec2ForCTC.swift`

---

## Phase 2: Weight Loading (4 hours)

### 2.1 Weight Mapping

HuggingFace weights use different naming conventions. Create a mapping:

```swift
// package/STT/Wav2Vec2/Wav2Vec2WeightLoader.swift
import Foundation
import MLX

public enum Wav2Vec2WeightLoader {
    
    /// Load weights from HuggingFace safetensors format
    public static func load(
        model: Wav2Vec2ForCTC,
        from directory: URL
    ) throws {
        let weightsURL = directory.appendingPathComponent("model.safetensors")
        let weights = try MLX.loadArrays(url: weightsURL)
        
        // Feature extractor convolutions
        for (i, layer) in model.featureExtractor.convLayers.enumerated() {
            let prefix = "wav2vec2.feature_extractor.conv_layers.\(i)"
            layer.conv.weight = weights["\(prefix).conv.weight"]!
            layer.conv.bias = weights["\(prefix).conv.bias"]
            
            if i == 0 {
                // Group norm for first layer
                // (handled differently - may need gamma/beta)
            } else if let ln = layer.layerNorm {
                ln.weight = weights["\(prefix).layer_norm.weight"]
                ln.bias = weights["\(prefix).layer_norm.bias"]
            }
        }
        
        // Feature projection
        model.encoder.featureProjection.weight = weights["wav2vec2.feature_projection.projection.weight"]!
        model.encoder.featureProjection.bias = weights["wav2vec2.feature_projection.projection.bias"]
        
        // Positional encoding conv
        model.encoder.posEmbedding.conv.weight = weights["wav2vec2.encoder.pos_conv_embed.conv.weight"]!
        model.encoder.posEmbedding.conv.bias = weights["wav2vec2.encoder.pos_conv_embed.conv.bias"]
        
        // Transformer layers
        for (i, layer) in model.encoder.layers.enumerated() {
            let prefix = "wav2vec2.encoder.layers.\(i)"
            
            // Self attention
            layer.selfAttn.queryProj.weight = weights["\(prefix).attention.q_proj.weight"]!
            layer.selfAttn.queryProj.bias = weights["\(prefix).attention.q_proj.bias"]
            layer.selfAttn.keyProj.weight = weights["\(prefix).attention.k_proj.weight"]!
            layer.selfAttn.keyProj.bias = weights["\(prefix).attention.k_proj.bias"]
            layer.selfAttn.valueProj.weight = weights["\(prefix).attention.v_proj.weight"]!
            layer.selfAttn.valueProj.bias = weights["\(prefix).attention.v_proj.bias"]
            layer.selfAttn.outProj.weight = weights["\(prefix).attention.out_proj.weight"]!
            layer.selfAttn.outProj.bias = weights["\(prefix).attention.out_proj.bias"]
            
            // Layer norms
            layer.selfAttnLayerNorm.weight = weights["\(prefix).layer_norm.weight"]
            layer.selfAttnLayerNorm.bias = weights["\(prefix).layer_norm.bias"]
            layer.ffLayerNorm.weight = weights["\(prefix).final_layer_norm.weight"]
            layer.ffLayerNorm.bias = weights["\(prefix).final_layer_norm.bias"]
            
            // Feed forward
            layer.feedForward.fc1.weight = weights["\(prefix).feed_forward.intermediate_dense.weight"]!
            layer.feedForward.fc1.bias = weights["\(prefix).feed_forward.intermediate_dense.bias"]
            layer.feedForward.fc2.weight = weights["\(prefix).feed_forward.output_dense.weight"]!
            layer.feedForward.fc2.bias = weights["\(prefix).feed_forward.output_dense.bias"]
        }
        
        // Final layer norm
        model.encoder.layerNorm.weight = weights["wav2vec2.encoder.layer_norm.weight"]
        model.encoder.layerNorm.bias = weights["wav2vec2.encoder.layer_norm.bias"]
        
        // CTC head
        model.ctcHead.weight = weights["lm_head.weight"]!
        model.ctcHead.bias = weights["lm_head.bias"]
        
        // Evaluate to materialize
        eval(model.parameters())
    }
}
```

### 2.2 Model Download

```swift
// package/STT/Wav2Vec2/Wav2Vec2Model+Loading.swift
import Foundation

public extension Wav2Vec2ForCTC {
    
    /// Load pre-trained model from HuggingFace
    static func fromPretrained(
        modelId: String = "facebook/wav2vec2-base-960h",
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Wav2Vec2ForCTC {
        progressHandler?(0.0, "Downloading model...")
        
        // Determine config
        let config: Wav2Vec2Config = modelId.contains("large") ? .large : .base
        
        // Download weights
        let cacheDir = try await HuggingFaceDownloader.download(
            modelId: modelId,
            files: ["model.safetensors", "vocab.json", "config.json"],
            progressHandler: { progress in
                progressHandler?(progress * 0.9, "Downloading...")
            }
        )
        
        progressHandler?(0.9, "Loading weights...")
        
        // Create and load model
        let model = Wav2Vec2ForCTC(config: config)
        try Wav2Vec2WeightLoader.load(model: model, from: cacheDir)
        
        progressHandler?(1.0, "Ready")
        return model
    }
}
```

**Deliverables:**
- `Wav2Vec2WeightLoader.swift`
- `Wav2Vec2Model+Loading.swift`
- Verified weight loading for base model

---

## Phase 3: CTC Forced Alignment (4 hours)

### 3.1 CTC Alignment Algorithm

```swift
// package/STT/Wav2Vec2/CTCForcedAlignment.swift
import Foundation
import MLX

public struct AlignedCharacter: Sendable {
    public let character: Character
    public let tokenId: Int
    public let startFrame: Int
    public let endFrame: Int
    
    public var startTime: Float { Float(startFrame) * 0.02 }  // 20ms per frame
    public var endTime: Float { Float(endFrame) * 0.02 }
}

public struct AlignedWord: Sendable {
    public let word: String
    public let startTime: Float
    public let endTime: Float
}

/// CTC Forced Alignment using Viterbi-style dynamic programming
public class CTCForcedAligner {
    private let blankId: Int
    
    public init(blankId: Int = 0) {
        self.blankId = blankId
    }
    
    /// Align character tokens to audio frames
    /// - Parameters:
    ///   - logProbs: Frame-level log probabilities [frames, vocabSize]
    ///   - tokens: Character token IDs (without blanks)
    /// - Returns: Character alignments with frame boundaries
    public func alignCharacters(logProbs: MLXArray, tokens: [Int]) -> [AlignedCharacter] {
        let T = logProbs.shape[0]  // frames
        let V = logProbs.shape[1]  // vocab size
        
        // Build target sequence with blanks: _c1_c2_c3_
        var targets = [blankId]
        for t in tokens {
            targets.append(t)
            targets.append(blankId)
        }
        let S = targets.count
        
        guard T >= S else {
            print("Warning: Not enough frames (\(T)) for targets (\(S))")
            return []
        }
        
        // Transfer log probs to CPU for DP
        let logProbsArray = logProbs.asArray(Float.self)
        
        // DP tables
        var alpha = [[Float]](repeating: [Float](repeating: -.infinity, count: S), count: T)
        var backptr = [[Int]](repeating: [Int](repeating: -1, count: S), count: T)
        
        // Helper to access logProbs
        func prob(_ t: Int, _ v: Int) -> Float {
            return logProbsArray[t * V + v]
        }
        
        // Initialize
        alpha[0][0] = prob(0, targets[0])
        if S > 1 {
            alpha[0][1] = prob(0, targets[1])
        }
        
        // Forward pass
        for t in 1..<T {
            for s in 0..<S {
                let emit = prob(t, targets[s])
                var best: Float = -.infinity
                var bestPtr = -1
                
                // Stay in same state
                if alpha[t-1][s] > -.infinity {
                    best = alpha[t-1][s]
                    bestPtr = s
                }
                
                // From previous state
                if s > 0 && alpha[t-1][s-1] > best {
                    best = alpha[t-1][s-1]
                    bestPtr = s - 1
                }
                
                // Skip blank (non-blank to different non-blank)
                if s > 1 && targets[s] != blankId && targets[s] != targets[s-2] {
                    if alpha[t-1][s-2] > best {
                        best = alpha[t-1][s-2]
                        bestPtr = s - 2
                    }
                }
                
                if best > -.infinity {
                    alpha[t][s] = best + emit
                    backptr[t][s] = bestPtr
                }
            }
        }
        
        // Find best final state
        var bestFinal = S - 1
        if S > 1 && alpha[T-1][S-2] > alpha[T-1][S-1] {
            bestFinal = S - 2
        }
        
        guard alpha[T-1][bestFinal] > -.infinity else {
            print("Warning: No valid alignment path found")
            return []
        }
        
        // Backtrace
        var path: [(frame: Int, state: Int)] = []
        var s = bestFinal
        for t in stride(from: T-1, through: 0, by: -1) {
            path.append((t, s))
            if t > 0 && backptr[t][s] >= 0 {
                s = backptr[t][s]
            }
        }
        path.reverse()
        
        // Convert to character alignments
        return extractCharacterAlignments(path: path, targets: targets, tokens: tokens)
    }
    
    private func extractCharacterAlignments(
        path: [(frame: Int, state: Int)],
        targets: [Int],
        tokens: [Int]
    ) -> [AlignedCharacter] {
        var alignments: [AlignedCharacter] = []
        var currentTokenIdx: Int? = nil
        var startFrame = 0
        
        for (frame, state) in path {
            let tokenId = targets[state]
            
            // Map state to token index (states are: _t0_t1_t2_)
            let tokenIdx = (state - 1) / 2  // -1 for leading blank, /2 for blank interleaving
            
            if tokenId != blankId {
                if currentTokenIdx == nil {
                    currentTokenIdx = tokenIdx
                    startFrame = frame
                } else if tokenIdx != currentTokenIdx {
                    // End previous token
                    if let prevIdx = currentTokenIdx, prevIdx < tokens.count {
                        alignments.append(AlignedCharacter(
                            character: Character(UnicodeScalar(tokens[prevIdx])!),
                            tokenId: tokens[prevIdx],
                            startFrame: startFrame,
                            endFrame: frame - 1
                        ))
                    }
                    currentTokenIdx = tokenIdx
                    startFrame = frame
                }
            }
        }
        
        // Last token
        if let lastIdx = currentTokenIdx, lastIdx < tokens.count {
            alignments.append(AlignedCharacter(
                character: Character(UnicodeScalar(tokens[lastIdx])!),
                tokenId: tokens[lastIdx],
                startFrame: startFrame,
                endFrame: path.last?.frame ?? startFrame
            ))
        }
        
        return alignments
    }
    
    /// Group character alignments into word alignments
    public func groupIntoWords(
        charAlignments: [AlignedCharacter],
        text: String
    ) -> [AlignedWord] {
        let words = text.components(separatedBy: .whitespaces).filter { !$0.isEmpty }
        var wordAlignments: [AlignedWord] = []
        
        var charIdx = 0
        for word in words {
            guard charIdx < charAlignments.count else { break }
            
            let startTime = charAlignments[charIdx].startTime
            var endTime = startTime
            
            // Consume characters for this word
            for _ in word {
                if charIdx < charAlignments.count {
                    endTime = charAlignments[charIdx].endTime
                    charIdx += 1
                }
            }
            
            wordAlignments.append(AlignedWord(
                word: word,
                startTime: startTime,
                endTime: endTime
            ))
            
            // Skip space character if present
            if charIdx < charAlignments.count && charAlignments[charIdx].character == " " {
                charIdx += 1
            }
        }
        
        return wordAlignments
    }
}
```

### 3.2 Tokenizer

```swift
// package/STT/Wav2Vec2/Wav2Vec2Tokenizer.swift
import Foundation

/// Character-level tokenizer for wav2vec2 CTC models
public class Wav2Vec2Tokenizer {
    private let charToId: [Character: Int]
    private let idToChar: [Int: Character]
    private let blankId: Int
    private let padId: Int
    private let unkId: Int
    private let spaceId: Int
    
    public init(vocabPath: URL) throws {
        let data = try Data(contentsOf: vocabPath)
        let vocab = try JSONDecoder().decode([String: Int].self, from: data)
        
        var c2i: [Character: Int] = [:]
        var i2c: [Int: Character] = [:]
        
        for (str, id) in vocab {
            if str == "<pad>" || str == "<s>" || str == "</s>" || str == "<unk>" {
                continue
            }
            if str == "|" {
                // Space character in wav2vec2 vocab
                c2i[" "] = id
                i2c[id] = " "
            } else if str.count == 1 {
                c2i[str.first!] = id
                i2c[id] = str.first!
            }
        }
        
        self.charToId = c2i
        self.idToChar = i2c
        self.blankId = vocab["<pad>"] ?? 0
        self.padId = vocab["<pad>"] ?? 0
        self.unkId = vocab["<unk>"] ?? 1
        self.spaceId = vocab["|"] ?? c2i[" "] ?? 4
    }
    
    /// Encode text to character token IDs
    public func encode(_ text: String) -> [Int] {
        // wav2vec2 uses uppercase
        let normalized = text.uppercased()
        return normalized.compactMap { char -> Int? in
            if let id = charToId[char] {
                return id
            } else if char == " " {
                return spaceId
            } else {
                return unkId
            }
        }
    }
    
    /// Decode token IDs to text
    public func decode(_ ids: [Int]) -> String {
        return String(ids.compactMap { idToChar[$0] })
    }
    
    public var blank: Int { blankId }
}
```

**Deliverables:**
- `CTCForcedAlignment.swift`
- `Wav2Vec2Tokenizer.swift`
- Unit tests for alignment algorithm

---

## Phase 4: High-Level API & Integration (1 day)

### 4.1 Unified Aligner API

```swift
// package/STT/Wav2Vec2/Wav2Vec2Aligner.swift
import Foundation
import MLX

/// High-level API for wav2vec2-based forced alignment
public class Wav2Vec2Aligner {
    private let model: Wav2Vec2ForCTC
    private let tokenizer: Wav2Vec2Tokenizer
    private let ctcAligner: CTCForcedAligner
    
    /// Compiled forward pass for performance
    private var compiledForward: ((MLXArray) -> MLXArray)?
    
    public init(model: Wav2Vec2ForCTC, tokenizer: Wav2Vec2Tokenizer) {
        self.model = model
        self.tokenizer = tokenizer
        self.ctcAligner = CTCForcedAligner(blankId: tokenizer.blank)
    }
    
    /// Load pre-trained aligner
    public static func fromPretrained(
        modelId: String = "facebook/wav2vec2-base-960h",
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Wav2Vec2Aligner {
        let model = try await Wav2Vec2ForCTC.fromPretrained(
            modelId: modelId,
            progressHandler: progressHandler
        )
        
        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
        let tokenizer = try Wav2Vec2Tokenizer(vocabPath: cacheDir.appendingPathComponent("vocab.json"))
        
        return Wav2Vec2Aligner(model: model, tokenizer: tokenizer)
    }
    
    /// Compile the forward pass for faster inference
    public func compile() {
        compiledForward = MLX.compile { [model] audio in
            model(audio)
        }
    }
    
    /// Align text to audio, returning word-level timestamps
    /// - Parameters:
    ///   - audio: 16kHz mono audio samples
    ///   - text: Known transcription
    /// - Returns: Word alignments with timestamps
    public func align(audio: [Float], text: String) -> [AlignedWord] {
        // Convert to MLXArray
        let audioArray = MLXArray(audio).expandedDimensions(axis: 0)
        
        // Forward pass
        let logProbs: MLXArray
        if let compiled = compiledForward {
            logProbs = compiled(audioArray)
        } else {
            logProbs = model(audioArray)
        }
        eval(logProbs)
        
        // Remove batch dimension: [1, frames, vocab] -> [frames, vocab]
        let logProbsUnbatched = logProbs.squeezed(axis: 0)
        
        // Tokenize text
        let tokens = tokenizer.encode(text)
        
        // CTC alignment
        let charAlignments = ctcAligner.alignCharacters(logProbs: logProbsUnbatched, tokens: tokens)
        
        // Group into words
        return ctcAligner.groupIntoWords(charAlignments: charAlignments, text: text)
    }
    
    /// Batch alignment for multiple segments
    public func alignBatch(
        audioSegments: [[Float]],
        texts: [String]
    ) -> [[AlignedWord]] {
        // TODO: Implement batched inference
        // For now, sequential
        return zip(audioSegments, texts).map { audio, text in
            align(audio: audio, text: text)
        }
    }
}
```

### 4.2 Integration with Whisper Pipeline

```swift
// package/STT/Whisper/WhisperSTT+Wav2Vec2.swift
import Foundation

extension WhisperSTT {
    
    /// Transcribe with wav2vec2-based word alignment (alternative to DTW)
    public func transcribeWithWav2Vec2Alignment(
        audio: MLXArray,
        language: String?,
        task: TranscriptionTask,
        aligner: Wav2Vec2Aligner
    ) async throws -> TranscriptionResult {
        // 1. Transcribe with Whisper (no word timestamps)
        let result = try await transcribe(
            audio: audio,
            language: language,
            task: task,
            temperature: 0.0,
            timestamps: .segment  // Segment-level only
        )
        
        // 2. Align each segment with wav2vec2
        var alignedSegments: [TranscriptionSegment] = []
        
        for segment in result.segments {
            // Extract audio for this segment
            let startSample = Int(segment.start * 16000)
            let endSample = min(Int(segment.end * 16000), audio.shape[0])
            let segmentAudio = audio[startSample..<endSample]
            
            // Align
            let wordAlignments = aligner.align(
                audio: segmentAudio.asArray(Float.self),
                text: segment.text
            )
            
            // Convert to Word format
            let words = wordAlignments.map { aligned in
                Word(
                    word: aligned.word,
                    start: TimeInterval(segment.start) + TimeInterval(aligned.startTime),
                    end: TimeInterval(segment.start) + TimeInterval(aligned.endTime),
                    probability: 1.0  // wav2vec2 doesn't provide word-level confidence
                )
            }
            
            alignedSegments.append(TranscriptionSegment(
                text: segment.text,
                start: segment.start,
                end: segment.end,
                tokens: segment.tokens,
                avgLogProb: segment.avgLogProb,
                noSpeechProb: segment.noSpeechProb,
                words: words
            ))
        }
        
        return TranscriptionResult(
            text: result.text,
            segments: alignedSegments,
            language: result.language
        )
    }
}
```

**Deliverables:**
- `Wav2Vec2Aligner.swift`
- `WhisperSTT+Wav2Vec2.swift`
- Benchmark CLI integration

---

## Phase 5: Optimization (Optional, 1 day)

### 5.1 Quantization

```swift
// 4-bit quantization for memory efficiency
let quantizedModel = try await Wav2Vec2ForCTC.fromPretrained(
    modelId: "facebook/wav2vec2-base-960h",
    quantization: .q4  // Custom quantization during weight loading
)
```

### 5.2 Compiled Kernels

```swift
// Compile critical path
let compiledAlign = MLX.compile { (audio: MLXArray) -> MLXArray in
    let features = featureExtractor(audio)
    let encoded = encoder(features)
    return ctcHead(encoded)
}
```

### 5.3 Batched CTC Alignment

For multiple segments, run CTC alignment in parallel using GCD:

```swift
func alignBatchParallel(segments: [(audio: [Float], text: String)]) -> [[AlignedWord]] {
    let queue = DispatchQueue(label: "ctc-align", attributes: .concurrent)
    var results = [[AlignedWord]?](repeating: nil, count: segments.count)
    
    DispatchQueue.concurrentPerform(iterations: segments.count) { i in
        let (audio, text) = segments[i]
        results[i] = align(audio: audio, text: text)
    }
    
    return results.compactMap { $0 }
}
```

---

## Files to Create

```
mlx-swift-audio/
├── package/STT/Wav2Vec2/
│   ├── Wav2Vec2Config.swift
│   ├── Wav2Vec2FeatureExtractor.swift
│   ├── Wav2Vec2Encoder.swift
│   ├── Wav2Vec2ForCTC.swift
│   ├── Wav2Vec2WeightLoader.swift
│   ├── Wav2Vec2Model+Loading.swift
│   ├── Wav2Vec2Tokenizer.swift
│   ├── CTCForcedAlignment.swift
│   └── Wav2Vec2Aligner.swift
├── package/STT/Whisper/
│   └── WhisperSTT+Wav2Vec2.swift (addition)
└── benchmarks/cli/
    └── WhisperBenchmarkCLI.swift (additions)
```

---

## Timeline Summary

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Architecture implementation | 2 days |
| 2 | Weight loading | 4 hours |
| 3 | CTC forced alignment | 4 hours |
| 4 | High-level API & integration | 1 day |
| 5 | Optimization (optional) | 1 day |
| **Total** | | **3-5 days** |

---

## Success Criteria

1. **Accuracy**: Start MAE < 60ms (target: 30-50ms)
2. **Speed**: Alignment RTF < 0.02 (comparable to CoreML)
3. **Memory**: < 500MB additional GPU memory
4. **Integration**: Drop-in replacement for DTW alignment
5. **Quality**: > 85% word match rate vs Python baseline

---

## Comparison: CoreML vs MLX

| Aspect | CoreML | MLX Native |
|--------|--------|------------|
| **Effort** | 1-2 days | 3-5 days |
| **Performance** | ANE-optimized | GPU-optimized |
| **Flexibility** | Fixed shapes | Dynamic |
| **Integration** | Separate runtime | Unified with Whisper |
| **Debugging** | Black box | Full visibility |
| **Quantization** | FP16 only | 4/8-bit available |

**Recommendation**: 
- Start with CoreML for quick validation
- If successful, consider MLX port for better long-term integration
