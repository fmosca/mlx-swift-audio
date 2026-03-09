# Swift MLX Whisper Optimization Plan

**Goal:** Swift implementation faster than Python mlx-whisper with equivalent output quality (word timestamps, condition_on_previous_text=False).

**Original baseline (sequential, no word timestamps):** Python mlx-whisper large-v3-turbo-q4 on AMI ES2002a
- RTF: ~0.050–0.055

**Current comparison point (word timestamps, condition_on_previous_text=False, AMI 5-min clip):**
| Implementation | RTF | Word Jaccard | Start MAE |
|---|---|---|---|
| Python mlx-whisper | 0.0596 | 1.0 (baseline) | — |
| Swift batched (Phase 5+6) | **0.0538** | 0.736 | 73 ms |

Swift batched is **10% faster** than Python with word-level timestamps.

---

## Phase 1: Async Eval Pipelining

**Impact:** ~10-15% improvement  
**Effort:** Medium  
**Risk:** Low

### Problem
Swift uses synchronous `eval()` in the decode loop, causing GPU idleness while CPU processes results. The GPU waits for CPU decision-making before starting the next forward pass.

### Solution
Implement `asyncEval()` pipelining similar to Python's `mx.async_eval()`:

```swift
// In GreedyDecoder.decode()
var nextTokens: MLXArray?
var pendingLogits: MLXArray?

for step in 0..<maxTokens {
    // Start next forward pass while processing current results
    if let pending = pendingLogits {
        // Process previous step's results (CPU work)
        let token = processLogits(pending)
        tokens.append(token)
        
        // Check stopping conditions
        if shouldStop(token) { break }
    }
    
    // Kick off next forward pass (GPU work)
    let logits = model.decode(tokens, audioFeatures: audioFeatures)
    asyncEval(logits)  // Don't block - let GPU work while we loop
    pendingLogits = logits
}
```

### Implementation Steps
1. Add helper to flatten Whisper KV cache for `asyncEval`:
   ```swift
   func flattenWhisperCache(_ cache: [[KVCache]]) -> [MLXArray] {
       cache.flatMap { layer in layer.flatMap { [$0.keys, $0.values] } }
   }
   ```
2. Modify `GreedyDecoder.decode()` to pipeline forward passes
3. Ensure cache arrays are included in `asyncEval` call
4. Benchmark to verify improvement

### Files to Modify
- `package/STT/Whisper/WhisperDecoding.swift`

---

## Phase 2: Compile Hot Paths

**Impact:** ~5-8% improvement  
**Effort:** Low  
**Risk:** Low

### Problem
Key computation blocks are interpreted rather than compiled:
- Timestamp probability heuristic
- Mask combination operations
- Argmax/sampling operations

### Solution
Use MLX Swift's `compile()` for hot paths:

```swift
// Pre-compiled argmax
private let compiledArgmax = compile { (logits: MLXArray) -> MLXArray in
    MLX.argMax(logits, axis: -1)
}

// Pre-compiled mask application
private let compiledMaskAndApply = compile { 
    (logits: MLXArray, mask: MLXArray) -> MLXArray in
    logits + mask
}

// Pre-compiled timestamp probability update
private let compiledTimestampProbUpdate = compile {
    (logprobs: MLXArray, textIndices: MLXArray, timestampIndices: MLXArray) -> (MLXArray, MLXArray) in
    let textLogprobs = logprobs.take(textIndices, axis: -1)
    let timestampLogprobs = logprobs.take(timestampIndices, axis: -1)
    return (MLX.logSumExp(textLogprobs, axis: -1), MLX.logSumExp(timestampLogprobs, axis: -1))
}
```

### Implementation Steps
1. Identify hot paths in `GreedyDecoder` via profiling
2. Create compiled function wrappers as `private let` properties
3. Replace inline computations with compiled versions
4. Benchmark to verify improvement

### Files to Modify
- `package/STT/Whisper/WhisperDecoding.swift`

---

## Phase 3: Decoder Reuse Across Temperature Fallbacks

**Impact:** ~3-5% improvement  
**Effort:** Low  
**Risk:** Low

### Problem
A new `GreedyDecoder` instance is created for each:
- Temperature fallback attempt
- 30-second seek window

This causes redundant initialization and re-encoding of audio features.

### Solution
Refactor to reuse decoder and cache audio encoding:

```swift
// In WhisperSTT.transcribe()
let decoder = GreedyDecoder(model: model, tokenizer: tokenizer)

for segment in segments {
    // Encode audio once per segment
    let audioFeatures = model.encoder(segmentMel)
    
    // Reuse decoder for temperature fallbacks
    for temperature in temperatures {
        let options = DecodingOptions(temperature: temperature, ...)
        let result = decoder.decode(audioFeatures: audioFeatures, options: options)
        
        if result.isAcceptable { break }
        decoder.reset()  // Clear KV cache, keep compiled functions
    }
}
```

### Implementation Steps
1. Add `reset()` method to `GreedyDecoder` to clear KV cache
2. Add `decode(audioFeatures:options:)` method that skips encoding
3. Modify `WhisperSTT.transcribe()` to reuse decoder instance
4. Benchmark with audio that triggers temperature fallback

### Files to Modify
- `package/STT/Whisper/WhisperDecoding.swift`
- `package/STT/Whisper/WhisperSTT.swift`

---

## Phase 4: Memory Optimization (Optional)

**Impact:** ~2-3% improvement  
**Effort:** Medium  
**Risk:** Medium

### Problem
Memory allocation/deallocation overhead during long transcriptions.

### Solution
- Pre-allocate token buffers
- Reuse mask arrays across decode steps
- Configure MLX memory cache limits

### Implementation Steps
1. Profile memory allocation patterns
2. Pre-allocate frequently used arrays
3. Tune `MLXMemory.configure(cacheLimit:)`

---

## Validation Workflow

```bash
# Build the benchmark CLI
cd ~/Work/mlx-swift-audio
xcodebuild -scheme WhisperBenchmark -destination "platform=macOS" build

BENCH=$(find ~/Library/Developer/Xcode/DerivedData/mlx-swift-audio-*/Build/Products/Debug/WhisperBenchmark -type f | head -1)

# Quick RTF check (no word timestamps, ~35 s)
$BENCH batch-quick

# Word timestamp comparison vs Python baseline (~60 s)
$BENCH word-compare

# Regenerate Python baseline (requires meeting-transcriber venv)
source ~/Work/meeting-transcriber/.venv/bin/activate
python benchmarks/scripts/word_baseline.py benchmarks/fixtures/ami_ES2002a_5min.wav
```

### Achieved Results (AMI ES2002a, 5 min, large-v3-turbo-q4)

| Metric | Python | Swift batched | Delta |
|---|---|---|---|
| RTF (w/ word timestamps) | 0.0596 | **0.0538** | Swift −10% |
| RTF (no word timestamps) | ~0.055 | **0.0551** | parity |
| Word Jaccard | 1.0 (ref) | 0.736 | −26% |
| Start MAE | — | 73 ms | good |
| Start p90 | — | 184 ms | good |

---

---

## Phase 5: Hybrid Batched Decoding (Offline Path)

**Status: Implemented and extended in Phase 6**

**Impact:** ~10% faster than Python mlx-whisper with full word timestamps (RTF 0.0538 vs 0.0596)
**Effort:** High

### Problem

The fundamental GPU memory-bandwidth ceiling (~0.060 RTF) applies to single-sequence decoding: each decode step reads 1.5 GB of weights to produce **one** output token. Micro-optimisations cannot overcome this limit without changing the algorithm.

### Solution: Amortise Weight Reads Across a Batch

If B short segments are decoded in parallel, the same weight read produces B tokens. VAD pre-segmentation at natural silence boundaries ensures no mid-word cuts.

**Two-path strategy:**

1. **VAD pre-segmentation** (Silero VAD via FluidAudio, CoreML/ANE-accelerated): splits audio at natural silence boundaries. Segments are then merged greedily up to 29 s to maximise context window utilisation.

2. **Parallel path (merged segments ≤ 29 s)**: Short segments are packed into batches of `batchSize` (default 4). A single `model.compiledEncode` call processes `[B, nFrames, nMels]`, and `BatchedGreedyDecoder.decodeBatch` runs the autoregressive loop on all B sequences simultaneously.

3. **Sequential path (segments > 29 s)**: Falls through to a windowed single-sequence decoder. VAD ensures this is rare in meeting audio.

### Architecture

**Key files:**
- `WhisperDecoding.swift`: `BatchedGreedyDecoder` class shares file-private compiled GPU kernels with `GreedyDecoder`
- `WhisperSTT.swift`: `transcribeBatched()` — VAD, merging, encode+decode loop, result assembly
- `WhisperEngine.swift`: public `transcribeBatched(_:)` API

**Key design decisions:**
- One GPU forward pass per iteration for all B sequences; one `eval()` sync; then B cheap CPU mask/token operations.
- Per-sequence O(1) timestamp state tracking mirrors `GreedyDecoder` exactly.
- Temperature fallback is NOT performed per batch element — quality filtering via gates (see Phase 6).
- `timestamps: .none` in the batched decode path — VAD provides boundaries, Whisper's internal timestamp tokens are redundant and add per-step GPU overhead.

### Usage

```swift
let result = try await engine.transcribeBatched(
    url,
    language: .english,
    timestamps: .word,
    batchSize: 4
)
```

### Achieved Performance (AMI 5-min, large-v3-turbo-q4)

| Mode | RTF |
|---|---|
| Swift sequential, no word timestamps | ~0.060 |
| Swift batched (B=4), no word timestamps | 0.0551 |
| Swift batched (B=4), with word timestamps | **0.0538** |
| Python mlx-whisper, with word timestamps | 0.0596 |

---

## Phase 6: Quality Gates, Inter-Batch Conditioning, Word Timestamps

**Status: Implemented**

### 6a: Word Timestamps in Batched Path

`addWordTimestamps` / `findAlignment` extended with an optional `audioFeatures: MLXArray?` parameter. When provided, the DTW alignment pass calls `model.decodeWithCrossQK(audioFeatures, tokens:)` — a decoder-only variant that skips the encoder entirely, reusing the features already computed during the decode step. This eliminated a full encoder re-run per segment, reducing the word-timestamp overhead from +67% to ~0% RTF.

### 6b: Inter-Batch Conditioning

`BatchedGreedyDecoder.decodeBatch` now accepts `options.prompt`, prepending `<|startofprev|>` + prompt tokens before the SOT sequence (identical logic to `GreedyDecoder.decode`). `transcribeBatched` carries `batchPrompt` across batches: each batch uses the previous batch's last segment tokens as context. This approximates `condition_on_previous_text` while preserving within-batch parallelism. Effect: Jaccard improved from 0.630 → 0.718.

### 6c: Quality Gates (from faster-whisper / WhisperX research)

| Gate | Implementation | Effect |
|---|---|---|
| Compression ratio > 2.4 | `shouldAccept` filter | Drops highly repetitive segments |
| `noRepeatNgramSize = 3` | `bannedNgramTokens` in `decodeBatch` per step | Prevents token-level repetitions during generation |
| `timestamps: .none` in batched path | `DecodingOptions` | Removes per-step timestamp rules GPU overhead (~10% RTF gain) |
| `hallucinationSilenceThreshold = 2.0 s` | Post-word-timestamp gate | Drops segments where DTW found a >2 s silence within VAD-marked speech |

### 6d: Benchmark Tooling

- `benchmarks/scripts/word_baseline.py`: generates Python word-level JSON baseline with `condition_on_previous_text=False`
- `word-compare` CLI mode: one-to-one word matching (±1.5 s window), Start MAE, End MAE, p90, signed bias

### Remaining Gap

Word Jaccard 0.736 vs Python 1.0 (same task). The residual gap has two sources:
1. Within a batch, all segments share the same prompt from the *previous* batch's last segment — segments earlier in the batch lack per-segment context
2. Minor model stochasticity at segment boundaries with no prior context for the first batch

WhisperX's approach (separate wav2vec2/MMS forced-alignment model replacing cross-attention DTW) would give better timing accuracy but requires a CoreML-compatible forced aligner, which does not currently exist for Apple platforms.

---

## Not Feasible

### Encode Once and Slice
Investigated encoding full audio mel spectrogram once and slicing for each 30-second window. **Not feasible** with current `AudioEncoder` architecture due to fixed-length positional embeddings (`n_audio_ctx = 1500`).

Would require model architecture changes to support variable-length positional encoding.

---

## References

- Python mlx-whisper: `~/.local/share/uv/python/lib/python3.11/site-packages/mlx_whisper/`
- MLX Swift compile(): `/tmp/mlx-swift/Source/MLX/Transforms.swift`
- MLX Swift asyncEval(): `/tmp/mlx-swift/Source/MLX/Transforms+Eval.swift`
