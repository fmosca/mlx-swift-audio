# Swift MLX Whisper Optimization Plan

**Goal:** Achieve RTF ≤ 0.035 (30% faster than Python mlx-whisper baseline of 0.050)

**Baseline:** Python mlx-whisper large-v3-turbo-q4 on AMI ES2002a (21min)
- RTF: 0.0500
- Processing time: 63.6s

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

After each phase:

```bash
cd ~/Work/mlx-swift-audio/benchmarks

# Quick validation (~25s)
~/Work/meeting-transcriber/.venv/bin/python benchmark.py quick

# Full benchmark (~65s)
~/Work/meeting-transcriber/.venv/bin/python benchmark.py full

# Compare with previous
~/Work/meeting-transcriber/.venv/bin/python benchmark.py compare
```

### Success Criteria
| Phase | Expected RTF | Cumulative Improvement |
|-------|--------------|------------------------|
| Baseline | 0.0500 | - |
| Phase 1 | 0.0435 | ~13% |
| Phase 2 | 0.0400 | ~20% |
| Phase 3 | 0.0380 | ~24% |
| Target | ≤0.0350 | ≥30% |

---

---

## Phase 5: Hybrid Batched Decoding (Offline Path)

**Impact:** 2–4× improvement over sequential (batch_size=4–8)
**Effort:** High
**Risk:** Medium (correctness at segment boundaries)

### Problem

The fundamental GPU memory-bandwidth ceiling (~0.060 RTF) applies to single-sequence decoding: each decode step reads 1.5 GB of weights to produce **one** output token. Micro-optimisations cannot overcome this limit without changing the algorithm.

### Solution: Amortise Weight Reads Across a Batch

If B short segments are decoded in parallel, the same weight read produces B tokens — approaching an ideal B× throughput gain. The challenge is ensuring no words are cut at segment boundaries (which would happen with naive fixed-30 s chunking).

**Two-path strategy:**

1. **VAD pre-segmentation** (`WhisperVAD.swift`): energy-based VAD splits audio at natural silence boundaries (≥ 0.5 s gaps, < −40 dBFS). This guarantees segments start and end with silence — no mid-word cuts.

2. **Parallel path (segments ≤ 29 s)**: Short segments are packed into batches of `batchSize` (default 4). A single `model.compiledEncode` call processes `[B, nFrames, nMels]`, and `BatchedGreedyDecoder.decodeBatch` runs the autoregressive loop on all B sequences simultaneously. Because `conditionOnPreviousText = false`, the SOT prefix is identical for every sequence, keeping all KV caches in sync throughout the loop.

3. **Sequential path (segments > 29 s)**: Long segments containing uninterrupted speech fall through to the standard single-sequence decoder (30-second windowed pass). Whisper's intelligent seek is not available in this path, but VAD ensures this case is rare in typical meeting audio.

### Architecture

**New files:**
- `WhisperVAD.swift`: `energyVAD()` + `findSpeechSegments()` + `SpeechSegment` struct

**Modified files:**
- `WhisperDecoding.swift`: `BatchedGreedyDecoder` class appended (uses file-private compiled functions)
- `WhisperSTT.swift`: `transcribeBatched()` method

**Key design decisions:**
- `BatchedGreedyDecoder` lives in the same file as `GreedyDecoder` to share file-private compiled GPU kernels (`compiledTimestampRules`, `compiledTimestampForce`, `compiledArgmax`, `compiledApplyMask`).
- One GPU forward pass per iteration for all B sequences; one `eval()` sync for `[B, vocab_size]` logits; then B cheap CPU mask operations.
- Per-sequence O(1) timestamp state tracking (identical logic to `GreedyDecoder`).
- Temperature fallback is NOT performed in the batched path — caller is responsible for quality filtering. This avoids the complexity of restarting individual sequences within a batch.
- The VAD pre-pass adds < 1 ms overhead for typical meeting audio.

### Usage

```swift
let result = await stt.transcribeBatched(
    audio: audio,
    language: "en",
    task: .transcribe,
    temperature: 0.0,
    timestamps: .segment,
    batchSize: 4        // tune: 4 for turbo-q4, 8 if VRAM allows
)
```

### Expected Performance

With batch_size=4 on `large-v3-turbo-q4` and clean meeting audio (VAD produces mostly short segments):
- Sequential RTF: ~0.060
- Batched RTF target: ~0.020–0.030 (2–3×)
- Batched RTF max theoretical: 0.060 / 4 = 0.015

Actual gain depends on the fraction of audio in short vs. long segments and the per-step mask overhead.

### Limitations / Future Work

- Temperature fallback per batch element (retry failed sequences individually)
- Silero VAD or Apple Speech framework VAD for higher accuracy in noisy audio
- `compiledEncode` retrace penalty on first batch: the compiled encoder retraces for `[B, nFrames, nMels]` vs. `[1, nFrames, nMels]`. After the first batch call the new shape is cached.

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
