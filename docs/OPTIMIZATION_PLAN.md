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

## Not Feasible

### Encode Once and Slice
Investigated encoding full audio mel spectrogram once and slicing for each 30-second window. **Not feasible** with current `AudioEncoder` architecture due to fixed-length positional embeddings (`n_audio_ctx = 1500`).

Would require model architecture changes to support variable-length positional encoding.

---

## References

- Python mlx-whisper: `~/.local/share/uv/python/lib/python3.11/site-packages/mlx_whisper/`
- MLX Swift compile(): `/tmp/mlx-swift/Source/MLX/Transforms.swift`
- MLX Swift asyncEval(): `/tmp/mlx-swift/Source/MLX/Transforms+Eval.swift`
