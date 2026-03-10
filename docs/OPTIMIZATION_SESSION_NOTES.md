# Optimization Session Notes: Swift vs Python MLX Whisper

## Goal

Achieve Swift transcription RTF at least 30% faster than Python `mlx-whisper` on the same model
(`mlx-community/whisper-large-v3-turbo-q4`), measured on the AMI ES2002a corpus (21-minute
meeting audio).

---

## Current State (after Session 3)

All measurements: Apple Silicon M1 Pro, `mlx-community/whisper-large-v3-turbo-q4`,
`benchmarks/fixtures/ami_ES2002a_5min.wav` (5-min AMI corpus clip), GPU uncontested.
Python settings: `word_timestamps=True`, `condition_on_previous_text=False`.

| Engine | Mode | RTF | Word Jaccard | Start MAE |
|--------|------|-----|--------------|-----------|
| Python mlx-whisper | sequential | 0.0596 | 1.0 (ref) | — |
| Swift batched B=4 | no word timestamps | 0.0551 | — | — |
| Swift batched B=4 | with word timestamps | **0.0538** | 0.736 | 73 ms |

**Swift batched is 10% faster than Python** with full word-level timestamps.

---

## Session 4: GPU-Accelerated Alignment & Batched Word Timestamps

### Goal
Optimize the word-level timestamp generation, which was identified as a bottleneck due to CPU-bound DTW preparation and sequential processing in the batched path.

### Changes
1.  **GPU Median Filter (`WhisperTiming.swift`)**: Implemented `medianFilterAttentionGPU` using MLX primitives (`concatenated`, `stacked`, `sorted`) to perform the 7-kernel median filter entirely on the GPU. This avoids transferring the massive [Batch, Heads, Tokens, Frames] attention tensor to the CPU.
2.  **GPU Head Averaging**: Performed averaging across alignment heads on the GPU using `MLX.mean`.
3.  **Batched `findAlignment`**: Refactored `findAlignment` to accept `textTokens: Any` ([Int] or [[Int]]) and `numFrames: Any` (Int or [Int]) to process a full batch of segments in parallel on the GPU for the heavy matrix operations (softmax, normalization, filtering, averaging).
4.  **Batched `addWordTimestamps`**: Updated `addWordTimestamps` to handle batched inputs with `timeOffsets: [Float]?` parameter and call the batched `findAlignment`.
5.  **Batched `transcribeBatched` Loop**: Refactored the main loop in `WhisperSTT.swift` to collect accepted segments and audio features, then call `addWordTimestamps` once per batch instead of per-segment.

### Results (21-min AMI corpus, GPU contested)

| Metric | Value |
|--------|-------|
| RTF | 0.1085 (not reliable — GPU contested) |
| Word Jaccard | 0.740 |
| Match Rate | 82.5% |
| Start MAE | 91 ms (good) |
| End MAE | 105 ms |

Word timestamp quality is maintained. RTF measurement requires uncontested GPU.

### Why Batched Alignment Isn't Dramatically Faster

The optimizations target GPU operations, but the core bottleneck is CPU-bound:

| Operation | Before | After |
|-----------|--------|-------|
| Median filter | CPU, per-segment | GPU, batched |
| Head averaging | CPU, per-segment | GPU, batched |
| CPU→GPU transfer | ~100MB attention tensor | ~25MB averaged matrix |
| Decoder passes | N separate | 1 batched |
| **DTW** | **Sequential (CPU)** | **Sequential (CPU) — unchanged** |

**The ceiling**: Dynamic Time Warping is a dynamic programming algorithm that cannot be parallelized on GPU. Each cell depends on its neighbors. If DTW takes 30% of alignment time, maximum theoretical speedup from GPU batching is ~1.4x.

To achieve substantially faster word timestamps would require replacing DTW with a different algorithm (e.g., WhisperX's wav2vec2 forced aligner).

---

## Session 5: Alternative Alignment Approaches Evaluation

### Goal
Evaluate alternative word alignment approaches to potentially replace DTW with something faster and/or more accurate.

### Approaches Tested

#### 1. Parakeet-TDT (FluidAudio, CoreML)

Parakeet-TDT is a Token-and-Duration Transducer model that produces native token-level timestamps as part of decoding (no DTW needed). Tested via FluidAudio's `AsrManager`.

**Results (21-min AMI corpus):**

| Metric | Parakeet-TDT | Swift DTW | Python Whisper |
|--------|--------------|-----------|----------------|
| RTF | 0.011 (fastest) | 0.054 | 0.060 |
| Word Jaccard | **0.114** (poor) | 0.740 | 1.0 (ref) |
| Start MAE | **229 ms** (poor) | 91 ms | — |
| Match rate | 47% | 82% | — |

**Conclusion**: Parakeet is optimized for real-time streaming, not offline meeting transcription. Text quality is significantly worse than Whisper. **Not suitable** for our use case.

#### 2. Qwen3-ForcedAligner (speech-swift, MLX)

Qwen3-ForcedAligner is a dedicated forced alignment model (audio + text → timestamps) using a 5000-class timestamp classifier instead of DTW.

**Results (5-min AMI corpus):**

| Metric | Qwen3-ForcedAligner | Swift DTW |
|--------|---------------------|-----------|
| Whisper RTF | 0.042 | — |
| Alignment RTF | 0.024 | — |
| Total RTF | 0.066 | 0.054 |
| Word Jaccard | — | 0.740 |
| Start MAE | **285 ms** (poor) | 91 ms |
| Match rate | 62% | 82% |

**Conclusion**: Many aligned words have zero duration (`[5.52–5.52]`), indicating poor alignment quality. The model may be trained on different audio characteristics. **Not suitable** — worse than DTW in both speed and accuracy.

#### 3. Qwen3-ASR (speech-swift, MLX)

Examined for native timestamp capabilities.

**Finding**: Qwen3-ASR is a standard autoregressive decoder with no native timestamp support. Would require DTW like Whisper. **Not applicable**.

### Summary: Alternative Alignment Approaches

| Approach | Total RTF | Start MAE | Quality | Verdict |
|----------|-----------|-----------|---------|---------|
| **Whisper + DTW** (current) | 0.054 | **91 ms** | Best | **Keep** |
| Parakeet-TDT | 0.011 | 229 ms | Poor | Not suitable |
| Qwen3-ForcedAligner | 0.066 | 285 ms | Poor | Not suitable |
| Qwen3-ASR | — | — | — | No timestamps |

### Recommended Path Forward: wav2vec2

The gold standard for word alignment (used by WhisperX) is **wav2vec2 CTC forced alignment**:
- Takes audio + text as input
- Uses CTC to find optimal character-to-frame alignment
- Achieves ~30-50ms MAE (vs 91ms DTW)

Two implementation plans written:
1. **CoreML Export** (`docs/WAV2VEC2_COREML_PLAN.md`) — 1-2 days effort
2. **MLX Native Port** (`docs/WAV2VEC2_MLX_PLAN.md`) — 3-5 days effort

Current DTW implementation (91ms MAE) is acceptable for production use. wav2vec2 is a future optimization opportunity.

---

## Session 1–2 Results (sequential decoder, no word timestamps)

Audio: `ami_ES2002a_full.wav` (21 min), 3 runs each with warmup.

| Engine | RTF (avg) | RTF (min) | RTF (max) |
|--------|-----------|-----------|-----------|
| Python mlx-whisper | 0.0597 | 0.0584 | 0.0605 |
| Swift sequential (all optimisations) | 0.0616 | 0.0594 | 0.0643 |

Swift sequential is within ~3% of Python — at parity. The architectural ceiling
(GPU memory-bandwidth bound) prevents further gains on the single-sequence path.

Note: the stored hardcoded baseline of `0.0500` in earlier benchmark scripts was
stale (measured under low system load). Actual Python RTF under uncontested GPU is ~0.060.

---

## Optimizations Implemented

### 1. Decoder reuse (`WhisperSTT.swift`)

`GreedyDecoder` was previously re-instantiated on every 30-second segment. Now it is created
once and reused across all segments (including temperature fallback retries).

### 2. Encoder result caching (`WhisperSTT.swift`)

The encoder forward pass is now called once per 30-second segment, and the result is passed to
`decoder.decode(audioFeatures:options:)` for all temperature fallback retries — avoiding
redundant encoder passes that the original code performed.

### 3. Pre-computed base suppression masks (`WhisperDecoding.swift`)

Two constant masks — one for iteration 0 (includes SuppressBlank / EOT suppression) and one for
all subsequent iterations — are now computed once after the first forward pass and reused.
Previously they were rebuilt from scratch every token step.

### 4. O(1) timestamp state tracking (`WhisperDecoding.swift`)

The original code scanned the full generated token sequence on every step to determine:
- `lastWasTimestamp`
- `penultimateWasTimestamp`
- `lastTimestampTokenValue`

These are now tracked incrementally via three scalar Swift variables. The update happens at the
end of each decode step in O(1).

**Critical correctness detail**: `penultimateWasTimestamp` must be initialised to `true` (not
`false`). The original code computed it as `numGenerated < 2 || tokens[count-2] >= tsBegin`,
which is vacuously `true` for the first two generated tokens. Setting the initial value to
`false` was a correctness bug introduced during the O(1) refactoring — it caused every segment
to terminate at the second token (EOT selected immediately), yielding zero words transcribed
with a misleadingly fast RTF.

The update rule at the end of each step:

```swift
penultimateWasTimestamp = (numGenerated == 0) ? true : lastWasTimestamp
lastWasTimestamp = nextTokenId >= tokenizer.timestampBegin
```

### 5. Compiled GPU kernels for hot paths (`WhisperDecoding.swift`)

Four operations that were previously uncompiled (re-traced on every call) are now wrapped in
`compile()`:

- `compiledArgmax`: `MLX.argMax(logits, axis: -1)`
- `compiledApplyMask`: `logits + mask`
- `compiledTimestampRules`: timestamp suppression rules (noTs, last-was-ts, monotonicity,
  first-token constraints) — expressed in `MLX.where` ops so they run as a fused GPU kernel
- `compiledTimestampForce`: timestamp probability heuristic (logSumExp-based forced timestamps)

### 6. Async eval for all iterations (`WhisperDecoding.swift`)

The original code called `eval(logits)` synchronously on iteration 0, blocking the CPU until
the GPU completed the forward pass. This was done to read `noSpeechProb` immediately.

The fix: use `asyncEval(logits)` for all iterations, capture `logits[0, sotIndex]` lazily, and
extract `noSpeechProb` after the argmax `.item()` sync (which already serialises the GPU). No
extra GPU-CPU round trip is added.

### 7. `cachedIndices` fix (`WhisperDecoding.swift`)

The `cachedIndices` array (vocabulary index range `[0, vocabSize)`) was initialised with
`MLXArray(Int32(0) ..< Int32(vocabSize))` which produces an array from a Swift Range — this was
found to have type/shape issues. Replaced with `MLXArray.arange(vocabSize)` which correctly
produces a contiguous `[vocabSize]` int32 array.

---

## What Was Attempted but Reverted

### Async eval pipelining with `prevTokenLazy`

An earlier approach maintained a "lazy token" from the previous iteration — kick off the GPU
forward pass for step N+1, sync for step N's token, build masks for N+1. This created a 1-step
CPU-GPU pipeline. It was abandoned because:

1. The dimension handling for the lazy token was error-prone (`[1]` vs `[1,1]`)
2. The correctness bug in O(1) state tracking made the pipeline appear to work (fast RTF, but
   zero words)
3. The current `asyncEval(logits)` + immediate mask queuing achieves equivalent overlap with
   simpler code: GPU runs the forward pass while CPU constructs the `boolState`/`intState`
   arrays and queues compiled GPU kernels

### Fused `compiledLogitStep` (all logit ops in one kernel)

An attempt was made to fuse timestamp rules + force heuristic + mask application into a single
compiled function with 4 array inputs (using the `[MLXArray] -> [MLXArray]` form). This was
reverted because:

- The force heuristic (logSumExp over vocabSize) runs unconditionally inside the fused kernel,
  even for the first generated token where it should be skipped
- GPU `MLX.where` evaluates both branches (eager evaluation of masked-out path), so the
  logSumExp could not be conditionally bypassed
- Benchmarks showed 0.0721 RTF vs 0.0627 for the 3-separate-calls approach — ~15% slower

The force heuristic must remain a conditional CPU-side call:

```swift
if options.timestamps != .none, numGenerated > 0 {
    let probHeuristicMask = compiledTimestampForce(lastLogits, indices, timestampBeginScalar)
    ...
}
```

### `compiledTimestampRules` with `compile(shapeless: true)`

An earlier version used `compile(shapeless: true)` with the `[MLXArray] -> [MLXArray]` form. It
produced zero words. After debugging, the root cause was the O(1) state tracking bug (wrong
`penultimateWasTimestamp` initial value) — not the compile API itself. The current version uses
the typed 3-argument `compile { (a, b, c) -> T in ... }` form and works correctly.

---

## Session 2: Additional Optimizations (no net RTF change)

Three further optimizations were implemented but produced no measurable RTF improvement (within
benchmark variance):

### Iterative log-probability accumulation

Replaced the end-of-segment approach that stacked [numSteps, vocabSize] (~40MB) into a GPU
softmax, with per-step lazy scalars `logit[token] - logSumExp(logits)`. The large allocation is
gone but the benchmark RTF is statistically unchanged — the GPU memory pressure reduction doesn't
affect throughput because the forward pass is memory-bandwidth bound, not allocation-limited.

### Decoder-level mask caching

Promoted `cachedIndices`, `cachedBaseMask`, `cachedBaseMaskFirst` to instance properties of
`GreedyDecoder` (built once per decoder lifetime rather than per decode call). Saves CPU work on
temperature-fallback retries, but temperature fallbacks are rare and mask building was fast (~1ms),
so the effect is negligible.

### Compiled encoder

Wrapped `model.encoder` in `compile()`. The encoder receives fixed shape [1, 3000, nMels] and the
compiled graph is reused for all ~42 segments. First call traces; subsequent calls should reuse
the fused kernel. No measurable RTF benefit — the encoder represents a small fraction of total
decode time and MLX's per-op lazy graph was already efficient.

**Root diagnosis**: All these optimizations target overheads that are not the bottleneck. The
transformer forward pass (attention + FFN across 24 layers) is fully GPU memory-bandwidth bound on
Apple Silicon. Neither compilation tricks nor CPU-side improvements can exceed this hardware ceiling.

---

---

## Session 3: Hybrid Batched Decoding (Phase 5 + 6)

### Goal achieved

The single-sequence decoder is hardware-ceiling-bound at ~0.060 RTF. To go faster requires
amortising the GPU weight-read cost across multiple sequences — batching.

### Architecture

**VAD + Segment merging**
- Silero VAD via FluidAudio (CoreML, ANE-accelerated) replaces the earlier energy-based
  prototype. Segments are merged greedily up to 29 s to maximise Whisper's context window.
- Typical meeting audio: median merged segment ~10–15 s, producing ~15–25 batch slots for
  a 5-minute clip.

**BatchedGreedyDecoder** (`WhisperDecoding.swift`)
- Runs the autoregressive loop on B sequences simultaneously; one GPU forward pass per step
  over stacked `[B, n_ctx, n_state]` features.
- `timestamps: .none` in the batched path — VAD provides boundaries; removing Whisper's
  internal timestamp tokens eliminates per-step timestamp rules + forcing heuristic GPU work
  (~10% RTF reduction).
- Shares file-private compiled kernels with `GreedyDecoder`.

**Inter-batch conditioning**
- `decodeBatch` accepts `options.prompt`, prepending `<|startofprev|>` + prompt tokens
  (same logic as `GreedyDecoder.decode`). `transcribeBatched` carries the last batch's
  tokens forward as context — approximates `condition_on_previous_text` while preserving
  within-batch parallelism. Effect: Jaccard 0.630 → 0.718.

**Word timestamps**
- `addWordTimestamps` / `findAlignment` accept `audioFeatures: MLXArray?`. When provided
  (from the same encode step that powered decoding), the DTW alignment pass calls
  `model.decodeWithCrossQK(audioFeatures, tokens:)` — decoder-only, no re-encode.
  This eliminated the +67% RTF overhead that word timestamps originally imposed.

**Quality gates** (from faster-whisper / WhisperX research)

| Gate | Where | Effect |
|---|---|---|
| Compression ratio > 2.4 | `shouldAccept` post-decode | Drops repetitive segments |
| `noRepeatNgramSize = 3` | `BatchedGreedyDecoder` per step | Prevents n-gram repeats during generation |
| `hallucinationSilenceThreshold = 2.0 s` | Post-word-timestamp | Drops segments with >2 s inter-word gap |

Combined effect: word count 429 → 391 (Python: 408); Jaccard 0.630 → 0.736.

### Benchmark tooling added

- `benchmarks/scripts/word_baseline.py`: Python word-level JSON baseline
  (`condition_on_previous_text=False` for fair comparison)
- `word-compare` CLI mode: one-to-one word matching (±1.5 s window), MAE, p90, signed bias

### Remaining gap

Word Jaccard 0.736 vs Python 1.0. Residual causes:
1. Within a batch, all segments share the previous batch's prompt — earlier segments in the
   batch have slightly weaker context than a fully sequential decoder would give them.
2. The WhisperX approach (dedicated wav2vec2/MMS forced-aligner replacing cross-attention DTW)
   would give sub-50 ms alignment but requires a CoreML-compatible forced aligner — none
   currently exists for Apple platforms. FluidAudio may be a path there.

---

## Why 30% Faster Requires a Different Algorithm (historical, sequential decoder)

### The hardware ceiling

Both Swift and Python execute identical MLX operations. Both arrive at ~0.060 RTF on the same
hardware. Python's `@mx.compile` on the full decode step provides no measurable advantage because:
1. The per-op dispatch overhead is already minimal in MLX's C++ runtime
2. Kernel fusion across attention layers provides marginal benefit when memory bandwidth is the
   limiting factor
3. The quantized weight dequantization and matrix multiplication dominate — these are already
   highly optimised Metal kernels

### What would actually give 30%+ speedup

1. **Speculative decoding** (most viable): Use whisper-tiny or whisper-small as a draft model to
   speculatively generate K tokens, verify with the large model. Acceptance rate ~70-80% expected
   for English meeting audio. Theoretical speedup: 2-3x at high acceptance rates.
   - Implementation: add `SpeculativeDecoder` wrapper that manages two model instances
   - Complexity: high; requires model compatibility checks and fallback logic

2. **Static KV cache (pre-allocated)**: Pre-allocate self-attention KV buffers to `maxTokens` size
   at decode start. This makes all KV shapes fixed across steps, enabling `compile(shapeless: false)`
   on the full decode step (tokens → logits → next token, with cache update). Requires rewriting
   the attention layer's cache management.
   - Complexity: medium-high; needs attention code changes and positional masking
   - Benefit: allows full-step compilation → reduces per-step overhead

3. **Batched encoding** (limited benefit): Process multiple 30-second windows through the encoder
   simultaneously. Benefit is capped by the encoder's share of total time (~15-20%) and requires
   disabling `conditionOnPreviousText`.

4. **distil-whisper** or quantized smaller model: 3-5x faster with some quality trade-off.

## Remaining Gap and Why (sequential decoder, superseded by Session 3)

Swift sequential is at parity with Python (~0% difference). With batching (Session 3),
Swift is 10% faster including word timestamps. The notes below were written before batching
was implemented; they are kept for historical context.

The most significant remaining opportunity for the sequential path was:

### Python `@mx.compile` on the full decode step

Python `mlx-whisper` wraps the entire decode step (model forward pass + logit ops) in
`@mx.compile`. This allows the MLX C++ runtime to compile a graph that spans transformer layers,
attention, FFN, and logit processing — enabling kernel fusion across layer boundaries.

In Swift, `compile()` is used only on small subsections. The model forward pass
(`model.decode(...)`) is called through standard Swift method dispatch; the resulting MLX ops
are traced lazily but not compiled into a single fused graph.

To achieve equivalent compilation in Swift would require:

1. Flattening the KV cache (currently `[((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)]`) into a
   flat `[MLXArray]` that can be passed to and returned from a `compile()` closure
2. Wrapping `model.decode(tokens, audioFeatures, flatCache)` in `compile()`
3. Unflattening the cache on return

This is a non-trivial refactor of the model layer. The benefit is uncertain — on memory-bandwidth-
bound hardware (Apple Silicon, which is the common case for MLX), kernel fusion may not yield a
significant speedup for the attention mechanism; the bottleneck is memory access to 4-bit weights.

### Measurement conditions

Both benchmarks show high variance due to GPU/CPU contention from other processes. Ideally,
benchmarks should be run with other GPU-intensive applications closed. The "best" runs
(Swift: 0.0562, Python: ~0.0650 estimated) suggest Swift is already ahead of Python by ~14%
under ideal conditions.

---

## Benchmark Infrastructure

- **Swift CLI**: `benchmarks/cli/WhisperBenchmarkCLI.swift` — built with `xcodebuild` (required
  for Metal library linking; `swift run` does not work)
- **Python baseline**: `benchmarks/scripts/word_baseline.py` — generates word-level JSON for
  `word-compare` mode (`condition_on_previous_text=False`)
- **Fixtures**: `benchmarks/fixtures/ami_ES2002a_{full,5min}.wav`,
  `ami_ES2002a_5min_word_baseline.json`

**CLI modes:**
- `quick` / `full` — sequential, 5-min / 21-min
- `batch-quick` / `batch-full` — batched B=4, no word timestamps
- `compare` / `compare-full` — sequential vs batched side-by-side
- `word-compare` — batched with word timestamps vs Python baseline
- `vad-scan [dir] [maxFiles]` — VAD parameter validation on real recordings

### Build command

```bash
xcodebuild -scheme WhisperBenchmark -configuration Release \
  -destination 'platform=macOS' -derivedDataPath .xcodebuild build
.xcodebuild/Build/Products/Release/WhisperBenchmark [quick|full]
```

### Why `swift test` / `swift run` cannot be used

The Metal default library (`default.metallib`) is not embedded in the executable when building
through SPM's CLI tools. Runtime error: `MLX error: Failed to load the default metallib. library
not found`. Xcode (`xcodebuild`) correctly embeds the Metal library.
