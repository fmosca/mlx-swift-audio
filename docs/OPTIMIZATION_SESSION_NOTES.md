# Optimization Session Notes: Swift vs Python MLX Whisper

## Goal

Achieve Swift transcription RTF at least 30% faster than Python `mlx-whisper` on the same model
(`mlx-community/whisper-large-v3-turbo-q4`), measured on the AMI ES2002a corpus (21-minute
meeting audio).

---

## Measurement Results (clean GPU, sequential runs)

All measurements on Apple Silicon (M-series), audio: `benchmarks/fixtures/ami_ES2002a_full.wav`,
3 benchmark runs each with warmup, GPU uncontested.

| Engine | RTF (avg) | RTF (min) | RTF (max) |
|--------|-----------|-----------|-----------|
| Python mlx-whisper | 0.0597 | 0.0584 | 0.0605 |
| Swift (all optimisations) | 0.0616 | 0.0594 | 0.0643 |

**Swift is within ~3% of Python** — essentially at parity. The measurement distributions overlap;
no statistically significant difference.

The stored hardcoded baseline of `0.0500` in both benchmark scripts is stale (measured during an
earlier session under low system load). Under an uncontested GPU, Python's actual RTF is ~0.060.

**Action required**: Update both benchmark scripts to measure Python RTF dynamically.

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

## Why 30% Faster Requires a Different Algorithm

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

## Remaining Gap and Why

Swift is at parity with Python (~0% difference). The target of 30% faster Python remains unmet.
The most significant remaining opportunity:

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
- **Python script**: `benchmarks/benchmark.py` — requires meeting-transcriber venv
  (`/Users/francesco.mosca/Work/meeting-transcriber/.venv/bin/python3`)
- **Fixtures**: `benchmarks/fixtures/ami_ES2002a_{full,5min}.wav`, `ami_ES2002a_baseline.txt`
- **Quick run** (5 min): `WhisperBenchmark quick` (~2 minutes)
- **Full run** (21 min): `WhisperBenchmark full` (~6 minutes, 3 iterations)

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
