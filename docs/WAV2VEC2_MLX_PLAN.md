# wav2vec2 Forced Aligner — MLX Native Port

## Goal

Replace the CoreML wav2vec2 implementation with a native MLX port to:
- Eliminate CoreML/MLX Metal context conflicts causing segfaults
- Achieve unified runtime with Whisper (shared GPU memory)
- Enable full optimization control (compilation, quantization)

**Target metrics:**
- Start MAE < 100ms (ideally 30-50ms, vs 91ms DTW baseline)
- Alignment RTF < 0.02
- Word match rate > 80%

---

## Architecture Overview

```
Audio ──► wav2vec2 (MLX) ──► Frame Log-Probs ──► CTC Forced Align ──► Words
          ├── FeatureExtractor (7× Conv1d + LayerNorm/GELU)
          ├── FeatureProjection (LayerNorm + Linear)
          ├── Encoder (12× Transformer layers)
          └── CTCHead (Linear → vocab)
```

---

## Current Status (March 11, 2026)

### ✅ Completed Phases

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Model Architecture | ✅ Complete | All components implemented and shape-validated |
| 2. Weight Loading | ✅ Complete | Clean key transformation, all tests passing |
| 3. Tokenizer | ✅ Complete | Reused existing `Wav2Vec2Tokenizer.swift` |
| 4. CTC Alignment | ✅ Complete | Bug fix: tokens now close on blank frames |
| 5. High-Level API | ✅ Complete | Drop-in API matching CoreML version |
| 6. VAD Preprocessing | ✅ Complete | Two-pass VAD, first word: 40ms error |

### ⚠️ Phase 7: Model Output Quality - ISSUE IDENTIFIED

**Current Results (ami_ES2002a_5min.wav, 5 minutes):**

| Metric | Target | Current | DTW Baseline | Status |
|--------|--------|---------|--------------|--------|
| First word start | < 100ms | **40ms** | 91ms | ✅ Excellent |
| Start MAE | < 100ms | **460ms** | 91ms | ❌ 5x target |
| Alignment RTF | < 0.02 | 0.053 | ~0.01 | ⚠️ 2.6x target |
| Word match rate | > 80% | **55.6%** | ~95% | ❌ 24% gap |

---

## Root Cause Analysis

### Numerical Verification

The MLX wav2vec2 model produces fundamentally different outputs than PyTorch:

```
Feature Extractor (first 10 values):
  PyTorch: [ 0.011,  0.020,  0.011,  0.010,  0.010,  0.011,  0.021, ...]  (all positive)
  MLX:    [-0.032, -0.035, -0.031, -0.033, -0.033, -0.032, -0.035, ...]  (all negative)
  ⚠️ OPPOSITE SIGNS!

Max absolute differences:
  Feature extractor: 0.115
  Encoder layer 0: 1.71
  Logits: 12.7
  Log probs: 17.8
```

### Identified Issue: LayerNorm vs GroupNorm Mismatch

**Root cause:** PyTorch wav2vec2 uses `GroupNorm(num_groups=512, num_channels=512)` which is InstanceNorm (each channel normalized independently across frames). The MLX port used `LayerNorm(512)` which normalizes each frame independently across channels - completely different behavior.

| Aspect | PyTorch GroupNorm(512, 512) | MLX LayerNorm(512) |
|--------|----------------------------|-------------------|
| Normalization axis | Each channel across frames | Each frame across channels |
| Format | [batch, channels, frames] | [batch, frames, channels] |
| Behavior | InstanceNorm | Per-frame normalization |

**Fix Attempted (March 11):**
- Replaced `LayerNorm` with `MLXNN.GroupNorm(groupCount: 512, dimensions: 512, pytorchCompatible: true)`
- **Result:** Alignment quality got WORSE (9.1% match rate, 663ms MAE vs 55.6%, 460ms baseline)
- **Decision:** REVERTED to LayerNorm

**Paradoxical Finding:** Despite being numerically "incorrect" (opposite signs to PyTorch), LayerNorm produces better practical alignment results than the "correct" GroupNorm implementation. This suggests compensating errors in the MLX model that happen to work better with our CTC alignment algorithm.

---

## Alternatives Attempted

### TrellisForcedAligner (WhisperX-style)

**File:** `package/Wav2Vec2Aligner/TrellisForcedAlign.swift`

Implemented alternative CTC algorithm that provides continuous frame coverage:
- **Frame coverage test:** CTC=0%, Trellis=71.4%
- **Benchmark result:** 9.1% match rate, 663ms MAE (WORSE than baseline)

**Conclusion:** The Trellis algorithm's continuous coverage doesn't compensate for the underlying model output quality issues.

---

## Final Recommendation

### Use DTW for Alignment ✅

**Rationale:**
- **DTW MAE:** 91ms (meets < 100ms target)
- **Match rate:** ~95% (exceeds 80% target)
- **Reliability:** Simple, well-tested, no model dependencies
- **Performance:** Faster RTF than MLX wav2vec2
- **No Metal conflicts**: CPU-based, works alongside MLX

### wav2vec2 Approaches: Summary

| Approach | Status | MAE | Match Rate | Issues |
|----------|--------|-----|------------|--------|
| **DTW** (current) | ✅ Working | 91ms | ~95% | CPU-bound |
| **CoreML wav2vec2** | ❌ Broken | — | — | Metal segfault + numerical corruption |
| **MLX wav2vec2** | ⚠️ Poor quality | 460ms | 55.6% | Numerical discrepancies |
| **Subprocess CoreML** | ❌ Not viable | — | — | Model outputs are wrong |

---

### CoreML Validation Results (March 11, 2026)

**ONNX → PyTorch**: ✅ PASSED (max diff: 0.000042)

**CoreML Model**: ❌ FAILED - Numerical corruption during ONNX → CoreML conversion

| Metric | CoreML | PyTorch | Error |
|--------|--------|---------|-------|
| Min | -42,012,168 | -16.71 | 2.5M× |
| Max | 0.000043 | 4.03 | Wrong |
| Mean | -2,587,863 | -2.62 | 1M× |

**Conclusion**: Subprocess CoreML workaround (to avoid Metal conflict) is not viable because the CoreML model itself produces numerically incorrect outputs. The conversion failed despite ONNX being correct.

### If wav2vec2 Alignment is Still Required

**Remaining options:**
1. **Fix CoreML conversion**: Re-export with different coremltools version or settings (high effort, uncertain)
2. **Deep MLX debugging**: Systematic layer-by-layer comparison beyond normalization (very high effort)
3. **Different model**: Consider other alignment models that may port more cleanly

**Note:** All options require significant development effort with uncertain outcomes. DTW is recommended.

---

## Implementation Notes

### Feature Extractor Normalization

The feature extractor uses LayerNorm despite the numerical mismatch with PyTorch:

```swift
// package/STT/Wav2Vec2/Wav2Vec2FeatureExtractor.swift
// NOTE: PyTorch uses GroupNorm(512, 512) which is InstanceNorm,
// but MLX LayerNorm gives better alignment results in practice.
_layerNorm.wrappedValue = LayerNorm(dimensions: outChannels, eps: 1e-5, affine: true)
```

This is a known compromise that produces better practical results despite the numerical discrepancy.

### CTC Alignment Bug Fix

**Location:** `CTCForcedAlign.swift`

**Bug:** Tokens weren't closed on blank frames, causing them to span entire blank regions.

**Fix:** Track `lastNonBlankFrame` and close tokens immediately when blanks are encountered.

---

## File Structure

```
package/STT/Wav2Vec2/
├── Wav2Vec2Config.swift          ✅ Complete
├── Wav2Vec2FeatureExtractor.swift ✅ Complete (uses LayerNorm)
├── Wav2Vec2Encoder.swift         ✅ Complete
├── Wav2Vec2ForCTC.swift          ✅ Complete
├── Wav2Vec2WeightLoader.swift    ✅ Complete
└── Wav2Vec2AlignerMLX.swift      ✅ Complete (in Wav2Vec2Aligner/)

package/Wav2Vec2Aligner/
├── Wav2Vec2Tokenizer.swift       ✅ Reused
├── CTCForcedAlign.swift          ✅ Bug fixed
├── TrellisForcedAlign.swift      ✅ Implemented (results worse)
└── Error.swift                   ✅ Reused

package/Tests/
├── Wav2Vec2MLXTests.swift        ✅ 17 tests passing
├── Wav2Vec2WeightLoadingTests.swift ✅ 3 tests passing
├── Wav2Vec2NumericalTests.swift ✅ Numerical comparison tests
└── Wav2Vec2OutputDebugTests.swift ✅ Debug output tests
```

---

## Testing Commands

```bash
# Run all Wav2Vec2 tests
MAC_ID=$(xcodebuild -scheme mlx-audio-Package -showdestinations | \
  grep "platform=macOS" | head -1 | awk -F"id:" '{print $2}' | awk -F"," '{print $1}')
xcodebuild test-without-building -scheme mlx-audio-Package \
  -destination "id=$MAC_ID,arch=arm64" \
  -derivedDataPath /tmp/DerivedData \
  -only-testing:MLXAudioTests/Wav2Vec2*

# Run alignment benchmark
xcodebuild -scheme WhisperBenchmark -configuration Release \
  -destination 'platform=macOS' \
  -derivedDataPath /tmp/DerivedData build

/tmp/DerivedData/Build/Products/Release/WhisperBenchmark wav2vec2-align
```

---

## Validation Scripts

### CoreML Validation
- `scripts/test_coreml_model.swift` - Swift script to validate CoreML model output
- **Result**: CoreML model produces numerically incorrect outputs (off by factors of millions)
- **ONNX validation**: Passed (max diff: 0.000042 vs PyTorch)
- **CoreML conversion failed**: ONNX → CoreML conversion introduces numerical corruption
