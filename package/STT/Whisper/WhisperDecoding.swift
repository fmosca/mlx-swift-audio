// Copyright © 2022 OpenAI (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/openai/whisper
// License: licenses/whisper.txt

import Compression
import Foundation
import MLX
import MLXNN
import MLXRandom

// MARK: - Decoding Options

/// Options for Whisper decoding
struct DecodingOptions {
  /// Task: transcribe or translate to English
  let task: TranscriptionTask

  /// Language code (e.g., "en", "zh"), nil for auto-detect
  let language: String?

  /// Sampling temperature (0.0 for greedy decoding)
  let temperature: Float

  /// Maximum number of tokens to generate
  let maxTokens: Int

  /// Whether to include timestamps
  let timestamps: TimestampGranularity

  /// Prompt tokens from previous segments (for conditioning on previous text)
  /// These are prepended to the SOT sequence
  let prompt: [Int]

  /// Prevents the model from repeating any n-gram of this size.
  /// 0 disables the check. Mirrors faster-whisper's no_repeat_ngram_size.
  let noRepeatNgramSize: Int

  init(
    task: TranscriptionTask = .transcribe,
    language: String? = nil,
    temperature: Float = 0.0,
    maxTokens: Int = 448,
    timestamps: TimestampGranularity = .segment,
    prompt: [Int] = [],
    noRepeatNgramSize: Int = 0
  ) {
    self.task = task
    self.language = language
    self.temperature = temperature
    self.maxTokens = maxTokens
    self.timestamps = timestamps
    self.prompt = prompt
    self.noRepeatNgramSize = noRepeatNgramSize
  }

  static let `default` = DecodingOptions()
}

// MARK: - Decoding Result

/// Result from decoding a single audio segment
struct DecodingResult {
  /// Generated token sequence
  let tokens: [Int]

  /// Decoded text
  let text: String

  /// Average log probability
  let avgLogProb: Float

  /// No-speech probability (0-1)
  let noSpeechProb: Float

  /// Temperature used
  let temperature: Float

  /// Compression ratio (text length / gzip(text) length)
  let compressionRatio: Float
}

// MARK: - Compiled Hot Paths

/// Compiled greedy sampling: argmax over vocab dimension
private let compiledArgmax = compile { (logits: MLXArray) -> MLXArray in
  MLX.argMax(logits, axis: -1)
}

/// Timestamp suppression rules compiled into a GPU kernel. (Step 2 of the decode loop).
/// Packs boolean and integer state into two MLXArray vectors to stay within
/// compile()'s 3-argument typed form.
///
/// boolState: [lastWasTimestamp, penultimateWasTimestamp] as Int32 (0 or 1)
/// intState:  [lastTsVal, numGenerated, tsBegin, noTimestamps, eot] as Int32
private nonisolated(unsafe) let compiledTimestampRules: (MLXArray, MLXArray, MLXArray) -> MLXArray =
  compile { (indices: MLXArray, boolState: MLXArray, intState: MLXArray) -> MLXArray in
    let lastWasTs = boolState[0]
    let penultimateWasTs = boolState[1]
    let lastTsVal = intState[0]
    let numGen = intState[1]
    let tsBegin = intState[2]
    let noTs = intState[3]
    let eot = intState[4]

    let negInf = MLXArray(-Float.infinity)
    let one = MLXArray(Int32(1))
    let zero = MLXArray(Int32(0))
    let zeros = MLXArray.zeros(indices.shape)

    var mask = MLX.where(indices .== noTs, negInf, zeros)

    let whenPenultTs = MLX.where(indices .>= tsBegin, negInf, zeros)
    let whenPenultText = MLX.where(indices .< eot, negInf, zeros)
    let tsRule = MLX.where(penultimateWasTs .== one, whenPenultTs, whenPenultText)
    mask = mask + MLX.where(lastWasTs .== one, tsRule, zeros)

    let monoRule = MLX.where(
      MLX.logicalAnd(indices .>= tsBegin, indices .< lastTsVal), negInf, zeros)
    mask = mask + MLX.where(lastTsVal .> tsBegin, monoRule, zeros)

    let textSuppress = MLX.where(indices .< tsBegin, negInf, zeros)
    let maxInitTs = tsBegin + MLXArray(Int32(50))
    let distantTsSuppress = MLX.where(indices .> maxInitTs, negInf, zeros)
    mask = mask + MLX.where(numGen .== zero, textSuppress + distantTsSuppress, zeros)

    return mask
  }

/// Compiled mask application: add -inf mask to logits
private let compiledApplyMask = compile { (logits: MLXArray, mask: MLXArray) -> MLXArray in
  logits + mask
}

/// Compiled timestamp probability heuristic.
/// Returns a [vocab_size] mask: -inf for text tokens when timestamps should be forced.
/// timestampBeginScalar: MLXArray scalar (e.g. MLXArray(Int32(tokenizer.timestampBegin)))
private nonisolated(unsafe) let compiledTimestampForce = compile {
  (logits: MLXArray, indices: MLXArray, timestampBeginScalar: MLXArray) -> MLXArray in
  let logProbs = logits - MLX.logSumExp(logits, axes: [-1], keepDims: true)
  let maskTs = indices .>= timestampBeginScalar
  let maskText = indices .< timestampBeginScalar
  let timestampLogProbSum = MLX.logSumExp(
    MLX.where(maskTs, logProbs, MLXArray(-Float.infinity)),
    axes: [-1],
    keepDims: true
  )
  let maxTextLogProb = MLX.where(maskText, logProbs, MLXArray(-Float.infinity))
    .max(axes: [-1], keepDims: true)
  let shouldForce = timestampLogProbSum .> maxTextLogProb
  return MLX.where(
    MLX.logicalAnd(shouldForce, indices .< timestampBeginScalar),
    MLXArray(-Float.infinity),
    MLXArray.zeros(indices.shape)
  )
}

// MARK: - Greedy Decoder

/// Greedy decoder for Whisper
///
/// Implements simple greedy decoding with KV caching.
/// Use `decode(audioFeatures:options:)` when encoding once and decoding multiple times
/// (e.g. temperature fallback) to avoid redundant encoder passes.
class GreedyDecoder {
  let model: WhisperModel
  let tokenizer: WhisperTokenizer

  // Lazily initialised once per decoder instance (fixed per model/tokenizer pair)
  private var _cachedIndices: MLXArray?
  private var _cachedBaseMask: MLXArray?
  private var _cachedBaseMaskFirst: MLXArray?

  init(model: WhisperModel, tokenizer: WhisperTokenizer) {
    self.model = model
    self.tokenizer = tokenizer
  }

  private func ensureMasksCached(vocabSize: Int) {
    guard _cachedIndices == nil else { return }
    _cachedIndices = MLXArray.arange(vocabSize)

    var baseIds = tokenizer.nonSpeechTokens()
    baseIds.append(contentsOf: [
      tokenizer.transcribe, tokenizer.translate,
      tokenizer.sot, tokenizer.sotPrev,
      tokenizer.sotLm, tokenizer.noSpeech,
    ])
    var maskValues = [Float](repeating: 0.0, count: vocabSize)
    for id in baseIds where id < vocabSize { maskValues[id] = -Float.infinity }
    _cachedBaseMask = MLXArray(maskValues)

    var firstMaskValues = maskValues
    if let blankTokens = try? tokenizer.encode(" ") {
      for id in blankTokens where id < vocabSize { firstMaskValues[id] = -Float.infinity }
    }
    firstMaskValues[tokenizer.eot] = -Float.infinity
    _cachedBaseMaskFirst = MLXArray(firstMaskValues)
  }

  /// Decode from mel spectrogram (encodes then decodes).
  /// Use `decode(audioFeatures:options:)` when reusing encoder output across temperature fallbacks.
  ///
  /// - Parameters:
  ///   - mel: Mel spectrogram (batch=1, n_mels, n_frames)
  ///   - options: Decoding options (temperature, prompt, etc.)
  /// - Returns: Decoding result
  func decode(_ mel: MLXArray, options: DecodingOptions) -> DecodingResult {
    let audioFeatures = model.compiledEncode(mel)
    eval(audioFeatures)
    return decode(audioFeatures: audioFeatures, options: options)
  }

  /// Decode from precomputed audio features.
  /// Reuse audio features across temperature fallbacks to avoid redundant encoder passes.
  ///
  /// - Parameters:
  ///   - audioFeatures: Output of `model.compiledEncode(mel)` (batch=1, n_audio_ctx, n_audio_state)
  ///   - options: Decoding options (temperature, prompt, etc.)
  /// - Returns: Decoding result
  func decode(audioFeatures: MLXArray, options: DecodingOptions) -> DecodingResult {
    // Build initial token sequence
    // If prompt tokens are provided, prepend them with <|startofprev|>
    // This matches Python's behavior for condition_on_previous_text
    var tokens: [Int] = []

    if !options.prompt.isEmpty {
      // Add <|startofprev|> followed by prompt tokens (previous segment's output).
      // Truncate to n_ctx // 2 - 1 tokens, matching Python:
      //   tokens = [sot_prev] + prompt_tokens[-(n_ctx // 2 - 1):] + sot_sequence
      // Without this cap the prompt grows unboundedly; once its length exceeds
      // maxTokens the decode loop generates 0 new tokens, producing empty results
      // for every subsequent 30-second window while seek still advances — silently
      // dropping the second half of any long recording.
      let maxPromptLen = (options.maxTokens / 2) - 1
      let truncatedPrompt =
        options.prompt.count > maxPromptLen
        ? Array(options.prompt.suffix(maxPromptLen))
        : options.prompt
      tokens.append(tokenizer.sotPrev)
      tokens.append(contentsOf: truncatedPrompt)
    }

    // Track SOT position for no-speech probability extraction (matches Python's sot_index)
    let sotIndex = tokens.count

    // Add SOT sequence (SOT, language, task tokens)
    tokens.append(contentsOf: tokenizer.sotSequence(language: options.language, task: options.task))

    // Only add no-timestamps token if timestamps are disabled
    // When timestamps are enabled, the first timestamp is the first GENERATED token
    if options.timestamps == .none {
      tokens.append(tokenizer.noTimestamps)
    }

    // Calculate how many tokens we can generate
    // We need to account for the initial SOT sequence
    let initialTokenCount = tokens.count
    let maxGenerateTokens = options.maxTokens - initialTokenCount

    var kvCache: [((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)]? = nil
    var noSpeechProb: Float = 0.0
    var cachedStepLogProbs: [MLXArray] = []

    // O(1) timestamp state tracking (replaces O(n) scan over generated tokens)
    var lastWasTimestamp = false
    var penultimateWasTimestamp = true  // vacuously true: no penultimate token yet
    var lastTimestampTokenValue = 0

    for iteration in 0 ..< maxGenerateTokens {
      // Convert tokens to MLXArray
      // With KV caching: only pass new token(s), not all tokens
      let tokensToProcess: MLXArray
      if kvCache != nil {
        // Subsequent iterations: pass only the last token
        let lastToken = tokens.last!
        tokensToProcess = MLXArray(Int32(lastToken)).expandedDimensions(axis: 0).expandedDimensions(axis: 0)
      } else {
        // First iteration: pass all initial tokens (SOT sequence)
        tokensToProcess = MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0)
      }

      // Forward pass through decoder
      let (logits, newCache, _) = model.decode(
        tokensToProcess,
        audioFeatures: audioFeatures,
        kvCache: kvCache
      )

      // asyncEval lets GPU run the forward pass while CPU builds masks below.
      // noSpeechProb uses lazy ops (no early sync needed; syncs with argmax below).
      asyncEval(logits)

      // Update KV cache
      kvCache = newCache

      // Lazily capture no-speech logit for iteration 0; the actual .item() sync
      // happens on noSpeechProb AFTER the argmax sync, so no extra GPU sync is added.
      let sotLogitsLazy: MLXArray? = iteration == 0 ? logits[0, sotIndex] : nil

      // Get logits for last token
      var lastLogits = logits[0, -1]
      let vocabSize = lastLogits.shape[0]

      // Get number of generated tokens (excluding initial SOT sequence)
      let numGenerated = tokens.count - initialTokenCount

      // =============================================================================
      // STEP 1: Base suppression mask (SuppressBlank + SuppressTokens)
      // =============================================================================
      ensureMasksCached(vocabSize: vocabSize)
      let indices = _cachedIndices!
      let baseMask = iteration == 0 ? _cachedBaseMaskFirst! : _cachedBaseMask!

      // =============================================================================
      // STEP 2: Build timestamp rules mask (ApplyTimestampRules internal mask)
      // =============================================================================
      var timestampMask = MLXArray.zeros([vocabSize])
      if options.timestamps != .none {
        let boolState = MLXArray([
          Int32(lastWasTimestamp ? 1 : 0),
          Int32(penultimateWasTimestamp ? 1 : 0),
        ])
        let intState = MLXArray([
          Int32(lastTimestampTokenValue),
          Int32(numGenerated),
          Int32(tokenizer.timestampBegin),
          Int32(tokenizer.noTimestamps),
          Int32(tokenizer.eot),
        ])
        timestampMask = compiledTimestampRules(indices, boolState, intState)
      }

      // =============================================================================
      // STEP 3: Apply timestamp probability heuristic (Python lines 381-394)
      // =============================================================================
      if options.timestamps != .none, numGenerated > 0 {
        let timestampBeginScalar = MLXArray(Int32(tokenizer.timestampBegin))
        let probHeuristicMask = compiledTimestampForce(lastLogits, indices, timestampBeginScalar)
        timestampMask = MLX.minimum(timestampMask, probHeuristicMask)
      }

      // =============================================================================
      // STEP 4: Combine masks and apply
      // =============================================================================
      let finalMask = MLX.minimum(baseMask, timestampMask)
      let maskedLogits = compiledApplyMask(lastLogits, finalMask)

      // Sample next token (the .item() call here is the primary GPU sync for this iteration)
      let nextTokenId: Int
      if options.temperature == 0.0 {
        let nextTokenArray = compiledArgmax(maskedLogits)
        nextTokenId = Int(nextTokenArray.item(Int32.self))
      } else {
        let probs = MLX.softmax(maskedLogits / options.temperature, axis: -1)
        let nextTokenArray = MLXRandom.categorical(MLX.log(probs + 1e-10))
        nextTokenId = Int(nextTokenArray.item(Int32.self))
      }

      // Extract no-speech probability after the GPU sync (sotLogitsLazy is already evaluated)
      if let sotLogits = sotLogitsLazy {
        let probs = MLX.softmax(sotLogits, axis: -1)
        noSpeechProb = probs[tokenizer.noSpeech].item(Float.self)
      }

      // Collect lazy log-prob scalar for this step (used for avgLogProb at the end).
      // lastLogits is the unmasked model output. log(softmax)[i] = logits[i] - logSumExp(logits).
      // No .item() here — keep lazy to avoid extra GPU sync per token.
      if nextTokenId != tokenizer.eot {
        let lse = MLX.logSumExp(lastLogits, axes: [-1])
        let logProb = MLX.take(lastLogits, MLXArray(Int32(nextTokenId))) - lse
        cachedStepLogProbs.append(logProb)
      }

      tokens.append(nextTokenId)

      if nextTokenId == tokenizer.eot {
        break
      }

      // Update O(1) timestamp tracking for the next iteration.
      // penultimateWasTimestamp mirrors original: numGenerated < 2 OR second-to-last was timestamp.
      // For iteration 0 (one token generated), penultimate is vacuously true (numGenerated+1 = 1 < 2).
      // For iteration 1+, penultimate = what lastWasTimestamp was before this token.
      penultimateWasTimestamp = (numGenerated == 0) ? true : lastWasTimestamp
      lastWasTimestamp = nextTokenId >= tokenizer.timestampBegin
      if lastWasTimestamp {
        lastTimestampTokenValue = nextTokenId
        if penultimateWasTimestamp { lastTimestampTokenValue += 1 }
      }
    }

    // Compute avgLogProb from lazily accumulated per-step log-probs (avoids 40MB batched softmax).
    let avgLogProb: Float
    if cachedStepLogProbs.isEmpty {
      avgLogProb = 0.0
    } else {
      let summed = MLX.stacked(cachedStepLogProbs, axis: 0).sum()
      eval(summed)
      avgLogProb = summed.item(Float.self) / Float(cachedStepLogProbs.count)
    }

    // Extract only the generated tokens (exclude prompt and SOT sequence)
    // This matches Python's behavior where result.tokens only contains generated tokens
    var generatedTokens = Array(tokens[initialTokenCount...])

    // Strip EOT from result tokens (matches Python line 669: t[: t.index(tokenizer.eot)])
    // This is important for timestamp detection in seek-based processing
    if let eotIndex = generatedTokens.firstIndex(of: tokenizer.eot) {
      generatedTokens = Array(generatedTokens[..<eotIndex])
    }

    // Decode text from generated tokens only
    let text = tokenizer.decode(generatedTokens)

    // Compute compression ratio using zlib (matches Python implementation)
    // Higher ratio = more repetitive text (hallucination indicator)
    let compressionRatio = computeCompressionRatio(text)

    return DecodingResult(
      tokens: generatedTokens,
      text: text,
      avgLogProb: avgLogProb,
      noSpeechProb: noSpeechProb,
      temperature: options.temperature,
      compressionRatio: compressionRatio
    )
  }
}

// MARK: - Compression Ratio

/// Compute compression ratio using zlib compression
/// Matches Python: len(text_bytes) / len(zlib.compress(text_bytes))
/// Higher ratio indicates more repetitive text (potential hallucination)
///
/// - Parameter text: Text to analyze
/// - Returns: Compression ratio (uncompressed/compressed size)
func computeCompressionRatio(_ text: String) -> Float {
  guard !text.isEmpty else { return 1.0 }

  let textBytes = Array(text.utf8)
  let sourceSize = textBytes.count

  // Use zlib compression (COMPRESSION_ZLIB matches Python's zlib.compress)
  // Allocate destination buffer - worst case is slight expansion
  let destinationBuffer = UnsafeMutablePointer<UInt8>.allocate(capacity: sourceSize + 512)
  defer { destinationBuffer.deallocate() }

  let compressedSize = textBytes.withUnsafeBufferPointer { sourceBuffer in
    compression_encode_buffer(
      destinationBuffer,
      sourceSize + 512,
      sourceBuffer.baseAddress!,
      sourceSize,
      nil,
      COMPRESSION_ZLIB
    )
  }

  // Handle compression failure
  guard compressedSize > 0 else { return 1.0 }

  return Float(sourceSize) / Float(compressedSize)
}

/// Returns the set of token IDs that would complete a repeated n-gram.
///
/// Scans `tokens` (the full sequence for one beam, including SOT prefix) for all
/// occurrences of the last (n-1) tokens and collects the token that followed each
/// occurrence.  If any of those tokens appear next, a repeat n-gram would form.
///
/// - Parameters:
///   - tokens: Full token sequence (SOT prefix + generated tokens so far)
///   - initialTokenCount: Number of SOT/prompt prefix tokens (generated region starts here)
///   - n: N-gram size (must be >= 2)
/// - Returns: Set of banned token IDs (may be empty)
func bannedNgramTokens(tokens: [Int], initialTokenCount: Int, n: Int) -> Set<Int> {
  guard n >= 2 else { return [] }
  // Convert to Array for 0-based indexing — ArraySlice preserves the source
  // array's indices which would make the loop subscripts wrong.
  let generated = Array(tokens[initialTokenCount...])
  guard generated.count >= n - 1 else { return [] }
  let suffix = Array(generated.suffix(n - 1))
  var banned = Set<Int>()
  let scanEnd = generated.count - (n - 1)
  if scanEnd <= 0 { return [] }
  for i in 0 ..< scanEnd {
    if Array(generated[i ..< i + (n - 1)]) == suffix {
      banned.insert(generated[i + (n - 1)])
    }
  }
  return banned
}

// MARK: - Batched Greedy Decoder

/// Decodes B audio segments in parallel, amortising the weight-read cost across the batch.
///
/// Constraints:
/// - `conditionOnPreviousText` must be `false` (no per-segment prompt) so that every
///   sequence in the batch shares an identical SOT prefix and the KV cache stays in sync.
/// - Temperature fallback is not performed; the caller is responsible for quality filtering.
class BatchedGreedyDecoder {
  let model: WhisperModel
  let tokenizer: WhisperTokenizer

  // Shared mask caches (same model/tokenizer → identical for all sequences)
  private var _cachedIndices: MLXArray?
  private var _cachedBaseMask: MLXArray?
  private var _cachedBaseMaskFirst: MLXArray?

  init(model: WhisperModel, tokenizer: WhisperTokenizer) {
    self.model = model
    self.tokenizer = tokenizer
  }

  private func ensureMasksCached(vocabSize: Int) {
    guard _cachedIndices == nil else { return }
    _cachedIndices = MLXArray.arange(vocabSize)

    var baseIds = tokenizer.nonSpeechTokens()
    baseIds.append(contentsOf: [
      tokenizer.transcribe, tokenizer.translate,
      tokenizer.sot, tokenizer.sotPrev,
      tokenizer.sotLm, tokenizer.noSpeech,
    ])
    var maskValues = [Float](repeating: 0.0, count: vocabSize)
    for id in baseIds where id < vocabSize { maskValues[id] = -Float.infinity }
    _cachedBaseMask = MLXArray(maskValues)

    var firstMaskValues = maskValues
    if let blankTokens = try? tokenizer.encode(" ") {
      for id in blankTokens where id < vocabSize { firstMaskValues[id] = -Float.infinity }
    }
    firstMaskValues[tokenizer.eot] = -Float.infinity
    _cachedBaseMaskFirst = MLXArray(firstMaskValues)
  }

  /// Decode a batch of audio segments in parallel.
  ///
  /// - Parameters:
  ///   - audioFeaturesStack: Stacked encoder outputs [B, n_audio_ctx, n_audio_state]
  ///   - options: Decoding options applied identically to every sequence
  /// - Returns: One `DecodingResult` per sequence (same order as the input batch)
  func decodeBatch(audioFeaturesStack: MLXArray, options: DecodingOptions) -> [DecodingResult] {
    let batchSize = audioFeaturesStack.shape[0]

    // Build the initial SOT token sequence, mirroring GreedyDecoder.decode().
    // When a prompt is provided, all B sequences in the batch share it — an
    // approximation of conditionOnPreviousText that preserves parallelism.
    var sotTokens: [Int] = []
    if !options.prompt.isEmpty {
      let maxPromptLen = (options.maxTokens / 2) - 1
      let truncatedPrompt =
        options.prompt.count > maxPromptLen
        ? Array(options.prompt.suffix(maxPromptLen))
        : options.prompt
      sotTokens.append(tokenizer.sotPrev)
      sotTokens.append(contentsOf: truncatedPrompt)
    }

    // sotIndex: the position of the first language/task token within sotTokens.
    // Used to extract the no-speech probability from the correct logit slice.
    let sotIndex = sotTokens.count

    sotTokens.append(contentsOf: tokenizer.sotSequence(language: options.language, task: options.task))
    if options.timestamps == .none {
      sotTokens.append(tokenizer.noTimestamps)
    }
    let initialTokenCount = sotTokens.count
    let maxGenerateTokens = options.maxTokens - initialTokenCount

    var tokensPerSeq: [[Int]] = Array(repeating: sotTokens, count: batchSize)
    var donePerSeq: [Bool] = Array(repeating: false, count: batchSize)
    var noSpeechProbPerSeq: [Float] = Array(repeating: 0.0, count: batchSize)
    var stepLogProbsPerSeq: [[MLXArray]] = Array(repeating: [], count: batchSize)

    // O(1) timestamp state per sequence (mirrors GreedyDecoder)
    var lastWasTimestamp: [Bool] = Array(repeating: false, count: batchSize)
    var penultimateWasTimestamp: [Bool] = Array(repeating: true, count: batchSize)  // vacuously true
    var lastTimestampTokenValue: [Int] = Array(repeating: 0, count: batchSize)

    var kvCache: [((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)]? = nil

    for iteration in 0 ..< maxGenerateTokens {
      // Build [B, n_tokens] input for this iteration
      let tokensToProcess: MLXArray
      if kvCache != nil {
        // Subsequent: one token per sequence [B, 1]
        // Done sequences keep feeding their last EOT token (KV cache must stay in sync)
        let lastTokens = (0 ..< batchSize).map { b -> Int32 in
          Int32(tokensPerSeq[b].last!)
        }
        tokensToProcess = MLXArray(lastTokens).reshaped(batchSize, 1)
      } else {
        // First: broadcast identical SOT sequence to all B sequences → [B, n_sot]
        let sotRow = MLXArray(sotTokens.map { Int32($0) })
        tokensToProcess = MLX.stacked(Array(repeating: sotRow, count: batchSize), axis: 0)
      }

      // GPU: forward pass for all B sequences simultaneously
      let (logits, newCache, _) = model.decode(
        tokensToProcess,
        audioFeatures: audioFeaturesStack,
        kvCache: kvCache
      )
      asyncEval(logits)
      kvCache = newCache

      // Sync GPU once to materialise all [B, vocab_size] logit slices
      let lastLogitsAll = logits[0..., -1, 0...]  // [B, vocab_size]
      eval(lastLogitsAll)
      let vocabSize = lastLogitsAll.shape[1]
      ensureMasksCached(vocabSize: vocabSize)

      // Capture SOT-position logits for no-speech extraction at iteration 0
      // (logits is already evaluated above, so this is a cheap CPU slice)
      var sotLogitsAll: MLXArray? = nil
      if iteration == 0 {
        sotLogitsAll = logits[0..., sotIndex, 0...]  // [B, vocab_size]
        eval(sotLogitsAll!)
      }

      // Per-sequence CPU processing (masks are cheap; GPU work already done)
      let indices = _cachedIndices!

      for b in 0 ..< batchSize {
        if donePerSeq[b] { continue }

        let numGenerated = tokensPerSeq[b].count - initialTokenCount
        let lastLogits = lastLogitsAll[b]  // [vocab_size], already evaluated

        // --- No-speech probability (iteration 0 only) ---
        if iteration == 0, let sotAll = sotLogitsAll {
          let probs = MLX.softmax(sotAll[b], axis: -1)
          noSpeechProbPerSeq[b] = probs[tokenizer.noSpeech].item(Float.self)
        }

        // --- Base suppression mask ---
        let baseMask = iteration == 0 ? _cachedBaseMaskFirst! : _cachedBaseMask!

        // --- Timestamp rules mask ---
        var timestampMask = MLXArray.zeros([vocabSize])
        if options.timestamps != .none {
          let boolState = MLXArray([
            Int32(lastWasTimestamp[b] ? 1 : 0),
            Int32(penultimateWasTimestamp[b] ? 1 : 0),
          ])
          let intState = MLXArray([
            Int32(lastTimestampTokenValue[b]),
            Int32(numGenerated),
            Int32(tokenizer.timestampBegin),
            Int32(tokenizer.noTimestamps),
            Int32(tokenizer.eot),
          ])
          timestampMask = compiledTimestampRules(indices, boolState, intState)
        }

        // --- Timestamp force heuristic ---
        if options.timestamps != .none, numGenerated > 0 {
          let tsBeginScalar = MLXArray(Int32(tokenizer.timestampBegin))
          let forceMask = compiledTimestampForce(lastLogits, indices, tsBeginScalar)
          timestampMask = MLX.minimum(timestampMask, forceMask)
        }

        // --- N-gram repetition ban ---
        // Build a suppression mask for any token that would complete a repeated n-gram.
        // Done on CPU before the argmax — cheap because banned sets are typically small.
        var ngramMask = MLXArray.zeros([vocabSize])
        if options.noRepeatNgramSize >= 2 {
          let banned = bannedNgramTokens(
            tokens: tokensPerSeq[b],
            initialTokenCount: initialTokenCount,
            n: options.noRepeatNgramSize
          )
          if !banned.isEmpty {
            var ngramVals = [Float](repeating: 0.0, count: vocabSize)
            for token in banned where token < vocabSize {
              ngramVals[token] = -Float.infinity
            }
            ngramMask = MLXArray(ngramVals)
          }
        }

        // --- Combine & apply ---
        let finalMask = MLX.minimum(MLX.minimum(baseMask, timestampMask), ngramMask)
        let maskedLogits = compiledApplyMask(lastLogits, finalMask)

        // --- Sample ---
        let nextTokenId: Int
        if options.temperature == 0.0 {
          nextTokenId = Int(compiledArgmax(maskedLogits).item(Int32.self))
        } else {
          let probs = MLX.softmax(maskedLogits / options.temperature, axis: -1)
          nextTokenId = Int(MLXRandom.categorical(MLX.log(probs + 1e-10)).item(Int32.self))
        }

        // --- Accumulate log-prob (lazy, non-EOT only) ---
        if nextTokenId != tokenizer.eot {
          let lse = MLX.logSumExp(lastLogits, axes: [-1])
          let logProb = MLX.take(lastLogits, MLXArray(Int32(nextTokenId))) - lse
          stepLogProbsPerSeq[b].append(logProb)
        }

        tokensPerSeq[b].append(nextTokenId)

        // --- Update timestamp state (mirrors GreedyDecoder exactly) ---
        let isTs = nextTokenId >= tokenizer.timestampBegin
        penultimateWasTimestamp[b] = (numGenerated == 0) ? true : lastWasTimestamp[b]
        lastWasTimestamp[b] = isTs
        if isTs {
          lastTimestampTokenValue[b] = nextTokenId
          if penultimateWasTimestamp[b] { lastTimestampTokenValue[b] += 1 }
        }

        if nextTokenId == tokenizer.eot {
          donePerSeq[b] = true
        }
      }

      if donePerSeq.allSatisfy({ $0 }) { break }
    }

    return (0 ..< batchSize).map { b in
      var generatedTokens = Array(tokensPerSeq[b][initialTokenCount...])
      // Strip EOT (matches GreedyDecoder)
      if let eotIndex = generatedTokens.firstIndex(of: tokenizer.eot) {
        generatedTokens = Array(generatedTokens[..<eotIndex])
      }
      let text = tokenizer.decode(generatedTokens)

      let avgLogProb: Float
      if stepLogProbsPerSeq[b].isEmpty {
        avgLogProb = 0.0
      } else {
        let summed = MLX.stacked(stepLogProbsPerSeq[b], axis: 0).sum()
        eval(summed)
        avgLogProb = summed.item(Float.self) / Float(stepLogProbsPerSeq[b].count)
      }

      return DecodingResult(
        tokens: generatedTokens,
        text: text,
        avgLogProb: avgLogProb,
        noSpeechProb: noSpeechProbPerSeq[b],
        temperature: options.temperature,
        compressionRatio: computeCompressionRatio(text)
      )
    }
  }
}
