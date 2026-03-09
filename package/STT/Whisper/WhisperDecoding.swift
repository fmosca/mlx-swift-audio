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

  init(
    task: TranscriptionTask = .transcribe,
    language: String? = nil,
    temperature: Float = 0.0,
    maxTokens: Int = 448,
    timestamps: TimestampGranularity = .segment,
    prompt: [Int] = []
  ) {
    self.task = task
    self.language = language
    self.temperature = temperature
    self.maxTokens = maxTokens
    self.timestamps = timestamps
    self.prompt = prompt
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

// MARK: - Cache Flattening for asyncEval

/// Flatten Whisper KV cache into [MLXArray] for asyncEval.
/// asyncEval's collect() does not handle Optional, so we flatten manually.
private func flattenWhisperCache(
  _ cache: [((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)]?
) -> [MLXArray] {
  guard let cache = cache else { return [] }
  var result: [MLXArray] = []
  for layer in cache {
    if let sa = layer.0 { result.append(contentsOf: [sa.0, sa.1]) }
    if let ca = layer.1 { result.append(contentsOf: [ca.0, ca.1]) }
  }
  return result
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

  init(model: WhisperModel, tokenizer: WhisperTokenizer) {
    self.model = model
    self.tokenizer = tokenizer
  }

  /// Decode from mel spectrogram (encodes then decodes).
  /// Use `decode(audioFeatures:options:)` when reusing encoder output across temperature fallbacks.
  ///
  /// - Parameters:
  ///   - mel: Mel spectrogram (batch=1, n_mels, n_frames)
  ///   - options: Decoding options (temperature, prompt, etc.)
  /// - Returns: Decoding result
  func decode(_ mel: MLXArray, options: DecodingOptions) -> DecodingResult {
    let audioFeatures = model.encode(mel)
    eval(audioFeatures)
    return decode(audioFeatures: audioFeatures, options: options)
  }

  /// Decode from precomputed audio features.
  /// Reuse audio features across temperature fallbacks to avoid redundant encoder passes.
  ///
  /// - Parameters:
  ///   - audioFeatures: Output of `model.encode(mel)` (batch=1, n_audio_ctx, n_audio_state)
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

    // Pre-compute constant arrays used in every iteration of the decode loop.
    // Building these inside the loop (the previous approach) allocates ~207 KB
    // of CPU memory per token step — ~1.2 GB for a typical 34-minute meeting.
    let vocabSizeForMask: Int = {
      // Encode a dummy token to get the vocab size from the model
      // We'll capture the actual vocabSize from the first logits call instead
      // Use a sentinel; it will be replaced after the first forward pass.
      0
    }()
    _ = vocabSizeForMask  // suppress unused warning; real masks built after first pass

    // Autoregressive decoding with async eval pipelining.
    // Pattern: overlap GPU compute (iteration N) with CPU sync for token (iteration N-1).
    // See FunASR/FunASRSTT.swift and T3Turbo.swift for reference.
    var kvCache: [((MLXArray, MLXArray)?, (MLXArray, MLXArray)?)]? = nil
    var noSpeechProb: Float = 0.0
    var cachedLogits: [MLXArray] = []
    var selectedTokenIndices: [Int] = []

    // Masks pre-computed after the first forward pass (need vocabSize from logits)
    var cachedIndices: MLXArray? = nil
    var cachedBaseMask: MLXArray? = nil          // without SuppressBlank (iterations > 0)
    var cachedBaseMaskFirst: MLXArray? = nil     // with SuppressBlank (iteration 0)

    // Lazy token from previous iteration; we sync for it at start of next iteration
    // after kicking off the forward pass (async pipelining).
    var prevTokenLazy: MLXArray? = nil

    for iteration in 0 ..< maxGenerateTokens {
      // Convert tokens to MLXArray
      // With KV caching: only pass new token(s), not all tokens
      let tokensToProcess: MLXArray
      if let prev = prevTokenLazy {
        // Pipelined: use lazy token from previous iteration (no sync yet)
        tokensToProcess = prev.expandedDimensions(axis: 0)
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

      // Iteration 0: must sync for noSpeechProb and mask building.
      // Iteration >= 1: asyncEval to overlap GPU compute with CPU sync for prev token.
      if iteration == 0 {
        eval(logits)
      } else {
        var toEval: [MLXArray] = [logits]
        toEval.append(contentsOf: flattenWhisperCache(newCache))
        asyncEval(toEval)
      }

      // Sync for previous token (iteration >= 1). GPU is already computing current step.
      if let prev = prevTokenLazy {
        let prevTokenId = Int(prev.item(Int32.self))
        tokens.append(prevTokenId)
        if prevTokenId == tokenizer.eot {
          break
        }
      }

      // Update KV cache
      kvCache = newCache

      // Compute no-speech probability from first forward pass
      if iteration == 0 {
        let sotLogits = logits[0, sotIndex]
        let probs = MLX.softmax(sotLogits, axis: -1)
        noSpeechProb = probs[tokenizer.noSpeech].item(Float.self)
      }

      // Get logits for last token
      var lastLogits = logits[0, -1]
      let vocabSize = lastLogits.shape[0]

      // Get number of generated tokens (excluding initial SOT sequence)
      let numGenerated = tokens.count - initialTokenCount

      // =============================================================================
      // STEP 1: Base suppression mask (SuppressBlank + SuppressTokens)
      // =============================================================================
      if cachedIndices == nil {
        cachedIndices = MLXArray(Int32(0) ..< Int32(vocabSize))

        var baseIds = tokenizer.nonSpeechTokens()
        baseIds.append(contentsOf: [
          tokenizer.transcribe, tokenizer.translate,
          tokenizer.sot, tokenizer.sotPrev,
          tokenizer.sotLm, tokenizer.noSpeech,
        ])
        var maskValues = [Float](repeating: 0.0, count: vocabSize)
        for id in baseIds where id < vocabSize { maskValues[id] = -Float.infinity }
        cachedBaseMask = MLXArray(maskValues)

        var firstMaskValues = maskValues
        if let blankTokens = try? tokenizer.encode(" ") {
          for id in blankTokens where id < vocabSize { firstMaskValues[id] = -Float.infinity }
        }
        firstMaskValues[tokenizer.eot] = -Float.infinity
        cachedBaseMaskFirst = MLXArray(firstMaskValues)
      }

      let indices = cachedIndices!
      let baseMask = iteration == 0 ? cachedBaseMaskFirst! : cachedBaseMask!

      // =============================================================================
      // STEP 2: Build timestamp rules mask (ApplyTimestampRules internal mask)
      // =============================================================================
      var timestampMask = MLXArray.zeros([vocabSize])

      if options.timestamps != .none {
        timestampMask = MLX.where(
          indices .== Int32(tokenizer.noTimestamps),
          MLXArray(-Float.infinity),
          timestampMask
        )

        let lastWasTimestamp = numGenerated >= 1 && tokens.last! >= tokenizer.timestampBegin
        let penultimateWasTimestamp = numGenerated < 2 || tokens[tokens.count - 2] >= tokenizer.timestampBegin

        if lastWasTimestamp {
          if penultimateWasTimestamp {
            timestampMask = MLX.where(
              indices .>= Int32(tokenizer.timestampBegin),
              MLXArray(-Float.infinity),
              timestampMask
            )
          } else {
            timestampMask = MLX.where(
              indices .< Int32(tokenizer.eot),
              MLXArray(-Float.infinity),
              timestampMask
            )
          }
        }

        let generatedTokens = tokens.suffix(numGenerated)
        let timestampTokenValues = generatedTokens.compactMap { token -> Int? in
          token > tokenizer.timestampBegin ? token : nil
        }

        if !timestampTokenValues.isEmpty {
          var lastTimestampToken = timestampTokenValues.last!
          if penultimateWasTimestamp {
            lastTimestampToken += 1
          }
          let lowerCond = indices .>= Int32(tokenizer.timestampBegin)
          let upperCond = indices .< Int32(lastTimestampToken)
          let rangeCond = MLX.logicalAnd(lowerCond, upperCond)
          timestampMask = MLX.where(rangeCond, MLXArray(-Float.infinity), timestampMask)
        }

        if numGenerated == 0 {
          timestampMask = MLX.where(
            indices .< Int32(tokenizer.timestampBegin),
            MLXArray(-Float.infinity),
            timestampMask
          )

          let maxInitialTimestampIndex = 50
          let lastAllowed = tokenizer.timestampBegin + maxInitialTimestampIndex
          if lastAllowed < vocabSize {
            timestampMask = MLX.where(
              indices .> Int32(lastAllowed),
              MLXArray(-Float.infinity),
              timestampMask
            )
          }
        }
      }

      // =============================================================================
      // STEP 3: Apply timestamp probability heuristic (Python lines 381-394)
      // =============================================================================
      if options.timestamps != .none, numGenerated > 0 {
        let logProbs = lastLogits - MLX.logSumExp(lastLogits, axes: [-1], keepDims: true)
        let timestampLogProbSum = MLX.logSumExp(logProbs[tokenizer.timestampBegin...], axes: [-1], keepDims: true)
        let maxTextLogProb = logProbs[0 ..< tokenizer.timestampBegin].max(axes: [-1], keepDims: true)
        let shouldForce = timestampLogProbSum .> maxTextLogProb

        timestampMask = MLX.where(
          MLX.logicalAnd(shouldForce, indices .< Int32(tokenizer.timestampBegin)),
          MLXArray(-Float.infinity),
          timestampMask
        )
      }

      // =============================================================================
      // STEP 4: Combine masks and apply
      // =============================================================================
      let finalMask = MLX.minimum(baseMask, timestampMask)
      lastLogits = lastLogits + finalMask

      // Sample next token (lazy MLXArray; sync deferred until next iteration or accumulation)
      let nextTokenLazy: MLXArray
      if options.temperature == 0.0 {
        nextTokenLazy = MLX.argMax(lastLogits, axis: -1)
      } else {
        let probs = MLX.softmax(lastLogits / options.temperature, axis: -1)
        nextTokenLazy = MLXRandom.categorical(MLX.log(probs + 1e-10))
      }

      // Accumulate for avgLogProb; must sync for token value
      let nextTokenId = Int(nextTokenLazy.item(Int32.self))
      if nextTokenId != tokenizer.eot {
        cachedLogits.append(lastLogits)
        selectedTokenIndices.append(nextTokenId)
      }

      prevTokenLazy = nextTokenLazy

      // EOT check: we'll append and break at start of next iteration, or exit loop
      if nextTokenId == tokenizer.eot {
        break
      }
    }

    // Append final token when loop exits normally (maxGenerateTokens reached)
    if let prev = prevTokenLazy {
      let prevTokenId = Int(prev.item(Int32.self))
      if prevTokenId != tokenizer.eot {
        tokens.append(prevTokenId)
      }
    }

    // Compute avgLogProb in a single batch pass over all cached logits,
    // avoiding per-step GPU→CPU round trips.
    let avgLogProb: Float
    if selectedTokenIndices.isEmpty {
      avgLogProb = 0.0
    } else {
      let numSteps = selectedTokenIndices.count
      let stackedLogits = MLX.stacked(cachedLogits, axis: 0)  // [numSteps, vocabSize]
      let allLogProbs = MLX.log(MLX.softmax(stackedLogits, axis: -1))  // [numSteps, vocabSize]
      let cachedVocabSize = cachedLogits[0].shape[0]
      let linearIdxs = (0..<numSteps).map {
        Int32($0) * Int32(cachedVocabSize) + Int32(selectedTokenIndices[$0])
      }
      let selectedLogProbs = MLX.take(allLogProbs.flattened(), MLXArray(linearIdxs))
      let sumLogProbMlx = selectedLogProbs.sum()
      eval(sumLogProbMlx)
      avgLogProb = sumLogProbMlx.item(Float.self) / Float(numSteps)
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
