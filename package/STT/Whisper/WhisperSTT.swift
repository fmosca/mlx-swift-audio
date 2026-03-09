// Copyright © 2022 OpenAI (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/openai/whisper
// License: licenses/whisper.txt

import FluidAudio
import Foundation
import MLX
import Synchronization

/// Actor wrapper for Whisper model that provides thread-safe transcription
actor WhisperSTT {
  // MARK: - Properties

  // Model is nonisolated(unsafe) because it contains non-Sendable types (MLXArray)
  // but is only accessed within the actor's methods
  nonisolated(unsafe) let model: WhisperModel
  nonisolated(unsafe) let tokenizer: WhisperTokenizer

  // MARK: - Initialization

  // Lazily initialised on the first transcribeBatched() call; cached for reuse.
  private var vadManager: VadManager?

  private init(model: WhisperModel, tokenizer: WhisperTokenizer) {
    self.model = model
    self.tokenizer = tokenizer
  }

  /// Coalesces consecutive VAD segments into larger chunks without exceeding `maxDuration`.
  ///
  /// Silero produces many short segments (median ~6 s on real meeting audio).  Merging
  /// them packs more speech into each Whisper context window, reducing the total number
  /// of encoder passes and improving GPU utilisation in the batched path.
  ///
  /// The merge is greedy left-to-right: extend the current merged segment with the next
  /// VAD segment as long as the combined span stays within `maxDuration`.  When it would
  /// overflow, flush the current segment and start a new one.  This preserves chronological
  /// order and never crosses a silence boundary that Silero deemed significant enough to
  /// split on — it only skips *within* the available slack.
  private func mergeAdjacentSegments(_ segments: [VadSegment], maxDuration: TimeInterval = 29.0) -> [VadSegment] {
    guard !segments.isEmpty else { return [] }
    var merged: [VadSegment] = []
    var mergeStart = segments[0].startTime
    var mergeEnd = segments[0].endTime

    for seg in segments.dropFirst() {
      if seg.endTime - mergeStart <= maxDuration {
        mergeEnd = seg.endTime
      } else {
        merged.append(VadSegment(startTime: mergeStart, endTime: mergeEnd))
        mergeStart = seg.startTime
        mergeEnd = seg.endTime
      }
    }
    merged.append(VadSegment(startTime: mergeStart, endTime: mergeEnd))
    return merged
  }

  private func getVadManager() async throws -> VadManager {
    if let existing = vadManager { return existing }
    // VadManager resolves its model directory from Application Support by default,
    // which is where FluidAudio caches the compiled .mlmodelc on first download.
    let manager = try await VadManager(config: VadConfig(computeUnits: .cpuAndNeuralEngine))
    vadManager = manager
    return manager
  }

  /// Load WhisperSTT from Hugging Face Hub by model size enum.
  ///
  /// - Parameters:
  ///   - modelSize: Model size to load
  ///   - quantization: Quantization level (fp16, 8bit, 4bit). Default is 4bit.
  ///   - progressHandler: Optional callback for download/load progress
  /// - Returns: Initialized WhisperSTT instance
  static func load(
    modelSize: WhisperModelSize,
    quantization: WhisperQuantization = .q4,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
  ) async throws -> WhisperSTT {
    let model = try await WhisperModel.load(
      modelSize: modelSize,
      quantization: quantization,
      progressHandler: progressHandler
    )
    return try await finishLoading(model: model)
  }

  /// Load WhisperSTT from an arbitrary Hugging Face repository ID.
  ///
  /// - Parameters:
  ///   - repoId: Full HuggingFace repository identifier (e.g. "mlx-community/whisper-large-v3-turbo")
  ///   - quantization: Quantization level, used only for the MLX quantize pass when weights
  ///     contain `.scales` keys. Default is fp16 (no quantization pass).
  ///   - progressHandler: Optional callback for download/load progress
  /// - Returns: Initialized WhisperSTT instance
  static func load(
    repoId: String,
    quantization: WhisperQuantization = .fp16,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
  ) async throws -> WhisperSTT {
    let model = try await WhisperModel.load(
      repoId: repoId,
      quantization: quantization,
      progressHandler: progressHandler
    )
    return try await finishLoading(model: model)
  }

  private static func finishLoading(model: sending WhisperModel) async throws -> WhisperSTT {

    // Then load tokenizer (fast operation) with correct vocabulary for model type
    // Pass model directory so tokenizer can find vocab files bundled with the model
    // numLanguages must match the model's actual language count (from n_vocab)
    // Different models have different counts (base: 99, large-v3-turbo: 100)
    let tokenizer = try await WhisperTokenizer.load(
      isMultilingual: model.isMultilingual,
      numLanguages: model.numLanguages,
      modelDirectory: model.modelDirectory
    )

    // Validate tokenizer configuration matches model expectations
    // This catches critical bugs like off-by-one errors in special token IDs
    let modelVocabSize = model.dims.n_vocab

    // Verify key special tokens are in valid range
    let maxTokenId = max(
      tokenizer.eot,
      tokenizer.sot,
      tokenizer.translate,
      tokenizer.transcribe,
      tokenizer.noSpeech,
      tokenizer.timestampBegin
    )

    if maxTokenId >= modelVocabSize {
      throw STTError.invalidArgument(
        """
        Tokenizer misconfiguration: token ID \(maxTokenId) >= model vocab size \(modelVocabSize). \
        This indicates a critical bug in tokenizer setup.
        """
      )
    }

    // Verify critical token IDs are consistent with model's numLanguages
    // Token IDs depend on numLanguages which varies by model (base: 99, large-v3-turbo: 100)
    let expectedBaseVocab = model.isMultilingual ? 50257 : 50256
    let expectedEot = expectedBaseVocab
    let expectedSot = expectedBaseVocab + 1
    let expectedTranscribe = expectedSot + 1 + model.numLanguages + 1 // sot + numLangs + translate + transcribe
    let expectedTimestampBegin = expectedTranscribe + 5 // transcribe + sotLm + sotPrev + noSpeech + noTimestamps + first_timestamp

    assert(tokenizer.eot == expectedEot, "EOT token mismatch: got \(tokenizer.eot), expected \(expectedEot)")
    assert(tokenizer.sot == expectedSot, "SOT token mismatch: got \(tokenizer.sot), expected \(expectedSot)")
    assert(tokenizer.transcribe == expectedTranscribe, "Transcribe token mismatch: got \(tokenizer.transcribe), expected \(expectedTranscribe)")
    assert(tokenizer.timestampBegin == expectedTimestampBegin, "TimestampBegin mismatch: got \(tokenizer.timestampBegin), expected \(expectedTimestampBegin)")

    return WhisperSTT(model: model, tokenizer: tokenizer)
  }

  // MARK: - Transcription

  /// Transcribe audio to text using seek-based processing (matching Python implementation)
  ///
  /// This uses a seek pointer to move through the audio, with content-aware advancement
  /// based on decoded timestamps and word boundaries. This matches Python's implementation
  /// and provides better handling of long audio with silence or boundary cases.
  ///
  /// - Parameters:
  ///   - audio: Audio waveform (T,) in 16 kHz
  ///   - language: Optional language code (e.g., "en", "zh"), nil for auto-detect
  ///   - task: Transcription task (transcribe or translate)
  ///   - temperature: Sampling temperature (0.0 for greedy)
  ///   - timestamps: Timestamp granularity
  ///   - conditionOnPreviousText: Whether to use previous segment's output as prompt (default: true)
  ///   - noSpeechThreshold: Skip segments with no_speech_prob > threshold (default: 0.6)
  ///   - logprobThreshold: Skip if avg_logprob < threshold (default: -1.0)
  ///   - compressionRatioThreshold: Retry with higher temperature if compression ratio > threshold (default: 2.4)
  ///     High compression ratio indicates repetitive text (potential hallucination).
  ///   - hallucinationSilenceThreshold: When word timestamps are enabled, skip silent periods
  ///     longer than this threshold (in seconds) when a possible hallucination is detected.
  ///     Set to nil (default) to disable hallucination filtering.
  /// - Returns: Transcription result
  func transcribe(
    audio: MLXArray,
    language: String?,
    task: TranscriptionTask,
    temperature: Float,
    timestamps: TimestampGranularity,
    conditionOnPreviousText: Bool = true,
    noSpeechThreshold: Float? = 0.6,
    logprobThreshold: Float? = -1.0,
    compressionRatioThreshold: Float? = 2.4,
    hallucinationSilenceThreshold: Float? = nil
  ) -> TranscriptionResult {
    let transcribeStartTime = CFAbsoluteTimeGetCurrent()

    // Constants matching Python
    let nFrames = WhisperAudio.nFrames // 3000 frames per 30s segment
    let hopLength = WhisperAudio.hopLength // 160
    let sampleRate = WhisperAudio.sampleRate // 16000
    let framesPerSecond = WhisperAudio.framesPerSecond // 100
    let inputStride = nFrames / model.dims.n_audio_ctx // mel frames per output token: 2
    let timePrecision = Float(inputStride * hopLength) / Float(sampleRate) // 0.02 seconds per token

    // Pad audio with 30 seconds of silence for boundary handling
    let paddedAudio = MLX.concatenated([audio, MLXArray.zeros([WhisperAudio.nSamples])], axis: 0)

    // Compute mel spectrogram for entire audio (with padding)
    // Returns (n_frames, n_mels) - already in the right shape for Conv1d
    let fullMel = whisperLogMelSpectrogram(audio: paddedAudio, nMels: model.dims.n_mels)
    eval(fullMel)

    // Content frames (excluding padding)
    let contentFrames = audio.shape[0] / hopLength
    let contentDuration = Float(contentFrames * hopLength) / Float(sampleRate)

    Log.model.info("Transcribing \(String(format: "%.1f", contentDuration))s audio with seek-based processing")

    // Detect language if not specified
    var detectedLanguage: String? = nil
    if language == nil {
      let melSegment = padOrTrimMel(fullMel[0 ..< nFrames], length: nFrames)
      let batchedMel = melSegment.expandedDimensions(axis: 0).asType(.float16)
      let (lang, prob) = detectLanguageFromMel(batchedMel)
      detectedLanguage = lang
      Log.model.info("Detected language: \(lang) (probability: \(String(format: "%.2f", prob)))")
    }
    let languageToUse = language ?? detectedLanguage ?? "en"

    // Seek-based transcription loop
    var seek = 0
    var allTokens: [Int] = []
    var allSegments: [TranscriptionSegment] = []
    var promptResetSince = 0
    var lastSpeechTimestamp: Float = 0.0

    // Reuse decoder across segments and temperature fallbacks to reduce allocation overhead
    let decoder = GreedyDecoder(model: model, tokenizer: tokenizer)

    while seek < contentFrames {
      let timeOffset = Float(seek * hopLength) / Float(sampleRate)
      let windowEndTime = Float((seek + nFrames) * hopLength) / Float(sampleRate)
      let segmentSize = min(nFrames, contentFrames - seek)
      let segmentDuration = Float(segmentSize * hopLength) / Float(sampleRate)

      Log.model.debug("Processing segment: seek=\(seek) (\(String(format: "%.2f", timeOffset))s), size=\(segmentSize) frames (\(String(format: "%.2f", segmentDuration))s)")

      // Extract mel segment and pad to nFrames
      // Cast to float16 to match Python's behavior (line 612-614 in whisper.py)
      let melSegment = padOrTrimMel(fullMel[seek ..< (seek + segmentSize)], length: nFrames)
      let batchedMel = melSegment.expandedDimensions(axis: 0).asType(.float16)

      // Build prompt from previous tokens (if conditioning enabled)
      // Use tokens since last prompt reset (matches Python: all_tokens[prompt_reset_since:])
      let promptTokens = conditionOnPreviousText ? Array(allTokens[promptResetSince...]) : []
      let prompt = promptTokens

      // Temperature fallback loop (matches Python's decode_with_fallback)
      // Try increasing temperatures when output is too repetitive (high compression ratio)
      // or has low confidence (low avg_logprob)
      //
      // Optimization: Encode once per segment, decode multiple times with different
      // temperatures. Encoder pass is expensive; decoder only needs temperature change.
      let temperatureFallbackSequence: [Float] = segmentDuration < 2.0
        ? [0.0, 0.5, 1.0] // Short segments: 3 steps
        : [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] // Normal segments: 6 steps
      var result: DecodingResult!

      // Encode once per segment (expensive); reuse across temperature fallbacks
      let audioFeatures = model.compiledEncode(batchedMel)
      eval(audioFeatures)

      for currentTemperature in temperatureFallbackSequence {
        let options = DecodingOptions(
          task: task,
          language: languageToUse,
          temperature: currentTemperature,
          maxTokens: 448,
          timestamps: timestamps,
          prompt: prompt
        )

        result = decoder.decode(audioFeatures: audioFeatures, options: options)

        // Check if we need to retry with higher temperature
        var needsFallback = false

        // Too repetitive (high compression ratio indicates hallucination)
        if let crThreshold = compressionRatioThreshold,
           result.compressionRatio > crThreshold
        {
          needsFallback = true
          Log.model.debug("Compression ratio \(String(format: "%.2f", result.compressionRatio)) > \(crThreshold), retrying with higher temperature")
        }

        // Too low confidence
        if let lpThreshold = logprobThreshold,
           result.avgLogProb < lpThreshold
        {
          needsFallback = true
          Log.model.debug("Avg log prob \(String(format: "%.2f", result.avgLogProb)) < \(lpThreshold), retrying with higher temperature")
        }

        // If it's likely silence, accept the result and don't retry
        // Python: if no_speech_prob > no_speech_threshold: needs_fallback = False
        if let nsThreshold = noSpeechThreshold,
           result.noSpeechProb > nsThreshold
        {
          needsFallback = false
        }

        if !needsFallback {
          break
        }

        // If we're at the last temperature, use whatever we got
        if currentTemperature == temperatureFallbackSequence.last {
          Log.model.warning("All temperature fallbacks exhausted, using final result")
        }
      }

      let decodedText = tokenizer.decode(result.tokens.filter { $0 < tokenizer.eot })
      // Log timestamp tokens for debugging
      let tsTokens = result.tokens.filter { $0 >= tokenizer.timestampBegin }
      let tsPositions = tsTokens.map { Float($0 - tokenizer.timestampBegin) * 0.02 }
      Log.model.debug("Decoded: noSpeechProb=\(String(format: "%.3f", result.noSpeechProb)), avgLogProb=\(String(format: "%.3f", result.avgLogProb)), tokens=\(result.tokens.count), timestamps=\(tsPositions), text='\(decodedText.prefix(100))'")

      // No-speech detection: skip if no_speech_prob > threshold
      if let nsThreshold = noSpeechThreshold {
        var shouldSkip = result.noSpeechProb > nsThreshold

        // Don't skip if logprob is high enough despite high no_speech_prob
        if let lpThreshold = logprobThreshold, result.avgLogProb > lpThreshold {
          shouldSkip = false
        }

        if shouldSkip {
          Log.model.debug("Skipping segment due to no-speech detection (prob=\(String(format: "%.3f", result.noSpeechProb)) > \(nsThreshold))")
          seek += segmentSize
          continue
        }
      }

      let previousSeek = seek
      var currentSegments: [TranscriptionSegment] = []

      // Parse tokens to extract segments based on timestamps
      let tokens = result.tokens
      let timestampTokens = tokens.map { $0 >= tokenizer.timestampBegin }

      // Find consecutive timestamp pairs
      var consecutiveIndices: [Int] = []
      if timestampTokens.count >= 2 {
        for i in 0 ..< (timestampTokens.count - 1) {
          if timestampTokens[i], timestampTokens[i + 1] {
            consecutiveIndices.append(i + 1)
          }
        }
      }

      // Check for single timestamp ending
      let singleTimestampEnding = timestampTokens.count >= 2 &&
        !timestampTokens[timestampTokens.count - 2] &&
        timestampTokens[timestampTokens.count - 1]

      if !consecutiveIndices.isEmpty {
        // Multiple segments based on consecutive timestamps
        var slices = consecutiveIndices
        if singleTimestampEnding {
          slices.append(tokens.count)
        }

        var lastSlice = 0
        for currentSlice in slices {
          let slicedTokens = Array(tokens[lastSlice ..< currentSlice])
          guard slicedTokens.count >= 2 else {
            lastSlice = currentSlice
            continue
          }

          let startTimestampPos = slicedTokens[0] - tokenizer.timestampBegin
          let endTimestampPos = slicedTokens[slicedTokens.count - 1] - tokenizer.timestampBegin

          let segmentStart = timeOffset + Float(startTimestampPos) * timePrecision
          let segmentEnd = timeOffset + Float(endTimestampPos) * timePrecision

          // Extract text tokens for this slice
          let textTokens = slicedTokens.filter { $0 < tokenizer.eot }
          let text = tokenizer.decode(textTokens)

          let segment = TranscriptionSegment(
            text: text,
            start: TimeInterval(segmentStart),
            end: TimeInterval(segmentEnd),
            tokens: slicedTokens,
            avgLogProb: result.avgLogProb,
            noSpeechProb: result.noSpeechProb,
            words: nil
          )
          currentSegments.append(segment)

          lastSlice = currentSlice
        }

        // Advance seek based on timestamps
        // When single_timestamp_ending and there's remaining audio,
        // advance to the timestamp position instead of full segment to avoid
        // skipping content in short audio clips.
        if singleTimestampEnding {
          if let lastTimestamp = tokens.last, lastTimestamp != tokenizer.timestampBegin {
            let lastTimestampPos = lastTimestamp - tokenizer.timestampBegin
            let timestampSeek = lastTimestampPos * inputStride
            if seek + timestampSeek < contentFrames {
              seek += timestampSeek
            } else {
              seek += segmentSize
            }
          } else {
            seek += segmentSize
          }
        } else {
          let lastTimestampPos = tokens[consecutiveIndices.last! - 1] - tokenizer.timestampBegin
          // Sanity check: don't let hallucinated timestamps cause seek to jump beyond segment.
          // The model can hallucinate timestamps pointing far into the future (e.g., 25s when
          // only 2s of audio remains), which would cause seek to jump past the end of content.
          let maxSeekAdvance = segmentSize
          let timestampSeek = lastTimestampPos * inputStride
          seek += min(timestampSeek, maxSeekAdvance)
        }
      } else {
        // Single segment (no consecutive timestamps)
        // Python: duration = segment_duration, then check for last timestamp
        var duration = segmentDuration

        // Find last timestamp token if any
        // Python: timestamps = tokens[timestamp_tokens.nonzero()[0]]
        let timestampIndices = tokens.enumerated().compactMap { i, t in t >= tokenizer.timestampBegin ? i : nil }
        if let lastIdx = timestampIndices.last, tokens[lastIdx] != tokenizer.timestampBegin {
          // Python: last_timestamp_pos = timestamps[-1].item() - tokenizer.timestamp_begin
          let lastTimestampPos = tokens[lastIdx] - tokenizer.timestampBegin
          duration = Float(lastTimestampPos) * timePrecision
        }

        let textTokens = tokens.filter { $0 < tokenizer.eot }
        let text = tokenizer.decode(textTokens)

        let segment = TranscriptionSegment(
          text: text,
          start: TimeInterval(timeOffset),
          end: TimeInterval(timeOffset + duration),
          tokens: tokens,
          avgLogProb: result.avgLogProb,
          noSpeechProb: result.noSpeechProb,
          words: nil
        )
        currentSegments.append(segment)

        // Python: seek += segment_size (ALWAYS advance by full segment, not duration)
        // The duration is only used for segment end time, not seek advancement
        //
        // When there's a single timestamp ending and remaining audio exists,
        // advance to the timestamp position instead of full segment to avoid
        // skipping content in short audio clips.
        if singleTimestampEnding, let lastIdx = timestampIndices.last, tokens[lastIdx] != tokenizer.timestampBegin {
          let lastTimestampPos = tokens[lastIdx] - tokenizer.timestampBegin
          let timestampSeek = lastTimestampPos * inputStride
          // Only use timestamp-based seek if there's remaining audio
          if seek + timestampSeek < contentFrames {
            seek += timestampSeek
          } else {
            seek += segmentSize
          }
        } else {
          seek += segmentSize
        }
      }

      // Ensure seek never moves backward (WhisperKit safety mechanism)
      seek = max(previousSeek, seek)

      Log.model.debug("Seek advanced: \(previousSeek) -> \(seek) (\(String(format: "%.2f", Float(seek * hopLength) / Float(sampleRate)))s), segments=\(currentSegments.count)")

      // Filter out zero-length segments (WhisperKit approach)
      currentSegments = currentSegments.filter { $0.end > $0.start }

      // Filter out segments with timestamps that exceed the segment window
      // This catches hallucinations where model generates impossible timestamps (e.g., 20s in a 2s segment)
      currentSegments = currentSegments.filter { segment in
        let relativeEnd = Float(segment.end) - timeOffset
        if relativeEnd > segmentDuration + 1.0 { // Allow 1s tolerance
          Log.model.warning("Filtering hallucinated segment (timestamp \(String(format: "%.1f", relativeEnd))s exceeds \(String(format: "%.1f", segmentDuration))s window): '\(segment.text.prefix(50))'")
          return false
        }
        return true
      }

      // Filter segments with very low confidence after temperature exhaustion
      // These are likely hallucinations from silence/unclear audio at end of content
      if result.temperature >= 0.8, result.avgLogProb < -2.0 {
        currentSegments = currentSegments.filter { segment in
          let trimmedText = segment.text.trimmingCharacters(in: .whitespaces)
          if !trimmedText.isEmpty {
            Log.model.warning("Filtering low-confidence segment (avgLogProb=\(String(format: "%.2f", result.avgLogProb)), temp=\(result.temperature)): '\(trimmedText.prefix(50))'")
          }
          return false
        }
      }

      // Add word timestamps if requested (batched for efficiency)
      if timestamps == .word {
        // Use batched word timestamp extraction (single forward pass for all segments)
        lastSpeechTimestamp = addWordTimestamps(
          segments: &currentSegments,
          model: model,
          tokenizer: tokenizer,
          mel: batchedMel,
          numFrames: segmentSize,
          language: languageToUse,
          task: task,
          timeOffset: timeOffset,
          lastSpeechTimestamp: lastSpeechTimestamp
        )

        // Content-aware seek advancement based on last word
        if !singleTimestampEnding {
          if let lastWordEnd = getLastWordEnd(currentSegments), lastWordEnd > timeOffset {
            seek = Int(lastWordEnd * Float(framesPerSecond))
          }
        }
        // Hallucination detection (inline, matching Python)
        if let threshold = hallucinationSilenceThreshold {
          // Python lines 756-767: Check remaining duration after last word
          // If remaining silence > threshold, keep the last_word_end seek
          // Otherwise, reset to previous_seek + segment_size
          if !singleTimestampEnding {
            if let lastWordEnd = getLastWordEnd(currentSegments), lastWordEnd > timeOffset {
              let remainingDuration = windowEndTime - lastWordEnd
              if remainingDuration > threshold {
                seek = Int(lastWordEnd * Float(framesPerSecond))
              } else {
                seek = previousSeek + segmentSize
              }
            }
          }

          // Check first segment for leading silence hallucination
          if let firstSegment = currentSegments.first(where: { $0.words != nil && !$0.words!.isEmpty }) {
            let wordTimings = firstSegment.words!.map {
              WordTiming(word: $0.word, tokens: [], start: Float($0.start), end: Float($0.end), probability: $0.probability)
            }
            if isSegmentAnomaly(wordTimings) {
              let gap = Float(firstSegment.start) - timeOffset
              if gap > threshold {
                seek = previousSeek + Int(gap * Float(framesPerSecond))
                continue
              }
            }
          }

          // Check for hallucinations surrounded by silence
          var halLastEnd = lastSpeechTimestamp
          for si in 0 ..< currentSegments.count {
            let segment = currentSegments[si]
            guard let words = segment.words, !words.isEmpty else { continue }

            let wordTimings = words.map {
              WordTiming(word: $0.word, tokens: [], start: Float($0.start), end: Float($0.end), probability: $0.probability)
            }

            if isSegmentAnomaly(wordTimings) {
              let segmentStart = Float(segment.start)
              let segmentEnd = Float(segment.end)

              // Find next segment with words
              let nextSeg = currentSegments[(si + 1)...].first { $0.words != nil && !$0.words!.isEmpty }
              let halNextStart: Float = if let next = nextSeg, let firstWord = next.words?.first {
                Float(firstWord.start)
              } else {
                timeOffset + segmentDuration
              }

              let silenceBefore = (segmentStart - halLastEnd > threshold) ||
                (segmentStart < threshold) ||
                (segmentStart - timeOffset < 2.0)

              let nextWordTimings: [WordTiming]? = nextSeg?.words?.map {
                WordTiming(word: $0.word, tokens: [], start: Float($0.start), end: Float($0.end), probability: $0.probability)
              }
              // Python: window_end_time - segment["end"] < 2.0
              let silenceAfter = (halNextStart - segmentEnd > threshold) ||
                isSegmentAnomaly(nextWordTimings) ||
                (windowEndTime - segmentEnd < 2.0)

              if silenceBefore, silenceAfter {
                seek = Int(max(timeOffset + 1, segmentStart) * Float(framesPerSecond))
                if contentDuration - segmentEnd < threshold {
                  seek = contentFrames
                }
                currentSegments.removeSubrange(si...)
                break
              }
            }
            halLastEnd = Float(segment.end)
          }
        }

        // Update last speech timestamp (outside hallucination check, inside word_timestamps)
        // Python lines 822-824
        if let lastWordEnd = getLastWordEnd(currentSegments) {
          lastSpeechTimestamp = lastWordEnd
        }
      }

      // Filter out problematic segments (inspired by WhisperKit's threshold-based approach):
      // 1. Zero-duration segments (start == end)
      // 2. Empty text after trimming whitespace
      // 3. Punctuation-only segments (likely artifacts)
      // 4. High no-speech probability segments (likely silence/hallucination)
      // 5. For word timestamps: segments where word alignment failed but text exists
      let punctuationOnly = CharacterSet.punctuationCharacters.union(.whitespaces)
      currentSegments = currentSegments.filter { segment in
        // Keep segments with valid duration and non-empty meaningful text
        let trimmedText = segment.text.trimmingCharacters(in: .whitespaces)
        let hasMeaningfulText = !trimmedText.isEmpty &&
          !trimmedText.unicodeScalars.allSatisfy { punctuationOnly.contains($0) }

        // Filter high no-speech probability segments (WhisperKit uses 0.6 default threshold)
        // These are likely hallucinations from silence/padding
        if segment.noSpeechProb > 0.9 {
          Log.model.warning("Filtering high no-speech segment (\(segment.noSpeechProb)): '\(trimmedText.prefix(50))'")
          return false
        }

        // For word timestamps: apply multiple hallucination checks
        if timestamps == .word {
          let hasWords = segment.words != nil && !segment.words!.isEmpty
          // If we have text but no words, be suspicious - only keep if it's very short text
          if hasMeaningfulText, !hasWords, trimmedText.count > 10 {
            Log.model.warning("Filtering potential hallucination (no word alignment): '\(trimmedText.prefix(50))'")
            return false
          }

          // Check for anomalous word patterns (very short, very long, or low probability)
          // This catches hallucinations like "iğ", "B gensham", etc.
          if hasWords {
            let wordTimings = segment.words!.map {
              WordTiming(word: $0.word, tokens: [], start: Float($0.start), end: Float($0.end), probability: $0.probability)
            }
            if isSegmentAnomaly(wordTimings) {
              Log.model.warning("Filtering anomalous segment (suspicious word patterns): '\(trimmedText.prefix(50))'")
              return false
            }
          }
        }

        return segment.start != segment.end && hasMeaningfulText
      }

      // Add segments and tokens
      allSegments.append(contentsOf: currentSegments)
      for segment in currentSegments {
        allTokens.append(contentsOf: segment.tokens)
      }

      // Reset prompt if temperature was high (use actual decode temperature, not parameter)
      // Python: if not condition_on_previous_text or result.temperature > 0.5
      if !conditionOnPreviousText || result.temperature > 0.5 {
        promptResetSince = allTokens.count
      }
    }

    let audioDuration = Double(audio.shape[0]) / Double(sampleRate)
    // Decode all tokens together to preserve natural spacing (matching WhisperKit/Python behavior)
    // Filter out special tokens (timestamps, etc.) before decoding
    // This avoids double spaces that occur when joining segment texts with a separator
    let textTokens = allTokens.filter { $0 < tokenizer.eot }
    let fullText = tokenizer.decode(textTokens).trimmingCharacters(in: .whitespaces)

    let transcribeEndTime = CFAbsoluteTimeGetCurrent()
    let totalTime = transcribeEndTime - transcribeStartTime

    Log.model.info("Transcription complete: \(String(format: "%.2f", totalTime))s for \(String(format: "%.2f", audioDuration))s audio (RTF: \(String(format: "%.2f", totalTime / audioDuration)))")

    return TranscriptionResult(
      text: fullText,
      language: detectedLanguage ?? language ?? "en",
      segments: allSegments,
      processingTime: totalTime,
      duration: audioDuration
    )
  }

  // MARK: - Batched Transcription

  /// Transcribe audio using VAD-guided hybrid batching for offline workloads.
  ///
  /// Runs an energy-based VAD pass first to locate natural silence boundaries,
  /// then routes segments to two parallel paths:
  /// - **Short segments (≤ 29 s):** packed into batches of `batchSize` and processed
  ///   with a single batched encoder + parallel decode loop, amortising GPU weight-load
  ///   cost across the batch.
  /// - **Long segments (> 29 s):** processed sequentially in 30-second windows via the
  ///   standard single-sequence decoder.
  ///
  /// Because `conditionOnPreviousText` is always `false` in the batched path (no inter-segment
  /// autoregressive dependency), the shared SOT prefix keeps all KV caches in sync throughout
  /// the parallel decode loop.
  ///
  /// - Parameters:
  ///   - audio: Audio waveform (T,) at 16 kHz
  ///   - language: Optional language code; `nil` triggers auto-detection from the first 30 s
  ///   - task: Transcription or translation
  ///   - temperature: Sampling temperature (0 = greedy)
  ///   - timestamps: Timestamp granularity for each segment
  ///   - noSpeechThreshold: Segments above this probability are discarded
  ///   - logprobThreshold: Low-confidence gate (average log-probability per token)
  ///   - batchSize: Number of short segments to encode and decode in parallel
  ///   - minSilenceDuration: Minimum silence gap (seconds) to split at
  ///   - silenceThresholdDB: Energy threshold (dBFS) below which a frame is silent
  /// - Returns: Assembled `TranscriptionResult` with chronologically ordered segments
  func transcribeBatched(
    audio: MLXArray,
    language: String?,
    task: TranscriptionTask,
    temperature: Float,
    timestamps: TimestampGranularity,
    noSpeechThreshold: Float? = 0.6,
    logprobThreshold: Float? = -1.0,
    hallucinationSilenceThreshold: Double = 2.0,
    batchSize: Int = 4
  ) async throws -> TranscriptionResult {
    let wallStart = CFAbsoluteTimeGetCurrent()
    let sampleRate = WhisperAudio.sampleRate
    let hopLength = WhisperAudio.hopLength
    let nMels = model.dims.n_mels
    let nFrames = WhisperAudio.nFrames

    // Materialise audio on the CPU for Silero VAD
    eval(audio)
    let audioFloats = audio.asArray(Float.self)
    let totalSamples = audioFloats.count
    let audioDuration = Double(totalSamples) / Double(sampleRate)

    // Silero VAD: find speech segments at natural silence boundaries.
    // maxSpeechDuration: 29 s keeps every segment within Whisper's 30-second context window,
    // so all output from segmentSpeech() feeds the short (batched) path.
    let vadMgr = try await getVadManager()
    // maxSpeechDuration: 29 s to stay within Whisper's 30-second context window.
    // minSilenceDuration: 0.5 s — tighter than FluidAudio's 0.75 s default to preserve
    // short pauses as segment boundaries rather than merging them into longer segments.
    let vadSegConfig = VadSegmentationConfig(
      minSpeechDuration: 0.15,
      minSilenceDuration: 0.5,
      maxSpeechDuration: 29.0,
      speechPadding: 0.1,
      silenceThresholdForSplit: 0.3
    )
    let vadSegments = try await vadMgr.segmentSpeech(audioFloats, config: vadSegConfig)

    // Merge adjacent segments to fill Whisper's 30-second context window more efficiently.
    // Silero produces a median segment of ~6 s on real meeting audio; merging reduces
    // encoder passes by packing more speech per batch slot.
    let mergedSegments = mergeAdjacentSegments(vadSegments)

    // With maxSpeechDuration = 29 s, Silero guarantees all segments ≤ 29 s before merging.
    // After merging we still respect the 29 s cap, so longSegments is expected to be empty.
    // The fallback to the sequential decoder is retained as a safety net.
    let shortSegments = mergedSegments.filter { $0.duration <= 29.0 }
    let longSegments  = mergedSegments.filter { $0.duration >  29.0 }
    Log.model.info(
      "Silero VAD: \(vadSegments.count) raw → \(mergedSegments.count) merged segments from \(String(format: "%.1f", audioDuration))s audio (short: \(shortSegments.count), long: \(longSegments.count))"
    )

    // Compute mel spectrogram for the full audio once (pad with 30 s silence to handle boundaries)
    let paddedAudio = MLX.concatenated([audio, MLXArray.zeros([WhisperAudio.nSamples])], axis: 0)
    let fullMel = whisperLogMelSpectrogram(audio: paddedAudio, nMels: nMels)
    eval(fullMel)
    let totalMelFrames = fullMel.shape[0]

    // Language detection from first 30 s if needed
    var detectedLanguage: String? = nil
    if language == nil {
      let firstMel = padOrTrimMel(fullMel[0 ..< min(nFrames, totalMelFrames)], length: nFrames)
      let batchedMel = firstMel.expandedDimensions(axis: 0).asType(.float16)
      let (lang, prob) = detectLanguageFromMel(batchedMel)
      detectedLanguage = lang
      Log.model.info("Detected language: \(lang) (probability: \(String(format: "%.2f", prob)))")
    }
    let languageToUse = language ?? detectedLanguage ?? "en"

    // Base options — prompt is updated per-batch for inter-batch conditioning.
    //
    // timestamps: .none — VAD provides segment boundaries; Whisper's internal
    //   timestamp tokens are redundant in the batched path and add per-step
    //   computation (timestamp rules, forcing heuristic).  The alignment pass in
    //   findAlignment constructs its own noTimestamps sequence independently.
    //
    // noRepeatNgramSize: 3 — mirrors faster-whisper's default; prevents the model
    //   from repeating any 3-gram, which eliminates the boundary hallucinations
    //   ("Okay. Okay. Okay.") that survive the compression-ratio gate.
    let baseOptions = DecodingOptions(
      task: task,
      language: languageToUse,
      temperature: temperature,
      maxTokens: 448,
      timestamps: .none,
      prompt: [],
      noRepeatNgramSize: 3
    )

    // Collect (globalStartTime, TranscriptionSegment) for chronological assembly
    var collected: [(globalStart: TimeInterval, segment: TranscriptionSegment)] = []

    // Helper: convert sample index to mel frame index
    func sampleToFrame(_ sample: Int) -> Int { min(sample / hopLength, totalMelFrames) }

    // Helper: accept-or-discard a DecodingResult based on quality gates.
    // Mirrors Python's whisper: no-speech probability, avg log-prob, AND
    // compression ratio (> 2.4 means the text is highly repetitive / hallucinated).
    func shouldAccept(_ result: DecodingResult) -> Bool {
      if result.compressionRatio > 2.4 { return false }
      guard let threshold = noSpeechThreshold else { return true }
      if result.noSpeechProb > threshold {
        if let lp = logprobThreshold, result.avgLogProb > lp { return true }
        return false
      }
      return true
    }

    // MARK: Short segments — batched path

    let batchDecoder = BatchedGreedyDecoder(model: model, tokenizer: tokenizer)

    // Inter-batch conditioning: carry the last batch's prompt tokens forward.
    // All segments in a batch share the same prompt (from the previous batch's
    // last segment) — an approximation that eliminates boundary hallucinations
    // while preserving the parallelism benefit.
    var batchPrompt: [Int] = []

    var idx = 0
    while idx < shortSegments.count {
      let end = min(idx + batchSize, shortSegments.count)
      let batch = Array(shortSegments[idx ..< end])

      // Slice mel frames for each segment; track valid (pre-padding) frame counts for DTW.
      var melSlices: [MLXArray] = []
      var validFrameCounts: [Int] = []
      for seg in batch {
        let startFrame = sampleToFrame(seg.startSample(sampleRate: sampleRate))
        let endFrame = min(sampleToFrame(seg.endSample(sampleRate: sampleRate)), totalMelFrames)
        let validFrames = max(1, min(endFrame - startFrame, nFrames))
        let slice = endFrame > startFrame ? fullMel[startFrame ..< endFrame] : fullMel[0 ..< 1]
        melSlices.append(padOrTrimMel(slice, length: nFrames).asType(.float16))
        validFrameCounts.append(validFrames)
      }

      // Stack into [B, nFrames, nMels] and encode in one GPU pass
      let melStack = MLX.stacked(melSlices, axis: 0)
      let audioFeaturesStack = model.compiledEncode(melStack)
      eval(audioFeaturesStack)

      let batchOptions = batchPrompt.isEmpty
        ? baseOptions
        : DecodingOptions(
            task: baseOptions.task,
            language: baseOptions.language,
            temperature: baseOptions.temperature,
            maxTokens: baseOptions.maxTokens,
            timestamps: baseOptions.timestamps,
            prompt: batchPrompt
          )

      let results = batchDecoder.decodeBatch(audioFeaturesStack: audioFeaturesStack, options: batchOptions)

      // Update prompt for the next batch: use the last segment's raw tokens
      // (including timestamps), stripping only EOT.  Mirrors Python's behavior
      // where condition_on_previous_text passes result.tokens as the next prompt.
      if let lastResult = results.last {
        batchPrompt = lastResult.tokens.filter { $0 != tokenizer.eot }
      }

      for (batchIdx, (seg, result)) in zip(batch, results).enumerated() {
        guard shouldAccept(result) else { continue }
        let textTokens = result.tokens.filter { $0 < tokenizer.eot }
        let text = tokenizer.decode(textTokens).trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty else { continue }

        var segment = TranscriptionSegment(
          text: text,
          start: seg.startTime,
          end: seg.endTime,
          tokens: result.tokens,
          avgLogProb: result.avgLogProb,
          noSpeechProb: result.noSpeechProb,
          words: nil
        )

        if timestamps == .word {
          // Decoder-only alignment pass: reuse audioFeaturesStack[batchIdx]
          // from the encode step above, skipping the encoder re-run entirely.
          var segs = [segment]
          _ = addWordTimestamps(
            segments: &segs,
            model: model,
            tokenizer: tokenizer,
            mel: melSlices[batchIdx],
            audioFeatures: audioFeaturesStack[batchIdx],
            numFrames: validFrameCounts[batchIdx],
            language: languageToUse,
            task: task,
            timeOffset: Float(seg.startTime),
            lastSpeechTimestamp: Float(seg.startTime)
          )
          segment = segs[0]

          // Hallucination silence gate: if any inter-word gap exceeds the threshold
          // the DTW aligner found a long silence inside what VAD marked as speech —
          // almost certainly a hallucination.  Mirrors faster-whisper's
          // hallucination_silence_threshold logic.
          if let words = segment.words, words.count > 1 {
            let maxGap = zip(words, words.dropFirst())
              .map { Double($1.start) - Double($0.end) }
              .max() ?? 0
            if maxGap > hallucinationSilenceThreshold {
              continue
            }
          }
        }

        collected.append((globalStart: seg.startTime, segment: segment))
      }

      idx = end
    }

    // MARK: Long segments — sequential 30-second window path (rare with Silero)

    let seqDecoder = GreedyDecoder(model: model, tokenizer: tokenizer)

    for seg in longSegments {
      var windowStartFrame = sampleToFrame(seg.startSample(sampleRate: sampleRate))
      let segEndFrame = sampleToFrame(seg.endSample(sampleRate: sampleRate))

      while windowStartFrame < segEndFrame {
        let windowEndFrame = min(windowStartFrame + nFrames, segEndFrame)
        let slice = fullMel[windowStartFrame ..< windowEndFrame]
        // Keep 2D mel (float16) for DTW; add batch dim only for the encoder.
        let windowMel = padOrTrimMel(slice, length: nFrames).asType(.float16)
        let audioFeatures = model.compiledEncode(windowMel.expandedDimensions(axis: 0))
        eval(audioFeatures)

        let result = seqDecoder.decode(audioFeatures: audioFeatures, options: baseOptions)

        guard shouldAccept(result) else {
          windowStartFrame = windowEndFrame
          continue
        }

        let windowStartTime = TimeInterval(windowStartFrame * hopLength) / TimeInterval(sampleRate)
        let windowEndTime = TimeInterval(windowEndFrame * hopLength) / TimeInterval(sampleRate)
        let textTokens = result.tokens.filter { $0 < tokenizer.eot }
        let text = tokenizer.decode(textTokens).trimmingCharacters(in: .whitespaces)

        if !text.isEmpty {
          var segment = TranscriptionSegment(
            text: text,
            start: windowStartTime,
            end: windowEndTime,
            tokens: result.tokens,
            avgLogProb: result.avgLogProb,
            noSpeechProb: result.noSpeechProb,
            words: nil
          )

          if timestamps == .word {
            var segs = [segment]
            _ = addWordTimestamps(
              segments: &segs,
              model: model,
              tokenizer: tokenizer,
              mel: windowMel,
              numFrames: windowEndFrame - windowStartFrame,
              language: languageToUse,
              task: task,
              timeOffset: Float(windowStartTime),
              lastSpeechTimestamp: Float(windowStartTime)
            )
            segment = segs[0]
          }

          collected.append((globalStart: windowStartTime, segment: segment))
        }

        windowStartFrame = windowEndFrame
      }
    }

    // Assemble in chronological order
    collected.sort { $0.globalStart < $1.globalStart }
    let orderedSegments = collected.map { $0.segment }

    // Decode full text from all token sequences together for natural spacing
    let allTextTokens = orderedSegments.flatMap { $0.tokens }.filter { $0 < tokenizer.eot }
    let fullText = tokenizer.decode(allTextTokens).trimmingCharacters(in: .whitespaces)

    let processingTime = CFAbsoluteTimeGetCurrent() - wallStart
    Log.model.info(
      "Batched transcription: \(String(format: "%.2f", processingTime))s for \(String(format: "%.2f", audioDuration))s audio (RTF: \(String(format: "%.4f", processingTime / audioDuration)))"
    )

    return TranscriptionResult(
      text: fullText,
      language: detectedLanguage ?? language ?? "en",
      segments: orderedSegments,
      processingTime: processingTime,
      duration: audioDuration
    )
  }

  /// Pad or trim mel spectrogram to specified length
  private func padOrTrimMel(_ mel: MLXArray, length: Int) -> MLXArray {
    let currentLength = mel.shape[0]
    if currentLength == length {
      return mel
    } else if currentLength > length {
      return mel[0 ..< length]
    } else {
      // Pad with zeros
      let padding = MLXArray.zeros([length - currentLength, mel.shape[1]])
      return MLX.concatenated([mel, padding], axis: 0)
    }
  }

  // MARK: - Language Detection

  /// Detect the language of audio
  ///
  /// - Parameter audio: Audio waveform (T,) in 16 kHz
  /// - Returns: Tuple of (language_code, probability)
  func detectLanguage(audio: MLXArray) -> (String, Float) {
    // Pad or trim to 30 seconds
    let paddedAudio = padOrTrim(audio)
    eval(paddedAudio)

    // Compute mel spectrogram - returns (n_frames, n_mels)
    let mel = whisperLogMelSpectrogram(audio: paddedAudio, nMels: model.dims.n_mels)
    // Ensure exactly 3000 frames to match encoder expectations
    let melTrimmed = padOrTrimMel(mel, length: WhisperAudio.nFrames)
    // Cast to float16 to match Python's behavior
    let batchedMel = melTrimmed.expandedDimensions(axis: 0).asType(.float16)

    return detectLanguageFromMel(batchedMel)
  }

  /// Detect language from mel spectrogram
  ///
  /// - Parameter mel: Mel spectrogram (batch=1 or unbatched)
  /// - Returns: Tuple of (language_code, probability)
  private func detectLanguageFromMel(_ mel: MLXArray) -> (String, Float) {
    // Add batch dimension if needed
    var melBatched = mel
    if mel.ndim == 2 {
      melBatched = mel.expandedDimensions(axis: 0)
    }

    // Encode audio
    let audioFeatures = model.compiledEncode(melBatched)

    // Create SOT token
    let sotToken = MLXArray([Int32(tokenizer.sot)]).expandedDimensions(axis: 0)

    // Get logits for first token after SOT
    let (logits, _, _) = model.decode(sotToken, audioFeatures: audioFeatures)

    // Extract language token logits
    // Language tokens start at sot + 1 and span numLanguages tokens
    // (base: 99, large-v3-turbo: 100)
    let languageTokenStart = tokenizer.sot + 1
    let languageTokenEnd = tokenizer.sot + 1 + tokenizer.numLanguages
    let languageLogits = logits[0, 0, languageTokenStart ..< languageTokenEnd]

    // Find language with highest probability
    let probs = MLX.softmax(languageLogits, axis: -1)
    let maxIdx = MLX.argMax(probs).item(Int32.self)
    let maxProb = probs[Int(maxIdx)].item(Float.self)

    // Map index to language code using tokenizer's single source of truth
    let languageIdx = Int(maxIdx)
    let languageCode = tokenizer.languageCode(forIndex: languageIdx) ?? "en"

    return (languageCode, maxProb)
  }

  // MARK: - Audio Segmentation

  /// Segment long audio into 30-second chunks
  ///
  /// - Parameter audio: Audio waveform (T,)
  /// - Returns: Array of audio segments
  private func segmentAudio(_ audio: MLXArray) -> [MLXArray] {
    let audioLength = audio.shape[0]
    let chunkSamples = WhisperAudio.nSamples // 480,000 samples (30s at 16kHz)

    // If audio is shorter than or equal to 30 seconds, return as single segment
    if audioLength <= chunkSamples {
      return [audio]
    }

    // Split into 30-second chunks
    var segments: [MLXArray] = []
    var start = 0

    while start < audioLength {
      let end = min(start + chunkSamples, audioLength)
      let segment = audio[start ..< end]
      segments.append(segment)
      start = end
    }

    return segments
  }
}
