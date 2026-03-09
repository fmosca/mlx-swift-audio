// Copyright © 2022 OpenAI (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/openai/whisper
// License: licenses/whisper.txt

import AVFoundation
import Foundation
import MLX

/// Whisper STT engine - multilingual speech recognition
///
/// Supports transcription and translation with automatic language detection
@Observable
@MainActor
public final class WhisperEngine: STTEngine {
  // MARK: - STTEngine Protocol Properties

  public let provider: STTProvider = .whisper
  public private(set) var isLoaded: Bool = false
  public private(set) var isTranscribing: Bool = false
  public private(set) var transcriptionTime: TimeInterval = 0

  // MARK: - Whisper-Specific Properties

  /// Model size (nil when loaded from a custom repo ID)
  public let modelSize: WhisperModelSize

  /// Quantization level
  public let quantization: WhisperQuantization

  // MARK: - Private Properties

  /// Custom HuggingFace repo ID, overrides the repo derived from modelSize + quantization.
  @ObservationIgnored private let customRepoId: String?

  @ObservationIgnored private var whisperSTT: WhisperSTT?

  // MARK: - Initialization

  public init(modelSize: WhisperModelSize = .base, quantization: WhisperQuantization = .q4) {
    self.modelSize = modelSize
    self.quantization = quantization
    self.customRepoId = nil
    Log.tts.debug("WhisperEngine initialized with model: \(modelSize.rawValue), quantization: \(quantization.rawValue)")
  }

  /// Initialize with an explicit HuggingFace repository ID.
  ///
  /// Use this when the repository does not follow the standard
  /// `mlx-community/whisper-<size>-<quantization>` naming (e.g. `mlx-community/whisper-large-v3-turbo`).
  ///
  /// - Parameters:
  ///   - repoId: Full HuggingFace repository identifier
  ///   - quantization: Quantization level, used only for the MLX quantize pass when weights
  ///     already contain `.scales` keys. Default is fp16 (no quantization pass).
  public init(repoId: String, quantization: WhisperQuantization = .fp16) {
    self.modelSize = .largeTurbo
    self.quantization = quantization
    self.customRepoId = repoId
    Log.tts.debug("WhisperEngine initialized with custom repo: \(repoId), quantization: \(quantization.rawValue)")
  }

  // MARK: - STTEngine Protocol Methods

  public func load(progressHandler: (@Sendable (Progress) -> Void)?) async throws {
    guard !isLoaded else {
      Log.model.info("Whisper model already loaded")
      return
    }

    let modelSize = modelSize
    let quantization = quantization
    Log.model.info("Loading Whisper \(modelSize.rawValue) (\(quantization.rawValue)) model...")

    if let repoId = customRepoId {
      whisperSTT = try await WhisperSTT.load(
        repoId: repoId,
        quantization: quantization,
        progressHandler: progressHandler ?? { _ in }
      )
    } else {
      whisperSTT = try await WhisperSTT.load(
        modelSize: modelSize,
        quantization: quantization,
        progressHandler: progressHandler ?? { _ in }
      )
    }

    isLoaded = true
    Log.model.info("Whisper model loaded successfully")
  }

  public func stop() async {
    // TODO: Implement cancellation mechanism
    Log.model.info("Stop requested (not fully implemented yet)")
  }

  public func unload() async {
    whisperSTT = nil
    isLoaded = false
    Log.model.info("Whisper model unloaded")
  }

  public func cleanup() async throws {
    await unload()
    Log.model.info("Whisper cleanup complete")
  }

  // MARK: - Transcription Methods

  /// Transcribe audio in original language from a URL
  ///
  /// - Parameters:
  ///   - url: URL to audio file (supports WAV, MP3, M4A, etc.)
  ///   - language: Language of the audio (nil = auto-detect)
  ///   - temperature: Sampling temperature (0.0 = greedy, higher = more random)
  ///   - timestamps: Timestamp granularity
  ///   - hallucinationSilenceThreshold: When word timestamps are enabled, skip silent periods
  ///     longer than this threshold (in seconds) when a possible hallucination is detected.
  ///     Set to nil (default) to disable hallucination filtering.
  /// - Returns: Transcription result
  public nonisolated func transcribe(
    _ url: URL,
    language: Language? = nil,
    temperature: Float = 0.0,
    timestamps: TimestampGranularity = .segment,
    hallucinationSilenceThreshold: Float? = nil
  ) async throws -> TranscriptionResult {
    guard await isLoaded, let whisperSTT = await whisperSTT else {
      throw STTError.modelNotLoaded
    }

    guard await !isTranscribing else {
      throw STTError.invalidArgument("Transcription already in progress")
    }

    await MainActor.run { isTranscribing = true }
    defer { Task { await MainActor.run { isTranscribing = false } } }

    // Load and preprocess audio
    let audio16k = try loadAndPreprocessAudio(from: url)

    // Auto-enable hallucination filtering for word timestamps if not explicitly set
    let effectiveHallucinationThreshold = hallucinationSilenceThreshold ??
      (timestamps == .word ? 2.0 : nil)

    // Transcribe
    let result = await whisperSTT.transcribe(
      audio: audio16k,
      language: language?.code,
      task: .transcribe,
      temperature: temperature,
      timestamps: timestamps,
      hallucinationSilenceThreshold: effectiveHallucinationThreshold
    )

    await MainActor.run { transcriptionTime = result.processingTime }

    return result
  }

  /// Transcribe audio in original language from an MLXArray
  ///
  /// - Parameters:
  ///   - audio: Audio waveform (T,) at 16kHz
  ///   - language: Language of the audio (nil = auto-detect)
  ///   - temperature: Sampling temperature (0.0 = greedy, higher = more random)
  ///   - timestamps: Timestamp granularity
  ///   - hallucinationSilenceThreshold: When word timestamps are enabled, skip silent periods
  ///     longer than this threshold (in seconds) when a possible hallucination is detected.
  ///     Set to nil (default) to disable hallucination filtering.
  /// - Returns: Transcription result
  public nonisolated func transcribe(
    _ audio: sending MLXArray,
    language: Language? = nil,
    temperature: Float = 0.0,
    timestamps: TimestampGranularity = .segment,
    hallucinationSilenceThreshold: Float? = nil
  ) async throws -> TranscriptionResult {
    guard await isLoaded, let whisperSTT = await whisperSTT else {
      throw STTError.modelNotLoaded
    }

    guard await !isTranscribing else {
      throw STTError.invalidArgument("Transcription already in progress")
    }

    await MainActor.run { isTranscribing = true }
    defer { Task { await MainActor.run { isTranscribing = false } } }

    // Auto-enable hallucination filtering for word timestamps if not explicitly set
    let effectiveHallucinationThreshold = hallucinationSilenceThreshold ??
      (timestamps == .word ? 2.0 : nil)

    // Transcribe
    let result = await whisperSTT.transcribe(
      audio: audio,
      language: language?.code,
      task: .transcribe,
      temperature: temperature,
      timestamps: timestamps,
      hallucinationSilenceThreshold: effectiveHallucinationThreshold
    )

    await MainActor.run { transcriptionTime = result.processingTime }

    return result
  }

  /// Transcribe audio using Silero VAD-guided hybrid batching (offline fast path).
  ///
  /// Runs Silero VAD (CoreML, ANE-accelerated) to split at natural silence boundaries,
  /// then processes all segments in parallel batches of `batchSize`, amortising GPU
  /// weight-read cost across the batch.  `conditionOnPreviousText` is always `false`.
  ///
  /// The Silero model is downloaded and cached by FluidAudio on first use
  /// (~2 MB, stored in Application Support/FluidAudio/Models).
  ///
  /// - Parameters:
  ///   - url: URL to audio file
  ///   - language: Language code (nil = auto-detect from first 30 s)
  ///   - temperature: Sampling temperature (0 = greedy)
  ///   - timestamps: Timestamp granularity
  ///   - batchSize: Segments encoded and decoded in parallel (default 4)
  ///   - hallucinationSilenceThreshold: When word timestamps are enabled, segments
  ///     whose word-level alignment contains an inter-word gap longer than this (seconds)
  ///     are discarded as likely hallucinations.  Default 2.0 s; set to `.infinity` to disable.
  /// - Returns: Assembled transcription result
  public nonisolated func transcribeBatched(
    _ url: URL,
    language: Language? = nil,
    temperature: Float = 0.0,
    timestamps: TimestampGranularity = .segment,
    hallucinationSilenceThreshold: Double = 2.0,
    batchSize: Int = 4
  ) async throws -> TranscriptionResult {
    guard await isLoaded, let whisperSTT = await whisperSTT else {
      throw STTError.modelNotLoaded
    }

    guard await !isTranscribing else {
      throw STTError.invalidArgument("Transcription already in progress")
    }

    await MainActor.run { isTranscribing = true }
    defer { Task { await MainActor.run { isTranscribing = false } } }

    let audio16k = try loadAndPreprocessAudio(from: url)

    let result = try await whisperSTT.transcribeBatched(
      audio: audio16k,
      language: language?.code,
      task: .transcribe,
      temperature: temperature,
      timestamps: timestamps,
      hallucinationSilenceThreshold: hallucinationSilenceThreshold,
      batchSize: batchSize
    )

    await MainActor.run { transcriptionTime = result.processingTime }
    return result
  }

  /// Translate audio to English from a URL
  ///
  /// - Parameters:
  ///   - url: URL to audio file (supports WAV, MP3, M4A, etc.)
  ///   - language: Source language hint (nil = auto-detect)
  ///   - timestamps: Timestamp granularity (default: .segment for faster processing)
  /// - Returns: Translation result in English
  public nonisolated func translate(
    _ url: URL,
    language: Language? = nil,
    timestamps: TimestampGranularity = .segment
  ) async throws -> TranscriptionResult {
    guard await isLoaded, let whisperSTT = await whisperSTT else {
      throw STTError.modelNotLoaded
    }

    guard await !isTranscribing else {
      throw STTError.invalidArgument("Transcription already in progress")
    }

    await MainActor.run { isTranscribing = true }
    defer { Task { await MainActor.run { isTranscribing = false } } }

    // Load and preprocess audio
    let audio16k = try loadAndPreprocessAudio(from: url)

    // Translate (always to English)
    // Relax logprobThreshold for translation (model is less confident when translating)
    // Enable hallucination filtering only when word timestamps are requested
    let result = await whisperSTT.transcribe(
      audio: audio16k,
      language: language?.code,
      task: .translate,
      temperature: 0.0,
      timestamps: timestamps,
      logprobThreshold: nil,
      hallucinationSilenceThreshold: timestamps == .word ? 2.0 : nil
    )

    await MainActor.run { transcriptionTime = result.processingTime }

    return result
  }

  /// Translate audio to English from an MLXArray
  ///
  /// - Parameters:
  ///   - audio: Audio waveform (T,) at 16kHz
  ///   - language: Source language hint (nil = auto-detect)
  ///   - timestamps: Timestamp granularity (default: .segment for faster processing)
  /// - Returns: Translation result in English
  public nonisolated func translate(
    _ audio: sending MLXArray,
    language: Language? = nil,
    timestamps: TimestampGranularity = .segment
  ) async throws -> TranscriptionResult {
    guard await isLoaded, let whisperSTT = await whisperSTT else {
      throw STTError.modelNotLoaded
    }

    guard await !isTranscribing else {
      throw STTError.invalidArgument("Transcription already in progress")
    }

    await MainActor.run { isTranscribing = true }
    defer { Task { await MainActor.run { isTranscribing = false } } }

    // Translate (always to English)
    // Relax logprobThreshold for translation (model is less confident when translating)
    // Enable hallucination filtering only when word timestamps are requested
    let result = await whisperSTT.transcribe(
      audio: audio,
      language: language?.code,
      task: .translate,
      temperature: 0.0,
      timestamps: timestamps,
      logprobThreshold: nil,
      hallucinationSilenceThreshold: timestamps == .word ? 2.0 : nil
    )

    await MainActor.run { transcriptionTime = result.processingTime }

    return result
  }

  /// Detect the language of audio from a URL
  ///
  /// - Parameter url: URL to audio file
  /// - Returns: Detected language and confidence score (0-1)
  public nonisolated func detectLanguage(_ url: URL) async throws -> (Language, Float) {
    guard await isLoaded, let _ = await whisperSTT else {
      throw STTError.modelNotLoaded
    }

    // Load and preprocess audio
    let audio16k = try loadAndPreprocessAudio(from: url)

    return try await detectLanguage(audio16k)
  }

  /// Detect the language of audio from an MLXArray
  ///
  /// - Parameter audio: Audio waveform (T,) at 16kHz
  /// - Returns: Detected language and confidence score (0-1)
  public nonisolated func detectLanguage(_ audio: sending MLXArray) async throws -> (Language, Float) {
    guard await isLoaded, let whisperSTT = await whisperSTT else {
      throw STTError.modelNotLoaded
    }

    // Use WhisperSTT's detectLanguage method
    let (languageCode, probability) = await whisperSTT.detectLanguage(audio: audio)

    // Convert language code to Language enum
    guard let language = Language(code: languageCode) else {
      throw STTError.invalidArgument("Detected language '\(languageCode)' not recognized")
    }

    return (language, probability)
  }

  // MARK: - Audio Loading

  /// Load audio file, resample to 16kHz, and return as MLXArray
  ///
  /// - Parameter url: URL to audio file
  /// - Returns: Audio array at 16kHz
  private nonisolated func loadAndPreprocessAudio(from url: URL) throws -> MLXArray {
    let (audioArray, sampleRate) = try loadAudioFile(from: url)

    // Resample to 16kHz if needed
    if sampleRate != WhisperAudio.sampleRate {
      Log.model.debug("Resampling audio from \(sampleRate)Hz to \(WhisperAudio.sampleRate)Hz")
      return AudioResampler.resample(
        audioArray,
        from: sampleRate,
        to: WhisperAudio.sampleRate
      )
    } else {
      return audioArray
    }
  }

  /// Load audio file and convert to MLXArray
  ///
  /// - Parameter url: URL to audio file
  /// - Returns: Tuple of (audio_array, sample_rate)
  private nonisolated func loadAudioFile(from url: URL) throws -> (MLXArray, Int) {
    // Load audio using AVFoundation
    let audioFile = try AVAudioFile(forReading: url)
    let format = audioFile.processingFormat
    let frameCount = AVAudioFrameCount(audioFile.length)

    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
      throw STTError.invalidAudio("Failed to create audio buffer")
    }

    try audioFile.read(into: buffer)

    // Convert to mono Float32 array
    let channelCount = Int(format.channelCount)
    let sampleRate = Int(format.sampleRate)
    let length = Int(buffer.frameLength)

    var audioSamples: [Float] = []
    audioSamples.reserveCapacity(length)

    if let floatData = buffer.floatChannelData {
      if channelCount == 1 {
        // Mono audio
        for i in 0 ..< length {
          audioSamples.append(floatData[0][i])
        }
      } else {
        // Stereo or multi-channel: average all channels to mono
        for i in 0 ..< length {
          var sum: Float = 0.0
          for channel in 0 ..< channelCount {
            sum += floatData[channel][i]
          }
          audioSamples.append(sum / Float(channelCount))
        }
      }
    } else {
      throw STTError.invalidAudio("Audio buffer has no float data")
    }

    return (MLXArray(audioSamples), sampleRate)
  }
}
