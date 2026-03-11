// Copyright © Anthony DePasquale

import Foundation
import MLX
import MLXNN
@_exported import MLXAudio

// MARK: - Wav2Vec2AlignerMLX

/// MLX-native wav2vec2 forced aligner
///
/// Combines the MLX wav2vec2 model, tokenizer, and CTC alignment
/// to provide word-level timestamps for known text. This is a drop-in
/// replacement for the CoreML version that runs entirely on MLX.
public final class Wav2Vec2AlignerMLX {
  // MARK: - Properties

  /// The MLX wav2vec2 model
  private let model: Wav2Vec2ForCTC

  /// The character-level tokenizer
  private let tokenizer: Wav2Vec2Tokenizer

  /// The CTC forced alignment algorithm
  private let ctcAligner: CTCForcedAligner

  /// Model configuration
  public let config: Wav2Vec2Config

  /// Frame duration in seconds (20ms for wav2vec2)
  public let frameDuration: Float = 0.02

  /// Maximum audio length in seconds (30 seconds for fixed input models)
  public let maxAudioLength: Float = 600.0  // 10 minutes for long-form alignment

  /// The CTC blank token ID (typically 0 for wav2vec2)
  public var blankId: Int { tokenizer.blankId }

  /// Vocabulary size
  public var vocabSize: Int { config.vocabSize }

  // MARK: - Initialization

  /// Create a new wav2vec2 aligner with pre-loaded components
  ///
  /// - Parameters:
  ///   - model: Pre-loaded MLX wav2vec2 model
  ///   - tokenizer: Pre-loaded tokenizer
  ///   - ctcAligner: CTC forced aligner (defaults to new instance with blankId from tokenizer)
  public init(
    model: Wav2Vec2ForCTC,
    tokenizer: Wav2Vec2Tokenizer,
    ctcAligner: CTCForcedAligner? = nil
  ) {
    self.model = model
    self.tokenizer = tokenizer
    self.config = model.config
    self.ctcAligner = ctcAligner ?? CTCForcedAligner(blankId: tokenizer.blankId)
  }

  // MARK: - Public Methods

  /// Align known text to audio, returning word-level timestamps
  ///
  /// This is the main entry point for forced alignment. It:
  /// 1. Chunks audio into segments (max 30s each)
  /// 2. Runs wav2vec2 to get frame-level log probabilities for each chunk
  /// 3. Tokenizes the text to character IDs
  /// 4. Runs CTC forced alignment to get character-level timestamps
  /// 5. Groups characters into words and adjusts timestamps
  ///
  /// - Parameters:
  ///   - audio: 16kHz mono audio samples
  ///   - text: Known transcription text (should match audio content)
  /// - Returns: Array of word-level alignments with timestamps
  /// - Throws: Wav2Vec2Error if alignment fails
  public func align(audio: [Float], text: String) throws -> [WordAlignment] {
    // Validate inputs
    guard !audio.isEmpty else {
      throw Wav2Vec2Error.invalidAudio("Audio samples array is empty")
    }

    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmedText.isEmpty else {
      throw Wav2Vec2Error.invalidText("Text is empty after trimming whitespace")
    }

    let audioDuration = Float(audio.count) / 16000

    // If audio fits in one chunk, process directly
    if audioDuration <= maxAudioLength {
      return try alignSingleChunk(audio: audio, text: trimmedText, timeOffset: 0)
    }

    // Otherwise, chunk and process
    print("  Chunking \(fmt2(audioDuration))s audio into ~30s segments...")

    let samplesPerChunk = Int(maxAudioLength * 16000)
    var allWordAlignments: [WordAlignment] = []
    var chunkIndex = 0

    for chunkStart in stride(from: 0, to: audio.count, by: samplesPerChunk) {
      let chunkEnd = min(chunkStart + samplesPerChunk, audio.count)
      let chunkAudio = Array(audio[chunkStart..<chunkEnd])
      let timeOffset = Float(chunkStart) / 16000

      let chunkDuration = Float(chunkAudio.count) / 16000
      chunkIndex += 1
      print("  Chunk \(chunkIndex): [\(fmt2(timeOffset))–\(fmt2(timeOffset + chunkDuration))s")

      // Align this chunk
      let chunkWords = try alignSingleChunk(audio: chunkAudio, text: trimmedText, timeOffset: timeOffset)

      // Add to results with time offset applied
      allWordAlignments.append(contentsOf: chunkWords)
    }

    return allWordAlignments
  }

  /// Align a single chunk of audio (must be <= maxAudioLength)
  private func alignSingleChunk(audio: [Float], text: String, timeOffset: Float) throws -> [WordAlignment] {
    // 1. Get frame-level probabilities from wav2vec2
    let logProbs = try getFrameLogProbs(audio)

    // 2. Tokenize text to characters
    let tokens = tokenizer.encode(text)

    // 3. Build ID-to-token mapping for alignment results
    var idToToken: [Int: String] = [:]
    for id in 0..<vocabSize {
      if let char = tokenizer.character(for: id) {
        idToToken[id] = String(char)
      }
    }

    // 4. Run CTC forced alignment
    let charAlignments = try ctcAligner.align(
      logProbs: logProbs,
      tokens: tokens,
      idToToken: idToToken
    )

    // Check if alignment succeeded
    if charAlignments.isEmpty && !tokens.isEmpty {
      throw Wav2Vec2Error.alignmentFailure("CTC alignment produced no results")
    }

    // 5. Group characters into words with time offset
    let words = groupIntoWords(
      charAlignments: charAlignments,
      text: text
    )

    // Apply time offset
    return words.map { word in
      WordAlignment(
        word: word.word,
        startTime: word.startTime + timeOffset,
        endTime: word.endTime + timeOffset
      )
    }
  }

  /// Get frame-level log probabilities without performing alignment
  ///
  /// Useful for debugging or custom alignment algorithms.
  ///
  /// - Parameter audio: 16kHz mono audio samples
  /// - Returns: 2D array [frames][vocab_size] of log probabilities
  /// - Throws: Wav2Vec2Error if inference fails
  public func getFrameLogProbs(_ audio: [Float]) throws -> [[Float]] {
    guard !audio.isEmpty else {
      throw Wav2Vec2Error.invalidAudio("Audio samples array is empty")
    }

    // Convert audio to MLXArray and add batch dimension
    let audioTensor = MLXArray(audio).reshaped(1, -1)

    // Run model forward pass
    // Output shape: [batch, frames, vocabSize]
    let output = model(audioTensor)

    // Convert MLXArray output to [[Float]] for CTC
    return mlxArrayToLogProbs(output)
  }

  /// Tokenize text to character IDs without alignment
  ///
  /// - Parameter text: Input text
  /// - Returns: Array of token IDs
  public func tokenize(_ text: String) -> [Int] {
    tokenizer.encode(text)
  }

  /// Decode token IDs back to text
  ///
  /// - Parameter ids: Token IDs
  /// - Returns: Decoded text string
  public func decode(_ ids: [Int]) -> String {
    tokenizer.decode(ids)
  }

  // MARK: - Private Methods

  /// Convert MLXArray output from model to [[Float]] for CTC alignment
  ///
  /// The model output is [batch, frames, vocabSize] - we need to extract
  /// batch 0 and convert to [[Float]] format expected by CTCForcedAligner.
  ///
  /// - Parameter mlxArray: Output tensor of shape [batch, frames, vocabSize]
  /// - Returns: 2D array [frames][vocabSize] of log probabilities
  private func mlxArrayToLogProbs(_ mlxArray: MLXArray) -> [[Float]] {
    // mlxArray shape is [batch, frames, vocabSize]
    // Extract first batch: [frames, vocabSize]
    let batch0 = mlxArray[0]

    // Convert to Swift array
    // Use .asArray() to get the underlying data
    let flatArray = batch0.asArray(Float.self)

    // Get dimensions
    let dims = batch0.shape
    let frames = dims[0]
    let vocabSize = dims[1]

    // Reshape flat array to 2D
    var result: [[Float]] = []
    result.reserveCapacity(frames)

    for frame in 0..<frames {
      let start = frame * vocabSize
      let end = start + vocabSize
      let row = Array(flatArray[start..<end])
      result.append(row)
    }

    return result
  }

  /// Format time value with 2 decimal places
  private func fmt2(_ value: Float) -> String {
    String(format: "%.2f", value)
  }

  /// Group character-level alignments into word-level alignments
  ///
  /// This merges consecutive character alignments into words based on
  /// whitespace boundaries in the original text.
  ///
  /// - Parameters:
  ///   - charAlignments: Character-level alignments from CTC
  ///   - text: Original text (used for word boundaries)
  /// - Returns: Word-level alignments
  private func groupIntoWords(
    charAlignments: [AlignedToken],
    text: String
  ) -> [WordAlignment] {
    // Split text into words, preserving the order
    let words = text.components(separatedBy: .whitespaces)
      .filter { !$0.isEmpty }

    guard !words.isEmpty, !charAlignments.isEmpty else {
      return []
    }

    var wordAlignments: [WordAlignment] = []
    var charIndex = 0

    for word in words {
      let wordLength = word.count

      // Check if we have enough character alignments
      guard charIndex + wordLength <= charAlignments.count else {
        // Not enough alignments - skip remaining words
        break
      }

      // Get the span of character alignments for this word
      let startCharAlignment = charAlignments[charIndex]
      let endCharAlignment = charAlignments[charIndex + wordLength - 1]

      // Create word alignment
      let wordAlignment = WordAlignment(
        word: word,
        startTime: startCharAlignment.startTime,
        endTime: endCharAlignment.endTime
      )
      wordAlignments.append(wordAlignment)

      // Move past this word's characters
      charIndex += wordLength

      // Skip space character if present in alignments
      // Some tokenizers include spaces as tokens
      if charIndex < charAlignments.count,
         charAlignments[charIndex].token == " " ||
           charAlignments[charIndex].token == "" ||
           charAlignments[charIndex].token == "|"
      {
        charIndex += 1
      }
    }

    return wordAlignments
  }
}

// MARK: - Static Factory Methods

extension Wav2Vec2AlignerMLX {

  /// Load aligner from Hugging Face Hub
  ///
  /// - Parameters:
  ///   - modelId: Hugging Face model ID (e.g., "facebook/wav2vec2-base-960h")
  ///   - progressHandler: Optional callback for download progress
  /// - Returns: Initialized Wav2Vec2AlignerMLX
  /// - Throws: STTError if loading fails
  public static func fromPretrained(
    modelId: String,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
  ) async throws -> Wav2Vec2AlignerMLX {

    // Load model using Wav2Vec2ForCTC.fromPretrained
    let model = try await Wav2Vec2ForCTC.fromPretrained(
      modelId: modelId,
      progressHandler: progressHandler
    )

    // Load tokenizer from downloaded vocab.json
    // The model directory is in the Hub cache
    let modelDirectory = try await HubConfiguration.shared.snapshot(
      from: modelId,
      matching: ["vocab.json"],
      progressHandler: progressHandler
    )

    let vocabPath = modelDirectory.appending(path: "vocab.json")
    let tokenizer = try Wav2Vec2Tokenizer(vocabPath: vocabPath)

    // Create aligner
    return Wav2Vec2AlignerMLX(
      model: model,
      tokenizer: tokenizer
    )
  }
}
