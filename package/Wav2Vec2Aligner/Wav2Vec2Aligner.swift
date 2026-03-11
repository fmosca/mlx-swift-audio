// Copyright © Anthony DePasquale

import CoreML
import Foundation

/// Word-level alignment result
public struct WordAlignment {
  /// The word text
  public let word: String

  /// Start time in seconds
  public let startTime: Float

  /// End time in seconds
  public let endTime: Float

  /// Duration of the word in seconds
  public var duration: Float {
    endTime - startTime
  }

  public init(word: String, startTime: Float, endTime: Float) {
    self.word = word
    self.startTime = startTime
    self.endTime = endTime
  }
}

/// High-level wav2vec2 forced aligner
///
/// Combines the CoreML wav2vec2 model, tokenizer, and CTC alignment
/// to provide word-level timestamps for known text.
public final class Wav2Vec2Aligner {
  // MARK: - Properties

  /// The CoreML wav2vec2 model wrapper
  private let model: Wav2Vec2Model

  /// The character-level tokenizer
  private let tokenizer: Wav2Vec2Tokenizer

  /// The CTC forced alignment algorithm
  private let ctcAligner: CTCForcedAligner

  /// Frame duration in seconds (20ms for wav2vec2)
  public let frameDuration: Float = 0.02

  /// Maximum audio length in seconds (30 seconds for fixed input models)
  public let maxAudioLength: Float = 30.0

  // MARK: - Initialization

  /// Create a new wav2vec2 aligner
  ///
  /// - Parameters:
  ///   - modelPath: URL to the .mlpackage or .mlmodelc directory
  ///   - vocabPath: URL to the vocab.json file
  ///   - computeUnits: Compute units for CoreML (default: .all)
  /// - Throws: Wav2Vec2Error if initialization fails
  public init(
    modelPath: URL,
    vocabPath: URL,
    computeUnits: MLComputeUnits = .all
  ) throws {
    self.model = try Wav2Vec2Model(
      modelPath: modelPath,
      vocabPath: vocabPath,
      computeUnits: computeUnits
    )
    self.tokenizer = try Wav2Vec2Tokenizer(vocabPath: vocabPath)
    self.ctcAligner = CTCForcedAligner(blankId: model.blankId)
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
    let logProbs = try model.getFrameLogProbs(audio)

    // 2. Tokenize text to characters
    let tokens = tokenizer.encode(text)

    // 3. Build ID-to-token mapping for alignment results
    var idToToken: [Int: String] = [:]
    for id in 0..<model.vocabSize {
      if let token = model.token(for: id) {
        idToToken[id] = token
      }
    }

    // 4. Run CTC forced alignment
    let charAlignments = try ctcAligner.align(
      logProbs: logProbs,
      tokens: tokens,
      idToToken: idToToken
    )

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
    return try model.getFrameLogProbs(audio)
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

// MARK: - Convenience Extensions

extension Wav2Vec2Aligner {
  /// Load aligner from model directory containing both .mlpackage and vocab.json
  ///
  /// - Parameters:
  ///   - modelDirectory: Directory containing wav2vec2.mlpackage and vocab.json
  ///   - modelName: Name of the .mlpackage file (without extension)
  ///   - vocabName: Name of the vocabulary JSON file (default: "vocab.json")
  ///   - computeUnits: Compute units for CoreML
  /// - Throws: Wav2Vec2Error if loading fails
  public convenience init(
    modelDirectory: URL,
    modelName: String = "wav2vec2",
    vocabName: String = "vocab.json",
    computeUnits: MLComputeUnits = .all
  ) throws {
    let modelPath = modelDirectory.appendingPathComponent(modelName).appendingPathExtension("mlpackage")
    let vocabPath = modelDirectory.appendingPathComponent(vocabName)

    // Try .mlmodelc if .mlpackage doesn't exist (compiled model)
    let finalModelPath: URL
    if FileManager.default.fileExists(atPath: modelPath.path) {
      finalModelPath = modelPath
    } else {
      let modelCPath = modelDirectory.appendingPathComponent(modelName).appendingPathExtension("mlmodelc")
      guard FileManager.default.fileExists(atPath: modelCPath.path) else {
        throw Wav2Vec2Error.modelNotFound(modelPath)
      }
      finalModelPath = modelCPath
    }

    try self.init(
      modelPath: finalModelPath,
      vocabPath: vocabPath,
      computeUnits: computeUnits
    )
  }
}

// MARK: - Array Extensions for Word Alignment

extension Array where Element == WordAlignment {
  /// Get all words as a single space-separated string
  public var text: String {
    map(\.word).joined(separator: " ")
  }

  /// Find alignment for a specific word (first occurrence)
  public func find(word: String) -> WordAlignment? {
    first { $0.word.caseInsensitiveCompare(word) == .orderedSame }
  }

  /// Get total duration covered by alignments
  public var totalDuration: Float {
    guard let first = first, let last = last else {
      return 0
    }
    return last.endTime - first.startTime
  }

  /// Check if alignments are monotonically increasing in time
  public var isValid: Bool {
    zip(self, dropFirst()).allSatisfy { $0.startTime <= $1.startTime && $0.endTime <= $1.endTime }
  }

  /// Get alignment gaps (silence between words)
  public func gaps() -> [(gap: Float, afterWord: String)] {
    guard count > 1 else { return [] }
    var result: [(Float, String)] = []

    for i in 0..<(count - 1) {
      let gap = self[i + 1].startTime - self[i].endTime
      if gap > 0 {
        result.append((gap, self[i].word))
      }
    }

    return result
  }
}
