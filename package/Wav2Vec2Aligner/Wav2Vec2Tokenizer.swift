// Copyright © Anthony DePasquale

import Foundation

/// Character-level tokenizer for wav2vec2
///
/// wav2vec2 uses a simple character vocabulary where each character
/// maps to a single token ID. This tokenizer handles the conversion
/// between text strings and token IDs.
public final class Wav2Vec2Tokenizer {
  // MARK: - Properties

  /// Mapping from character to token ID
  private let charToId: [Character: Int]

  /// Mapping from token ID to character
  private let idToChar: [Int: Character]

  /// Special tokens
  public let blankId: Int
  public let padId: Int
  public let unkId: Int
  public let vocabSize: Int

  // MARK: - Initialization

  /// Load tokenizer from vocabulary JSON file
  ///
  /// The vocab.json file should contain mappings from strings (characters)
  /// to integer token IDs.
  ///
  /// - Parameter vocabPath: URL to the vocab.json file
  /// - Throws: Wav2Vec2Error if vocabulary cannot be loaded
  public init(vocabPath: URL) throws {
    guard FileManager.default.fileExists(atPath: vocabPath.path) else {
      throw Wav2Vec2Error.vocabularyNotFound(vocabPath)
    }

    let data: Data
    do {
      data = try Data(contentsOf: vocabPath)
    } catch {
      throw Wav2Vec2Error.fileIOError(underlying: error)
    }

    let vocabDict: [String: Int]
    do {
      vocabDict = try JSONDecoder().decode([String: Int].self, from: data)
    } catch {
      throw Wav2Vec2Error.invalidVocabulary("Failed to decode JSON: \(error.localizedDescription)")
    }

    var c2i: [Character: Int] = [:]
    var i2c: [Int: Character] = [:]

    // Find special tokens
    var blank = 0
    var pad = 0
    var unk = 1

    for (token, id) in vocabDict {
      // Single-character tokens go into char mappings
      if token.count == 1, let char = token.first {
        c2i[char] = id
        i2c[id] = char
      }

      // Identify special tokens
      switch token {
      case "<pad>", "[PAD]":
        pad = id
      case "<unk>", "[UNK]":
        unk = id
      case "", " ":
        // Some models use empty string or space as blank
        if id == 0 {
          blank = id
        }
      default:
        break
      }
    }

    self.charToId = c2i
    self.idToChar = i2c
    self.blankId = blank
    self.padId = pad
    self.unkId = unk
    self.vocabSize = vocabDict.count
  }

  // MARK: - Public Methods

  /// Encode text to token IDs
  ///
  /// Text is uppercased (wav2vec2 standard) and each character
  /// is mapped to its corresponding token ID. Unknown characters
  /// are mapped to the unknown token ID.
  ///
  /// - Parameter text: Input text string
  /// - Returns: Array of token IDs
  public func encode(_ text: String) -> [Int] {
    guard !text.isEmpty else {
      return []
    }

    // wav2vec2 uses uppercase for English
    let normalized = text.uppercased()
    return normalized.map { char in
      charToId[char] ?? unkId
    }
  }

  /// Decode token IDs back to text
  ///
  /// Converts token IDs back to their character representations.
  /// Tokens without character mappings are skipped.
  ///
  /// - Parameter ids: Array of token IDs
  /// - Returns: Decoded text string
  public func decode(_ ids: [Int]) -> String {
    guard !ids.isEmpty else {
      return ""
    }

    return String(ids.compactMap { idToChar[$0] })
  }

  /// Get token ID for a character
  ///
  /// - Parameter char: Single character
  /// - Returns: Token ID, or nil if character not in vocabulary
  public func id(for char: Character) -> Int? {
    charToId[char]
  }

  /// Get character for a token ID
  ///
  /// - Parameter id: Token ID
  /// - Returns: Character, or nil if ID not in vocabulary
  public func character(for id: Int) -> Character? {
    idToChar[id]
  }

  /// Check if a character is in the vocabulary
  ///
  /// - Parameter char: Character to check
  /// - Returns: true if character has a token ID
  public func contains(_ char: Character) -> Bool {
    charToId[char] != nil
  }

  /// Get the number of tokens for a text string
  ///
  /// Useful for pre-validating text length before alignment.
  ///
  /// - Parameter text: Input text
  /// - Returns: Number of character tokens (uppercase)
  public func tokenCount(for text: String) -> Int {
    text.uppercased().count
  }
}
