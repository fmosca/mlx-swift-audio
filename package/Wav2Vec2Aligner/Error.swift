// Copyright © Anthony DePasquale

import Foundation

/// Unified error type for wav2vec2 alignment operations
public enum Wav2Vec2Error: LocalizedError {
  /// CoreML model hasn't been loaded yet
  case modelNotLoaded

  /// Model loading failed
  case modelLoadFailed(underlying: Error)

  /// Model file not found at specified path
  case modelNotFound(URL)

  /// Vocabulary file not found or invalid
  case vocabularyNotFound(URL)

  /// Vocabulary file format is invalid
  case invalidVocabulary(String)

  /// Audio input is invalid or empty
  case invalidAudio(String)

  /// Text input is invalid or empty
  case invalidText(String)

  /// Inference failed during model execution
  case inferenceFailure(underlying: Error)

  /// CTC alignment failed
  case alignmentFailure(String)

  /// File I/O error
  case fileIOError(underlying: Error)

  /// Invalid configuration or arguments
  case invalidArgument(String)

  // MARK: - LocalizedError

  public var errorDescription: String? {
    switch self {
      case .modelNotLoaded:
        "Model not loaded. Call load() first."
      case let .modelLoadFailed(error):
        "Failed to load model: \(error.localizedDescription)"
      case let .modelNotFound(url):
        "Model not found at: \(url.path)"
      case let .vocabularyNotFound(url):
        "Vocabulary file not found at: \(url.path)"
      case let .invalidVocabulary(message):
        "Invalid vocabulary format: \(message)"
      case let .invalidAudio(message):
        "Invalid audio: \(message)"
      case let .invalidText(message):
        "Invalid text: \(message)"
      case let .inferenceFailure(error):
        "Inference failed: \(error.localizedDescription)"
      case let .alignmentFailure(message):
        "Alignment failed: \(message)"
      case let .fileIOError(error):
        "File I/O error: \(error.localizedDescription)"
      case let .invalidArgument(message):
        "Invalid argument: \(message)"
    }
  }

  public var failureReason: String? {
    switch self {
      case .modelNotLoaded:
        "The wav2vec2 model must be loaded before performing alignment."
      case .modelLoadFailed:
        "The CoreML model could not be loaded or compiled."
      case .modelNotFound:
        "The specified .mlpackage or .mlmodelc file does not exist."
      case .vocabularyNotFound:
        "The vocab.json file could not be found."
      case .invalidVocabulary:
        "The vocabulary file is not valid JSON or has unexpected format."
      case .invalidAudio:
        "The audio data is empty, not at 16kHz, or in an invalid format."
      case .invalidText:
        "The text is empty or contains unsupported characters."
      case .inferenceFailure:
        "An error occurred during CoreML model execution."
      case .alignmentFailure:
        "The CTC forced alignment algorithm could not align the text to the audio."
      case .fileIOError:
        "A file system operation failed."
      case .invalidArgument:
        "An invalid argument was provided to the method."
    }
  }

  public var recoverySuggestion: String? {
    switch self {
      case .modelNotLoaded:
        "Initialize the Wav2Vec2Aligner with valid model and vocabulary paths."
      case .modelLoadFailed:
        "Ensure the model file is a valid CoreML model and compute units are available."
      case .modelNotFound:
        "Check that the model path is correct and the file exists."
      case .vocabularyNotFound:
        "Check that the vocab.json path is correct and the file exists."
      case .invalidVocabulary:
        "Ensure vocab.json is valid JSON with character-to-id mappings."
      case .invalidAudio:
        "Provide non-empty audio at 16kHz sample rate as [Float]."
      case .invalidText:
        "Provide non-empty text with supported characters."
      case .inferenceFailure:
        "Check that the ANE/GPU is available and the model input shape matches audio length."
      case .alignmentFailure:
        "Ensure the text matches the audio content and has sufficient audio length."
      case .fileIOError:
        "Check file permissions and available disk space."
      case .invalidArgument:
        "Review the method documentation for valid argument values."
    }
  }
}
