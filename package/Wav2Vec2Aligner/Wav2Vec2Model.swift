// Copyright © Anthony DePasquale

import CoreML
import Foundation

/// CoreML wrapper for wav2vec2 model inference
///
/// Loads a CoreML-converted wav2vec2 model and provides frame-level
/// log probabilities for CTC forced alignment.
public final class Wav2Vec2Model {
  // MARK: - Properties

  /// The loaded CoreML model
  private let model: MLModel

  /// Vocabulary mapping from token ID to character/string
  private let idToToken: [Int: String]

  /// Vocabulary mapping from character/string to token ID
  private let tokenToId: [String: Int]

  /// The CTC blank token ID (typically 0 for wav2vec2)
  public let blankId: Int

  /// Input name expected by the CoreML model
  private let inputName: String

  /// Output name for logits from the CoreML model
  private let outputName: String

  // MARK: - Initialization

  /// Load a wav2vec2 CoreML model from the specified path
  ///
  /// - Parameters:
  ///   - modelPath: URL to the .mlpackage or .mlmodelc directory
  ///   - vocabPath: URL to the vocab.json file
  ///   - computeUnits: Compute units to use (default: .all for CPU+GPU+ANE)
  /// - Throws: Wav2Vec2Error if model or vocabulary cannot be loaded
  public init(
    modelPath: URL,
    vocabPath: URL,
    computeUnits: MLComputeUnits = .all
  ) throws {
    // Validate model file exists
    guard FileManager.default.fileExists(atPath: modelPath.path) else {
      throw Wav2Vec2Error.modelNotFound(modelPath)
    }

    // Validate vocabulary file exists
    guard FileManager.default.fileExists(atPath: vocabPath.path) else {
      throw Wav2Vec2Error.vocabularyNotFound(vocabPath)
    }

    // Compile .mlpackage to .mlmodelc if needed
    let modelURLToLoad: URL
    if modelPath.pathExtension == "mlpackage" {
      // .mlpackage is a source format that needs compilation
      // Try to compile to a permanent location next to the .mlpackage
      let compiledURL = modelPath.deletingPathExtension().appendingPathExtension("mlmodelc")

      // Check if pre-compiled version exists
      if FileManager.default.fileExists(atPath: compiledURL.path) {
        modelURLToLoad = compiledURL
      } else {
        // Compile the model to a temp location first
        do {
          let tempCompiledURL = try MLModel.compileModel(at: modelPath)
          // Create parent directory if needed
          try FileManager.default.createDirectory(
            at: compiledURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
          )
          // Move compiled model to permanent location
          try FileManager.default.moveItem(at: tempCompiledURL, to: compiledURL)
          modelURLToLoad = compiledURL
        } catch {
          throw Wav2Vec2Error.modelLoadFailed(underlying: error)
        }
      }
    } else {
      // Already compiled (.mlmodelc) - use directly
      modelURLToLoad = modelPath
    }

    // Load CoreML model configuration
    let config = MLModelConfiguration()
    config.computeUnits = computeUnits
    config.allowLowPrecisionAccumulationOnGPU = true

    // Load the model
    do {
      self.model = try MLModel(contentsOf: modelURLToLoad, configuration: config)
    } catch {
      throw Wav2Vec2Error.modelLoadFailed(underlying: error)
    }

    // Determine input/output names from model description
    let inputDescription = model.modelDescription.inputDescriptionsByName
    let outputDescription = model.modelDescription.outputDescriptionsByName

    // Try common input names
    if inputDescription["input_audio"] != nil {
      inputName = "input_audio"
    } else if inputDescription["audio"] != nil {
      inputName = "audio"
    } else if inputDescription["input"] != nil {
      inputName = "input"
    } else if inputDescription["input_1"] != nil {
      inputName = "input_1"
    } else {
      // Use the first available input as fallback
      if let firstInput = inputDescription.keys.first {
        inputName = firstInput
      } else {
        throw Wav2Vec2Error.invalidArgument("Cannot determine input name from model description")
      }
    }

    // Try common output names
    if outputDescription["logits"] != nil {
      outputName = "logits"
    } else if outputDescription["output"] != nil {
      outputName = "output"
    } else if outputDescription["output_1"] != nil {
      outputName = "output_1"
    } else if outputDescription["var_891"] != nil {
      outputName = "var_891"
    } else {
      // Use the first available output as fallback
      if let firstOutput = outputDescription.keys.first {
        outputName = firstOutput
      } else {
        throw Wav2Vec2Error.invalidArgument("Cannot determine output name from model description")
      }
    }

    // Load vocabulary
    do {
      let vocabData = try Data(contentsOf: vocabPath)

      // Try to decode as [String: Int] (token: id) first
      var id2token: [Int: String] = [:]
      var token2id: [String: Int] = [:]
      var blank = 0

      if let vocabDict = try? JSONDecoder().decode([String: Int].self, from: vocabData) {
        // Standard format: {token: id}
        for (token, id) in vocabDict {
          token2id[token] = id
          id2token[id] = token
          if token == "<pad>" || token == "" || id == 0 {
            blank = id
          }
        }
      } else if let vocabDictStr = try? JSONDecoder().decode([String: String].self, from: vocabData) {
        // Alternative format: {id_string: token}
        for (idStr, token) in vocabDictStr {
          if let id = Int(idStr) {
            id2token[id] = token
            token2id[token] = id
            if token == "<pad>" || token == "" || id == 0 {
              blank = id
            }
          }
        }
      } else {
        throw Wav2Vec2Error.invalidVocabulary("Failed to parse vocab.json: unsupported format")
      }

      self.idToToken = id2token
      self.tokenToId = token2id
      self.blankId = blank
    } catch {
      throw Wav2Vec2Error.invalidVocabulary("Failed to parse vocab.json: \(error.localizedDescription)")
    }
  }

  // MARK: - Public Methods

  /// Get frame-level log probabilities for audio input
  ///
  /// Runs the wav2vec2 model and applies log_softmax to the output logits.
  /// Each frame represents 20ms of audio (50 frames per second at 16kHz).
  ///
  /// - Parameter audio: 16kHz mono audio samples as [Float]
  /// - Returns: 2D array of log probabilities [frames][vocab_size]
  /// - Throws: Wav2Vec2Error if inference fails
  public func getFrameLogProbs(_ audio: [Float]) throws -> [[Float]] {
    guard !audio.isEmpty else {
      throw Wav2Vec2Error.invalidAudio("Audio samples array is empty")
    }

    // Validate audio length is reasonable (not too long for fixed input)
    // Note: The converted model supports variable-length input up to 30 seconds (480k samples).
    // For longer audio, chunking is required (future enhancement).
    let maxSamples = 16000 * 30  // 30 seconds max (model conversion limit)
    guard audio.count <= maxSamples else {
      throw Wav2Vec2Error.invalidAudio(
        "Audio too long: \(audio.count) samples (max \(maxSamples) for 30 seconds). " +
        "Longer audio requires chunking support."
      )
    }

    // Create MLMultiArray input
    // Shape: [1, samples] where samples is the audio length
    let samplesCount = audio.count
    guard let inputArray = try? MLMultiArray(
      shape: [1, NSNumber(value: samplesCount)],
      dataType: .float32
    ) else {
      throw Wav2Vec2Error.inferenceFailure(
        underlying: NSError(
          domain: "Wav2Vec2Model",
          code: -1,
          userInfo: [NSLocalizedDescriptionKey: "Failed to create MLMultiArray"]
        )
      )
    }

    // Copy audio data into MLMultiArray
    inputArray.withUnsafeMutableBytes { rawBuffer, _ in
      guard let ptr = rawBuffer.baseAddress?.assumingMemoryBound(to: Float.self) else {
        return
      }
      for (i, sample) in audio.enumerated() {
        ptr[i] = sample
      }
    }

    // Run inference
    let input: MLFeatureProvider
    let output: MLFeatureProvider

    do {
      input = try MLDictionaryFeatureProvider(dictionary: [inputName: inputArray])
      output = try model.prediction(from: input)
    } catch {
      throw Wav2Vec2Error.inferenceFailure(underlying: error)
    }

    // Extract logits
    guard let logitsValue = output.featureValue(for: outputName)?.multiArrayValue else {
      throw Wav2Vec2Error.inferenceFailure(
        underlying: NSError(
          domain: "Wav2Vec2Model",
          code: -2,
          userInfo: [NSLocalizedDescriptionKey: "Output '\(outputName)' not found or not an MLMultiArray"]
        )
      )
    }

    // Convert logits to log probabilities using log_softmax
    return logSoftmax(logitsValue)
  }

  /// Get the vocabulary size
  public var vocabSize: Int {
    idToToken.count
  }

  /// Look up a token string by ID
  public func token(for id: Int) -> String? {
    idToToken[id]
  }

  /// Look up a token ID by string
  public func id(for token: String) -> Int? {
    tokenToId[token]
  }

  // MARK: - Private Methods

  /// Apply log_softmax to logits from the model
  ///
  /// MLMultiArray doesn't provide log_softmax, so we implement it manually.
  /// Uses the numerically stable version: log_softmax(x) = x - log(sum(exp(x)))
  ///
  /// - Parameter logits: MLMultiArray of shape [batch, frames, vocab_size] or [frames, vocab_size]
  /// - Returns: 2D array [frames][vocab_size] of log probabilities
  private func logSoftmax(_ logits: MLMultiArray) -> [[Float]] {
    let shape = logits.shape.map { $0.intValue }
    let strides = logits.strides.map { $0.intValue }

    // Handle different output shapes:
    // - [1, frames, vocab_size] (batched)
    // - [frames, vocab_size] (unbatched)
    let frames: Int
    let vocabSize: Int
    let strideFrame: Int
    let strideVocab: Int
    let baseOffset: Int

    if shape.count == 3 {
      // Batched: [1, frames, vocab_size]
      frames = shape[1]
      vocabSize = shape[2]
      baseOffset = 0
      strideFrame = strides.count == 3 ? strides[1] : vocabSize
      strideVocab = strides.count == 3 ? strides[2] : 1
    } else if shape.count == 2 {
      // Unbatched: [frames, vocab_size]
      frames = shape[0]
      vocabSize = shape[1]
      baseOffset = 0
      strideFrame = strides.count == 2 ? strides[0] : vocabSize
      strideVocab = strides.count == 2 ? strides[1] : 1
    } else {
      return []
    }

    // Calculate capacity for bindMemory: use the largest stride * corresponding dimension
    // For strides [47968, 32, 1] and shape [1, 1499, 32], capacity = 47968 * 1
    let capacity = strides.count > 0 ? strides[0] * shape[0] : 1

    // Access data directly - MLMultiArray memory is valid during this function call
    let dataPointer = logits.dataPointer.bindMemory(to: Float.self, capacity: capacity)

    var result: [[Float]] = []
    result.reserveCapacity(frames)

    for t in 0..<frames {
      let rowStart = baseOffset + t * strideFrame
      var row = [Float](repeating: 0, count: vocabSize)

      // Find max for numerical stability (log-sum-exp trick)
      var maxVal: Float = -.infinity
      for v in 0..<vocabSize {
        let val = dataPointer[rowStart + v * strideVocab]
        if val > maxVal {
          maxVal = val
        }
      }

      // Compute log_sum_exp = max + log(sum(exp(x - max)))
      var sumExp: Float = 0
      for v in 0..<vocabSize {
        sumExp += exp(dataPointer[rowStart + v * strideVocab] - maxVal)
      }
      let logSumExp = maxVal + log(sumExp)

      // Compute log_softmax: x - log_sum_exp
      for v in 0..<vocabSize {
        row[v] = dataPointer[rowStart + v * strideVocab] - logSumExp
      }

      result.append(row)
    }

    return result
  }
}
