// Copyright © 2021 Facebook AI Research (original model implementation)
// Ported to MLX from https://github.com/huggingface/transformers
// Copyright © Anthony DePasquale
// License: licenses/apache-2.0.txt

import Foundation
import Hub
import MLX
import MLXNN

// MARK: - Wav2Vec2 Weight Loader

/// Loads weights and configuration for Wav2Vec2 models from Hugging Face Hub.
///
/// Weight key transformation strategy:
/// 1. Remove "wav2vec2." prefix
/// 2. Convert "feature_extractor." to "featureExtractor." (ONLY module name conversion)
/// 3. Keep ALL other keys as snake_case to match HuggingFace format
/// 4. Transpose conv weights where needed for MLX format
///
/// ALL @ModuleInfo keys should use snake_case to match HuggingFace exactly.
public class Wav2Vec2WeightLoader {

  private init() {}

  /// Load weights and configuration from a Hugging Face repository.
  ///
  /// - Parameters:
  ///   - repoId: Hugging Face repository ID (e.g., "facebook/wav2vec2-base-960h")
  ///   - progressHandler: Optional callback for download progress
  /// - Returns: Tuple of (weights dictionary, model directory URL)
  public static func loadWeightsAndConfig(
    repoId: String,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
  ) async throws -> (weights: [String: MLXArray], modelDirectory: URL) {

    let modelDirectory = try await HubConfiguration.shared.snapshot(
      from: repoId,
      matching: [
        "model.safetensors",
        "vocab.json",
        "config.json"
      ],
      progressHandler: progressHandler
    )

    // Try model.safetensors first, then fall back to other common names
    let weightFileNames = ["model.safetensors", "pytorch_model.bin"]
    var weightFileURL: URL?
    for fileName in weightFileNames {
      let candidateURL = modelDirectory.appending(path: fileName)
      if FileManager.default.fileExists(atPath: candidateURL.path) {
        weightFileURL = candidateURL
        break
      }
    }

    guard let weightFileURL = weightFileURL else {
      throw STTError.modelUnavailable(
        "No model weights found in \(repoId). Tried: \(weightFileNames.joined(separator: ", "))"
      )
    }

    let weights = try MLX.loadArrays(url: weightFileURL)
    let sanitizedWeights = sanitizeWeights(weights)

    return (sanitizedWeights, modelDirectory)
  }

  /// Sanitize and remap Hugging Face weight names to Swift MLX property names.
  ///
  /// Transformation strategy (MINIMAL):
  /// 1. Remove "wav2vec2." prefix
  /// 2. Move feature_projection into encoder (HF has it at top level, our model nests it)
  /// 3. Convert "feature_extractor." to "featureExtractor." (ONLY module name conversion)
  /// 4. Keep ALL else as snake_case (matching HuggingFace)
  /// 5. Transpose conv weights for MLX format where needed
  ///
  /// ALL @ModuleInfo keys MUST use snake_case to match HuggingFace format exactly.
  static func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var sanitizedWeights: [String: MLXArray] = [:]

    for (key, value) in weights {
      var newKey = key
      var newValue = value

      // Step 1: Remove "wav2vec2." prefix
      if newKey.hasPrefix("wav2vec2.") {
        newKey = String(newKey.dropFirst("wav2vec2.".count))
      }

      // Step 1.5: Move feature_projection into encoder (HuggingFace has it at top level)
      // HF: feature_projection.*, Our model: encoder.feature_projection.*
      if newKey.hasPrefix("feature_projection.") {
        newKey = "encoder." + newKey
      }

      // Step 2: Convert ONLY "feature_extractor." to "featureExtractor."
      // This is the ONLY snake_case to camelCase conversion (for module name only)
      newKey = newKey.replacingOccurrences(of: "feature_extractor.", with: "featureExtractor.")

      // Step 3: Handle conv weight transpositions for MLX format
      // Feature extractor conv layers: need transpose
      if newKey.contains("featureExtractor.conv_layers.") && newKey.hasSuffix(".conv.weight") {
        if newValue.ndim == 3 {
          // HF: [outChannels, inChannels, kernelSize]
          // MLX: [outChannels, kernelSize, inChannels]
          newValue = newValue.transposed(0, 2, 1)
        }
      }

      // Positional conv weight_v: need transpose
      if newKey.hasSuffix("pos_conv_embed.conv.weight_v") {
        if newValue.ndim == 3 {
          // HF: [outChannels, inChannels/groups, kernelSize]
          // MLX: [outChannels, kernelSize, inChannels/groups]
          newValue = newValue.transposed(0, 2, 1)
        }
      }

      // Step 4: Keep ALL other keys as snake_case (no transformations needed)

      sanitizedWeights[newKey] = newValue
    }

    return sanitizedWeights
  }
}

// MARK: - Wav2Vec2ForCTC Extension

extension Wav2Vec2ForCTC {

  /// Load a pretrained Wav2Vec2ForCTC model from Hugging Face Hub.
  ///
  /// - Parameters:
  ///   - modelId: Hugging Face model ID (e.g., "facebook/wav2vec2-base-960h")
  ///   - progressHandler: Optional callback for download progress
  /// - Returns: Initialized Wav2Vec2ForCTC model with loaded weights
  public static func fromPretrained(
    modelId: String,
    progressHandler: @escaping @Sendable (Progress) -> Void = { _ in }
  ) async throws -> Wav2Vec2ForCTC {

    Log.model.info("Loading Wav2Vec2 from \(modelId)...")

    // Load weights and config
    let (weights, modelDirectory) = try await Wav2Vec2WeightLoader.loadWeightsAndConfig(
      repoId: modelId,
      progressHandler: progressHandler
    )

    // Load configuration
    let configURL = modelDirectory.appending(path: "config.json")
    let config = try Wav2Vec2Config.load(from: configURL)

    // Create model
    let model = Wav2Vec2ForCTC(config: config)

    // Load weights into model
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.noUnusedKeys])

    // Set model to evaluation mode
    model.train(false)
    eval(model)

    Log.model.info("Wav2Vec2 model loaded from \(modelId)")

    return model
  }
}

// MARK: - Wav2Vec2Config Extension

extension Wav2Vec2Config {

  /// Load Wav2Vec2 configuration from a JSON file.
  ///
  /// - Parameter url: URL to the config.json file
  /// - Returns: Decoded Wav2Vec2Config
  static func load(from url: URL) throws -> Wav2Vec2Config {
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(Wav2Vec2Config.self, from: data)
  }
}
