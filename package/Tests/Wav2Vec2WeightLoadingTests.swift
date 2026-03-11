// Copyright © Anthony DePasquale

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXAudio
@testable import Wav2Vec2Aligner

/// Tests that verify ALL HuggingFace weights can be loaded without errors.
/// This is a fast integration test that catches weight key mismatches early.
@Suite(.serialized)
struct Wav2Vec2WeightLoadingTests {

  @Test func testAllWeightsLoadWithoutUnusedKeys() async throws {
    // Load weights from HuggingFace
    let modelId = "facebook/wav2vec2-base-960h"
    let (weights, _) = try await Wav2Vec2WeightLoader.loadWeightsAndConfig(
      repoId: modelId
    )

    // Load config
    let config = Wav2Vec2Config.base

    // Create model
    let model = Wav2Vec2ForCTC(config: config)

    // Attempt to load ALL weights - this will throw if there are mismatches
    let parameters = ModuleParameters.unflattened(weights)

    // This should not throw with .noUnusedKeys
    try model.update(parameters: parameters, verify: [.noUnusedKeys])

    // If we get here, all weights loaded successfully
    #expect(true, "All weights loaded without unused keys")
  }

  @Test func testWeightKeySanityCheck() async throws {
    // Verify that the weight loader produces keys that match @ModuleInfo expectations
    let modelId = "facebook/wav2vec2-base-960h"
    let (weights, _) = try await Wav2Vec2WeightLoader.loadWeightsAndConfig(
      repoId: modelId
    )

    // Expected key patterns (all snake_case to match HF)
    let expectedPatterns = [
      "featureExtractor.conv_layers",
      "feature_projection",
      "encoder.pos_conv_embed.conv",
      "encoder.layers",
      "encoder.layer_norm",
    ]

    for pattern in expectedPatterns {
      let hasMatch = weights.keys.contains { $0.contains(pattern) }
      #expect(
        hasMatch,
        "Weights should contain keys matching pattern '\(pattern)'"
      )
    }

    // NO camelCase keys should exist (except featureExtractor which is intentional)
    let camelCaseKeys = weights.keys.filter { key in
      key.range(of: "[a-z][A-Z]", options: .regularExpression) != nil
        && !key.hasPrefix("featureExtractor")
    }

    #expect(
      camelCaseKeys.isEmpty,
      "No camelCase keys should exist (except featureExtractor). Found: \(camelCaseKeys)"
    )
  }

  @Test func testModelForwardPassWithRealWeights() async throws {
    // Test that the model can actually do a forward pass with real weights
    let modelId = "facebook/wav2vec2-base-960h"
    let model = try await Wav2Vec2ForCTC.fromPretrained(modelId: modelId)

    // Create short audio (0.5 seconds at 16kHz)
    let audioSamples = 8000
    let audio = MLXArray.zeros([1, audioSamples])

    // Run forward pass
    let output = model(audio)

    // Verify output shape
    #expect(output.shape.count == 3, "Output should be 3D")
    #expect(output.shape[0] == 1, "Batch dimension should be 1")
    #expect(output.shape[2] == 32, "Vocab size should be 32")
  }
}
