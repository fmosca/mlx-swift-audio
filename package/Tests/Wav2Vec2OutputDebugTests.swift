// Copyright © Anthony DePasquale

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXAudio
@testable import Wav2Vec2Aligner

@Suite(.serialized)
struct Wav2Vec2OutputDebugTests {

  @Test func testModelOutputQuality() async throws {
    // Load model with real weights
    let model = try await Wav2Vec2ForCTC.fromPretrained(
      modelId: "facebook/wav2vec2-base-960h"
    )

    // Create test audio (1 second of simple audio)
    let sampleRate = 16000
    let numSamples = 16000  // 1 second
    var audio: [Float] = []
    audio.reserveCapacity(numSamples)
    for i in 0..<numSamples {
      let t = Float(i) / Float(sampleRate)
      audio.append(sin(2 * .pi * 440 * t) * 0.5)
    }

    // Run model
    let audioTensor = MLXArray(audio).reshaped(1, numSamples)
    let output = model(audioTensor)

    // Expected output shape: [1, 49, 32] for 1 second
    #expect(output.shape[0] == 1, "Batch size should be 1")
    #expect(output.shape[1] == 49, "Should have 49 frames for 1 second")
    #expect(output.shape[2] == 32, "Vocab size should be 32")

    // Check that output is log probabilities (all negative)
    let outputArray = output.asArray(Float.self)
    for i in 0..<min(100, outputArray.count) {
      #expect(
        outputArray[i] < 0,
        "Log prob at index \(i) should be negative, got \(outputArray[i])"
      )
    }

    // Check that probabilities are reasonable (not all -inf)
    let maxLogProb = outputArray.max()!
    #expect(
      maxLogProb > -10,
      "Max log prob should be > -10 (not too small), got \(maxLogProb)"
    )
  }
}
