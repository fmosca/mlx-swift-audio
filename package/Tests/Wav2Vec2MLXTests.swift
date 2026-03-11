// Copyright © Anthony DePasquale

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXAudio
@testable import Wav2Vec2Aligner

@Suite(.serialized)
struct Wav2Vec2MLXTests {

  // MARK: - Config Tests

  @Test func testConfigBase() {
    // Verify base config has correct values (hiddenSize=768, numLayers=12)
    let config = Wav2Vec2Config.base

    #expect(config.hiddenSize == 768, "Base config hiddenSize should be 768")
    #expect(config.numHiddenLayers == 12, "Base config numHiddenLayers should be 12")
    #expect(config.vocabSize == 32, "Base config vocabSize should be 32")
    #expect(config.convStride.count == 7, "Base config should have 7 conv layers")
  }

  @Test func testConfigLarge() {
    // Verify large config (hiddenSize=1024, numLayers=24)
    let config = Wav2Vec2Config.large

    #expect(config.hiddenSize == 1024, "Large config hiddenSize should be 1024")
    #expect(config.numHiddenLayers == 24, "Large config numHiddenLayers should be 24")
    #expect(config.numAttentionHeads == 16, "Large config numAttentionHeads should be 16")
    #expect(config.intermediateSize == 4096, "Large config intermediateSize should be 4096")
  }

  @Test func testConfigFrameRate() {
    // Verify frame rate is 50fps (20ms per frame)
    let config = Wav2Vec2Config.base

    let expectedFrameRate = 50.0
    let expectedFrameDuration = 0.02

    #expect(
      config.frameRate == expectedFrameRate,
      "Frame rate should be 50fps"
    )
    #expect(
      config.frameDuration == expectedFrameDuration,
      "Frame duration should be 0.02s"
    )
    #expect(
      config.downsamplingFactor == 320,
      "Downsampling factor should be 320"
    )
  }

  // MARK: - Feature Extractor Tests

  @Test func testFeatureExtractorShape() {
    // Create feature extractor, verify 16000 samples → ~49 frames
    let config = Wav2Vec2Config.base
    let featureExtractor = Wav2Vec2FeatureExtractor(config: config)

    // Create 1 second of audio at 16kHz
    let audioSamples = 16000
    let audio = MLXArray.zeros([1, audioSamples])

    // Run feature extractor
    let features = featureExtractor(audio)

    // Expected output shape: [batch, frames, channels]
    // For 16000 samples with stride 320: (16000 - 1) / 320 + 1 = 49.98... -> 49 frames
    let expectedFrames = 49
    let expectedChannels = 512

    #expect(
      features.shape[0] == 1,
      "Batch dimension should be 1"
    )
    #expect(
      features.shape[1] == expectedFrames,
      "Frame dimension should be 49"
    )
    #expect(
      features.shape[2] == expectedChannels,
      "Channel dimension should be 512"
    )
  }

  @Test func testFeatureExtractorShorterAudio() {
    // Test feature extractor with shorter audio
    let config = Wav2Vec2Config.base
    let featureExtractor = Wav2Vec2FeatureExtractor(config: config)

    // Create 0.5 second of audio at 16kHz
    let audioSamples = 8000
    let audio = MLXArray.zeros([1, audioSamples])

    let features = featureExtractor(audio)

    // For 8000 samples: (8000 - 1) / 320 + 1 = 24.99... -> 24 frames
    let expectedFrames = 24

    #expect(
      features.shape[1] == expectedFrames,
      "Frame dimension should be 24 for 8000 samples"
    )
  }

  // MARK: - Model Forward Pass Tests

  @Test func testModelForwardPass() {
    // Create Wav2Vec2ForCTC with base config, run forward pass on audio
    let config = Wav2Vec2Config.base
    let model = Wav2Vec2ForCTC(config: config)

    // Create audio with simple pattern (1 second at 16kHz)
    let audioSamples = 16000
    let audioData = [Float](repeating: 0.5, count: audioSamples)
    let audio = MLXArray(audioData).reshaped(1, audioSamples)

    // Run forward pass
    let output = model(audio)

    // Output shape should be [batch, frames, vocabSize]
    let expectedFrames = 49
    let expectedVocabSize = 32

    #expect(
      output.shape.count == 3,
      "Output should be 3D tensor"
    )
    #expect(
      output.shape[0] == 1,
      "Batch dimension should be 1"
    )
    #expect(
      output.shape[1] == expectedFrames,
      "Frame dimension should be 49"
    )
    #expect(
      output.shape[2] == expectedVocabSize,
      "Vocab dimension should be 32"
    )
  }

  @Test func testModelOutputFramesCalculation() {
    // Test outputFrames(forSamples:) calculation
    let config = Wav2Vec2Config.base
    let model = Wav2Vec2ForCTC(config: config)

    // Test various sample counts
    let testCases: [(samples: Int, expectedFrames: Int)] = [
      (16000, 49),  // 1 second
      (8000, 24),   // 0.5 seconds
      (3200, 9),    // 0.2 seconds
    ]

    for (samples, expectedFrames) in testCases {
      let calculatedFrames = model.outputFrames(forSamples: samples)
      #expect(
        calculatedFrames == expectedFrames,
        "outputFrames(forSamples: \(samples)) should be \(expectedFrames)"
      )
    }
  }

  @Test func testModelLogSoftmaxOutput() {
    // Verify output is log probabilities (values should be negative)
    let config = Wav2Vec2Config.base
    let model = Wav2Vec2ForCTC(config: config)

    let audioData = [Float](repeating: 0.5, count: 16000)
    let audio = MLXArray(audioData).reshaped(1, 16000)
    let output = model(audio)

    // Log softmax output should be negative (log of probability)
    // Sample a few values to check
    let outputArray = output.asArray(Float.self)
    let sampleValue = outputArray[0]  // First value

    #expect(
      sampleValue < 0,
      "Log softmax output should be negative"
    )

    // Additional check: all values should be negative (log probabilities)
    for i in 0..<min(100, outputArray.count) {
      #expect(
        outputArray[i] < 0,
        "Log softmax output at index \(i) should be negative, got \(outputArray[i])"
      )
    }
  }

  // MARK: - Tokenizer Tests

  @Test func testTokenizerBasicEncodeDecode() {
    // Test tokenizer basic encode/decode functionality
    // Create a simple vocab dictionary for testing
    let vocab: [String: Int] = [
      "<pad>": 0,
      "<unk>": 1,
      "": 2,
      "A": 3,
      "B": 4,
      "C": 5,
      "D": 6,
      "E": 7,
      " ": 8,
    ]

    // Write vocab to temporary file
    let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_vocab_\(UUID().uuidString).json")
    defer {
      try? FileManager.default.removeItem(at: tempURL)
    }

    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    let vocabData = try! encoder.encode(vocab)
    try! vocabData.write(to: tempURL)

    // Create tokenizer
    let tokenizer = try! Wav2Vec2Tokenizer(vocabPath: tempURL)

    // Test encoding
    let text = "ABC"
    let tokens = tokenizer.encode(text)

    #expect(tokens.count == 3, "Encoding 'ABC' should produce 3 tokens")
    #expect(tokens[0] == 3, "Token for 'A' should be 3")
    #expect(tokens[1] == 4, "Token for 'B' should be 4")
    #expect(tokens[2] == 5, "Token for 'C' should be 5")

    // Test decoding
    let decoded = tokenizer.decode(tokens)
    #expect(decoded == "ABC", "Decoding should produce 'ABC'")
  }

  @Test func testTokenizerUppercaseConversion() {
    // Test that tokenizer converts to uppercase
    let vocab: [String: Int] = [
      "<pad>": 0,
      "<unk>": 1,
      "": 2,
      "A": 3,
      "B": 4,
      "C": 5,
    ]

    let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_vocab_\(UUID().uuidString).json")
    defer {
      try? FileManager.default.removeItem(at: tempURL)
    }

    let encoder = JSONEncoder()
    let vocabData = try! encoder.encode(vocab)
    try! vocabData.write(to: tempURL)

    let tokenizer = try! Wav2Vec2Tokenizer(vocabPath: tempURL)

    // Test lowercase input is uppercased
    let tokens = tokenizer.encode("abc")

    #expect(tokens.count == 3, "Encoding 'abc' should produce 3 tokens")
    #expect(tokens[0] == 3, "Lowercase 'a' should map to uppercase token 3")
  }

  @Test func testTokenizerUnknownCharacters() {
    // Test that unknown characters map to unk token
    let vocab: [String: Int] = [
      "<pad>": 0,
      "<unk>": 1,
      "": 2,
      "A": 3,
    ]

    let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_vocab_\(UUID().uuidString).json")
    defer {
      try? FileManager.default.removeItem(at: tempURL)
    }

    let encoder = JSONEncoder()
    let vocabData = try! encoder.encode(vocab)
    try! vocabData.write(to: tempURL)

    let tokenizer = try! Wav2Vec2Tokenizer(vocabPath: tempURL)

    // Test unknown character
    let tokens = tokenizer.encode("AX")  // X is not in vocab

    #expect(tokens.count == 2, "Encoding 'AX' should produce 2 tokens")
    #expect(tokens[0] == 3, "Token for 'A' should be 3")
    #expect(tokens[1] == 1, "Unknown token 'X' should map to unk ID 1")
  }

  @Test func testTokenizerTokenCount() {
    // Test tokenCount method
    let vocab: [String: Int] = [
      "<pad>": 0,
      "<unk>": 1,
      "": 2,
      "A": 3,
      "B": 4,
    ]

    let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_vocab_\(UUID().uuidString).json")
    defer {
      try? FileManager.default.removeItem(at: tempURL)
    }

    let encoder = JSONEncoder()
    let vocabData = try! encoder.encode(vocab)
    try! vocabData.write(to: tempURL)

    let tokenizer = try! Wav2Vec2Tokenizer(vocabPath: tempURL)

    let text = "hello world"
    let count = tokenizer.tokenCount(for: text)

    #expect(count == 11, "tokenCount for 'hello world' should be 11")
  }

  // MARK: - CTC Alignment Tests

  @Test func testCTCAlignmentSimple() {
    // Test CTC alignment with simple input
    let blankId = 0
    let aligner = CTCForcedAligner(blankId: blankId)

    // Create simple log probs
    // 10 frames, vocab size 4 (blank=0, a=1, b=2, c=3)
    // Set high probabilities for tokens at specific frames
    var logProbs = [[Float]](
      repeating: [Float](repeating: -10.0, count: 4),
      count: 10
    )

    // Make frames 0-1 favor "a", 3-4 favor "b", 6-7 favor "c"
    logProbs[0][1] = -0.1  // 'a'
    logProbs[1][1] = -0.1  // 'a'
    logProbs[2][0] = -0.1  // blank
    logProbs[3][2] = -0.1  // 'b'
    logProbs[4][2] = -0.1  // 'b'
    logProbs[5][0] = -0.1  // blank
    logProbs[6][3] = -0.1  // 'c'
    logProbs[7][3] = -0.1  // 'c'
    logProbs[8][0] = -0.1  // blank
    logProbs[9][0] = -0.1  // blank

    let tokens = [1, 2, 3]  // "abc"
    let idToToken: [Int: String] = [1: "a", 2: "b", 3: "c"]

    let alignments = try! aligner.align(
      logProbs: logProbs,
      tokens: tokens,
      idToToken: idToToken
    )

    #expect(alignments.count == 3, "Should have 3 token alignments")
    #expect(alignments[0].token == "a", "First token should be 'a'")
    #expect(alignments[1].token == "b", "Second token should be 'b'")
    #expect(alignments[2].token == "c", "Third token should be 'c'")
  }

  @Test func testCTCAlignmentInsufficientFrames() {
    // Test CTC alignment with insufficient frames (should throw)
    let aligner = CTCForcedAligner(blankId: 0)

    // Create log probs with only 3 frames for 2 tokens
    // Need at least 2*2 + 1 = 5 frames
    let logProbs = [[Float]](
      repeating: [Float](repeating: -1.0, count: 4),
      count: 3
    )

    let tokens = [1, 2]

    var errorThrown = false
    do {
      _ = try aligner.align(logProbs: logProbs, tokens: tokens)
    } catch {
      errorThrown = true
    }

    #expect(errorThrown, "CTC alignment should throw with insufficient frames")
  }

  @Test func testCTCAlignmentEmptyTokens() {
    // Test CTC alignment with empty token array (should throw)
    let aligner = CTCForcedAligner(blankId: 0)

    let logProbs = [[Float]](
      repeating: [Float](repeating: -1.0, count: 4),
      count: 10
    )

    let tokens: [Int] = []

    var errorThrown = false
    do {
      _ = try aligner.align(logProbs: logProbs, tokens: tokens)
    } catch {
      errorThrown = true
    }

    #expect(errorThrown, "CTC alignment should throw with empty tokens")
  }

  @Test func testAlignedTokenTimeCalculation() {
    // Test AlignedToken time calculations (20ms per frame)
    let token = AlignedToken(
      token: "a",
      tokenId: 1,
      startFrame: 0,
      endFrame: 4
    )

    // Start time: 0 * 0.02 = 0.0
    // End time: (4 + 1) * 0.02 = 0.1
    // Duration: endTime - startTime = 0.1 - 0.0 = 0.1
    let expectedStartTime: Float = 0.0
    let expectedEndTime: Float = 0.1
    let duration = token.endTime - token.startTime

    #expect(
      token.startTime == expectedStartTime,
      "Start time should be 0.0s"
    )
    #expect(
      abs(token.endTime - expectedEndTime) < 0.0001,
      "End time should be approximately 0.1s"
    )
    #expect(
      abs(duration - 0.1) < 0.0001,
      "Duration should be approximately 0.1s"
    )
  }

  // MARK: - Real Weight Loading Tests

  // Note: These tests are commented out because they require network downloads
  // and Metal GPU access. To enable them, uncomment the @Test attributes.

  // @Test func testRealWeightLoading() async throws {
  //   // Test loading real weights from Hugging Face
  //   // This test downloads facebook/wav2vec2-base-960h and verifies:
  //   // 1. Model downloads successfully
  //   // 2. Weights load without errors
  //   // 3. Forward pass works on sample audio
  //
  //   let modelId = "facebook/wav2vec2-base-960h"
  //
  //   // Load model from Hugging Face
  //   let model = try await Wav2Vec2ForCTC.fromPretrained(modelId: modelId)
  //
  //   // Create sample audio (1 second at 16kHz)
  //   let audioSamples = 16000
  //   let audio = MLXArray.zeros([1, audioSamples])
  //
  //   // Run forward pass
  //   let output = model(audio)
  //
  //   // Verify output shape: [batch, frames, vocabSize]
  //   #expect(output.shape.count == 3, "Output should be 3D")
  //   #expect(output.shape[0] == 1, "Batch dimension should be 1")
  //   #expect(output.shape[2] == 32, "Vocab size should be 32")
  //
  //   // Verify output contains log probabilities (negative values)
  //   let sampleValue = output.asArray(Float.self)[0]
  //   #expect(sampleValue < 0, "Log softmax output should be negative")
  // }

  @Test func testWeightNameMapping() async throws {
    // Test that weight names from Hugging Face are correctly mapped
    // This helps verify the sanitizeWeights function is working

    let modelId = "facebook/wav2vec2-base-960h"

    // Load weights from Hugging Face
    let (weights, _) = try await Wav2Vec2WeightLoader.loadWeightsAndConfig(
      repoId: modelId
    )

    // Check that important weights are present and correctly named
    let expectedKeys = [
      "featureExtractor.conv_layers.0.conv.weight",
      "featureExtractor.conv_layers.0.layer_norm.weight",
      "encoder.feature_projection.layer_norm.weight",
      "encoder.feature_projection.projection.weight",
      "encoder.pos_conv_embed.conv.weight_g",
      "encoder.pos_conv_embed.conv.weight_v",
      "encoder.layers.0.attention.q_proj.weight",
      "encoder.layers.0.feed_forward.intermediate_dense.weight",
      "encoder.layer_norm.weight",
    ]

    for key in expectedKeys {
      if weights[key] == nil {
        // Print all available keys for debugging
        print("[testWeightNameMapping] Missing key: \(key)")
        print("[testWeightNameMapping] Total keys loaded: \(weights.count)")
        print("[testWeightNameMapping] Keys starting with 'featureP':")
        for availableKey in weights.keys.sorted() {
          if availableKey.hasPrefix("featureP") {
            print("  \(availableKey)")
          }
        }
        print("[testWeightNameMapping] All available keys (first 50):")
        for availableKey in weights.keys.sorted().prefix(50) {
          print("  \(availableKey)")
        }
      }
      #expect(
        weights[key] != nil,
        "Expected weight key '\(key)' not found after sanitization"
      )
    }

    // Verify weight shapes are correct for base model
    if let convWeight = weights["featureExtractor.conv_layers.0.conv.weight"] {
      // Conv layer 0: [outChannels, kernelSize, inChannels] = [512, 10, 1]
      #expect(convWeight.shape[0] == 512, "First conv layer should have 512 output channels")
      #expect(convWeight.shape[1] == 10, "First conv layer should have kernel size 10")
    }

    if let encoderQueryWeight = weights["encoder.layers.0.attention.q_proj.weight"] {
      // Query projection: [hiddenSize, hiddenSize] = [768, 768]
      #expect(encoderQueryWeight.shape[0] == 768, "Query projection should have 768 output features")
      #expect(encoderQueryWeight.shape[1] == 768, "Query projection should have 768 input features")
    }
  }
}
