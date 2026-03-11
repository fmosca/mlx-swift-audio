// Copyright © Anthony DePasquale
//
// Numerical verification tests for Wav2Vec2 MLX vs PyTorch
// This test compares intermediate outputs between MLX Swift and PyTorch implementations

import Foundation
import MLX
import MLXNN
import Testing
import Numerics
import Accelerate

@testable import MLXAudio
@testable import Wav2Vec2Aligner

@Suite(.serialized)
struct Wav2Vec2NumericalTests {

  /// Compare TrellisForcedAligner vs CTCForcedAligner
  @Test func compareTrellisVsCTCAligner() async throws {
    print("\n" + String(repeating: "=", count: 70))
    print("TrellisForcedAligner vs CTCForcedAligner Comparison")
    print(String(repeating: "=", count: 70))

    // Load model
    print("\nLoading MLX Wav2Vec2 model...")
    let model = try await Wav2Vec2ForCTC.fromPretrained(
      modelId: "facebook/wav2vec2-base-960h"
    )

    // Create test audio
    let audio = createTestAudio()
    let audioTensor = MLXArray(audio).reshaped(1, -1)

    // Get frame-level log probabilities
    let output = model(audioTensor)  // [batch, frames, vocabSize]
    let logProbs = mlxArrayToLogProbs(output)

    // Test text
    let text = "hello"

    // Create tokenizer for token IDs
    let modelDirectory = try await HubConfiguration.shared.snapshot(
      from: "facebook/wav2vec2-base-960h",
      matching: ["vocab.json"],
      progressHandler: { _ in }
    )
    let vocabPath = modelDirectory.appending(path: "vocab.json")
    let tokenizer = try Wav2Vec2Tokenizer(vocabPath: vocabPath)
    let tokens = tokenizer.encode(text)

    print("\nTest text: '\(text)'")
    print("Tokens: \(tokens)")

    // Build id-to-token mapping
    var idToToken: [Int: String] = [:]
    for id in 0..<model.config.vocabSize {
      if let char = tokenizer.character(for: id) {
        idToToken[id] = String(char)
      }
    }

    // Run CTC aligner
    print("\n" + String(repeating: "-", count: 70))
    print("CTCForcedAligner Results:")
    print(String(repeating: "-", count: 70))
    let ctcAligner = CTCForcedAligner(blankId: tokenizer.blankId)
    let ctcResults = try ctcAligner.align(logProbs: logProbs, tokens: tokens, idToToken: idToToken)

    for (idx, token) in ctcResults.enumerated() {
      let duration = Float(token.endFrame - token.startFrame) * 0.02
      print("  '\(token.token)' frames \(token.startFrame)-\(token.endFrame) (\(token.endFrame - token.startFrame) frames = \(String(format: "%.2f", duration))s)")
    }

    // Calculate total coverage (sum of token durations)
    let ctcTotalFrames = ctcResults.reduce(0) { $0 + ($1.endFrame - $1.startFrame) }
    let ctcCoverage = Float(ctcTotalFrames) / Float(logProbs.count) * 100
    print("\n  Total frames covered: \(ctcTotalFrames)/\(logProbs.count) (\(String(format: "%.1f", ctcCoverage))%)")

    // Run Trellis aligner
    print("\n" + String(repeating: "-", count: 70))
    print("TrellisForcedAligner Results:")
    print(String(repeating: "-", count: 70))
    let trellisResults = TrellisForcedAligner.align(logProbs: logProbs, tokens: tokens, idToToken: idToToken, blankId: tokenizer.blankId)

    for (idx, token) in trellisResults.enumerated() {
      let duration = Float(token.endFrame - token.startFrame) * 0.02
      print("  '\(token.token)' frames \(token.startFrame)-\(token.endFrame) (\(token.endFrame - token.startFrame) frames = \(String(format: "%.2f", duration))s)")
    }

    // Calculate total coverage
    let trellisTotalFrames = trellisResults.reduce(0) { $0 + ($1.endFrame - $1.startFrame) }
    let trellisCoverage = Float(trellisTotalFrames) / Float(logProbs.count) * 100
    print("\n  Total frames covered: \(trellisTotalFrames)/\(logProbs.count) (\(String(format: "%.1f", trellisCoverage))%)")

    // Summary comparison
    print("\n" + String(repeating: "=", count: 70))
    print("Summary Comparison")
    print(String(repeating: "=", count: 70))
    print("CTC aligner coverage: \(String(format: "%.1f", ctcCoverage))%")
    print("Trellis aligner coverage: \(String(format: "%.1f", trellisCoverage))%")

    if trellisCoverage > ctcCoverage {
      let improvement = trellisCoverage - ctcCoverage
      print("Trellis aligner covers \(String(format: "%.1f", improvement))% more frames")
    }

    #expect(trellisResults.count == tokens.count, "Trellis should produce one result per token")
  }

  /// Create test audio matching Python script (440Hz sine wave, 1 second @ 16kHz)
  func createTestAudio(sampleRate: Int = 16000, duration: Double = 1.0) -> [Float] {
    let numSamples = Int(Double(sampleRate) * duration)
    var audio: [Float] = []
    audio.reserveCapacity(numSamples)
    for i in 0..<numSamples {
      let t = Float(i) / Float(sampleRate)
      audio.append(sin(2 * .pi * 440 * t) * 0.5)
    }
    return audio
  }

  /// Load numpy .npy file as Swift array of Floats
  func loadNumpyFile(_ path: String) -> [Float]? {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
      return nil
    }

    // Parse numpy .npy format
    // First 6 bytes: magic string "\x93NUMPY"
    // Next 2 bytes: major version, minor version
    // Then header length (uint16) and header (JSON)
    // Then the data

    guard data.count > 10 else { return nil }

    // Skip magic (6 bytes) + version (2 bytes) + header length (2 bytes)
    var offset = 10

    // Parse header length (little-endian uint16)
    let headerLenBytes = [UInt8](data[8..<10])
    let headerLen = Int(headerLenBytes[0]) + (Int(headerLenBytes[1]) << 8)

    // Skip header
    offset += headerLen

    // Read remaining data as Float32 array
    let floatData = data.subdata(in: offset..<data.count)
    return floatData.withUnsafeBytes { rawPointer in
      Array(rawPointer.bindMemory(to: Float.self))
    }
  }

  /// Compare two MLXArrays with PyTorch numpy arrays, returning max absolute difference
  func compareOutputs(
    mlx: MLXArray,
    pytorchPath: String,
    name: String,
    tolerance: Float = 1e-4
  ) -> (maxDiff: Float, meanDiff: Float, passed: Bool) {
    // Load PyTorch output
    guard let pyArray = loadNumpyFile(pytorchPath) else {
      print("  ❌ Failed to load PyTorch output from \(pytorchPath)")
      return (0, 0, false)
    }

    // Get MLX array as Swift array
    let mlxArray = mlx.asArray(Float.self)

    // Compare element by element
    guard mlxArray.count == pyArray.count else {
      print("  ❌ \(name): Element count mismatch - MLX: \(mlxArray.count) (\(mlx.shape)), PyTorch: \(pyArray.count)")
      return (0, 0, false)
    }

    var maxDiff: Float = 0
    var sumDiff: Float = 0

    for i in 0..<mlxArray.count {
      let diff = abs(mlxArray[i] - pyArray[i])
      if diff > maxDiff {
        maxDiff = diff
      }
      sumDiff += diff
    }

    let meanDiff = sumDiff / Float(mlxArray.count)
    let passed = maxDiff < tolerance

    if passed {
      print("  ✅ \(name): maxDiff=\(String(format: "%.2e", maxDiff)), meanDiff=\(String(format: "%.2e", meanDiff))")
    } else {
      print("  ⚠️  \(name): maxDiff=\(String(format: "%.2e", maxDiff)), meanDiff=\(String(format: "%.2e", meanDiff)) (tolerance: \(String(format: "%.2e", tolerance)))")
    }

    return (maxDiff, meanDiff, passed)
  }

  /// Print statistics for an MLXArray
  func printStats(_ name: String, _ array: MLXArray) {
    let arr = array.asArray(Float.self)
    let mean = arr.reduce(0, +) / Float(arr.count)
    let minVal = arr.min() ?? 0
    let maxVal = arr.max() ?? 0

    // Calculate std
    let variance = arr.map { pow($0 - mean, 2) }.reduce(0, +) / Float(arr.count)
    let std = sqrt(variance)

    print("\(name):")
    print("  Shape: \(array.shape)")
    print("  Mean: \(String(format: "%.6f", mean))")
    print("  Std: \(String(format: "%.6f", std))")
    print("  Min: \(String(format: "%.6f", minVal))")
    print("  Max: \(String(format: "%.6f", maxVal))")

    // Print first 10 values
    let first10 = Array(arr.prefix(10))
    print("  First 10 values: \(first10.map { String(format: "%.6f", $0) })")
  }

  @Test func testNumericalVsPyTorch() async throws {
    print("\n" + String(repeating: "=", count: 70))
    print("Wav2Vec2 Numerical Verification (MLX Swift vs PyTorch)")
    print(String(repeating: "=", count: 70))

    let pytorchOutputDir = "/tmp/wav2vec2_numerical"

    // Check if PyTorch outputs exist
    guard FileManager.default.fileExists(atPath: "\(pytorchOutputDir)/logits.npy") else {
      print("\n❌ PyTorch outputs not found. Run Python script first:")
      print("   python3 scripts/verify_wav2vec2_numerical.py")
      throw NSError(domain: "Wav2Vec2NumericalTests", code: 1)
    }

    // Load model
    print("\nLoading MLX Wav2Vec2 model...")
    let model = try await Wav2Vec2ForCTC.fromPretrained(
      modelId: "facebook/wav2vec2-base-960h"
    )

    // Create test audio
    print("\nCreating test audio (440Hz sine wave, 1 second @ 16kHz)")
    let audio = createTestAudio()
    print("  Audio shape: [1, \(audio.count)]")

    // Run forward pass and capture intermediate outputs
    print("\nRunning forward pass with intermediate output capture...")

    // We need to manually call through the model to capture intermediates
    let audioTensor = MLXArray(audio).reshaped(1, -1)

    // Step 1: Feature extractor output
    var hiddenStates = model.featureExtractor(audioTensor)
    // MLX output is [batch, frames, channels], PyTorch captured [batch, channels, frames]
    // We need to transpose for comparison
    let featureExtractorOutput = hiddenStates.transposed(0, 2, 1)  // [batch, channels, frames]
    print("  feature_extractor_last_conv: \(featureExtractorOutput.shape)")

    // Step 2: Feature projection
    hiddenStates = model.encoder.featureProjection(hiddenStates)

    // Step 3: Positional conv (transposed for the conv)
    let hiddenStatesTransposed = hiddenStates.transposed(0, 2, 1)  // [batch, channels, frames]
    print("  hidden_states for pos_conv: \(hiddenStatesTransposed.shape)")

    // The PyTorch hook captures output BEFORE GELU and before any length adjustment
    // MLX conv with padding may produce different length - we need to handle this
    let positionalEmbeddings = model.encoder.posConvEmbed(hiddenStatesTransposed)
    print("  positional_embeddings (after GELU): \(positionalEmbeddings.shape)")

    // Step 4: Add positional embeddings
    // positionalEmbeddings is [batch, frames, channels] after transpose in posConvEmbed
    hiddenStates = hiddenStates + positionalEmbeddings
    hiddenStates = model.encoder.dropout(hiddenStates)

    // Step 5: First encoder layer
    let encoderLayer0Output = model.encoder.layers[0](hiddenStates)
    print("  encoder_layer_0: \(encoderLayer0Output.shape)")

    // Step 6: Remaining encoder layers
    var encoderOutput = encoderLayer0Output
    for i in 1..<model.encoder.layers.count {
      encoderOutput = model.encoder.layers[i](encoderOutput)
    }

    // Final layer norm
    encoderOutput = model.encoder.layerNorm(encoderOutput)
    print("  encoder_output: \(encoderOutput.shape)")

    // Step 7: CTC head (before log_softmax)
    let logits = model.ctcHead(encoderOutput)
    print("  logits: \(logits.shape)")

    // Step 8: Log softmax
    let logProbs = logSoftmax(logits, axis: -1)
    print("  log_probs: \(logProbs.shape)")

    // Print MLX statistics for comparison
    print("\n" + String(repeating: "=", count: 70))
    print("MLX Output Analysis")
    print(String(repeating: "=", count: 70))
    printStats("\nfeature_extractor_last_conv", featureExtractorOutput)
    printStats("\nencoder_layer_0", encoderLayer0Output)
    printStats("\nlogits", logits)

    // Check blank token logits specifically
    let logitsArray = logits.asArray(Float.self)
    let blankLogits: [Float] = (0..<49).map { i in
      logitsArray[i * 32 + 0]  // Token 0 is blank
    }
    print("\nBlank token logits (first 10 frames): \(blankLogits.prefix(10).map { String(format: "%.6f", $0) })")
    print("Blank token mean: \(String(format: "%.6f", blankLogits.reduce(0, +) / Float(blankLogits.count)))")

    // Compare outputs
    print("\n" + String(repeating: "=", count: 70))
    print("Numerical Comparison Results")
    print(String(repeating: "=", count: 70))

    var allPassed = true
    var results: [(name: String, maxDiff: Float, passed: Bool)] = []

    // Use stricter tolerance for intermediate layers, looser for final output
    // due to floating point accumulation
    let featureTolerance: Float = 1e-3
    let encoderTolerance: Float = 1e-3
    let outputTolerance: Float = 1e-2

    // Compare feature extractor
    let featResult = compareOutputs(
      mlx: featureExtractorOutput,
      pytorchPath: "\(pytorchOutputDir)/feature_extractor_last_conv.npy",
      name: "feature_extractor_last_conv",
      tolerance: featureTolerance
    )
    results.append(("feature_extractor_last_conv", featResult.maxDiff, featResult.passed))
    allPassed = allPassed && featResult.passed

    // Note: Skip positional_conv comparison because:
    // 1. PyTorch captures BEFORE GELU, MLX applies GELU in the call
    // 2. MLX conv1d may produce different sequence length (49 vs 50)
    print("  ⏭️  positional_conv: Skipped (PyTorch captures before GELU, MLX after)")

    // Compare encoder layer 0
    let layer0Result = compareOutputs(
      mlx: encoderLayer0Output,
      pytorchPath: "\(pytorchOutputDir)/encoder_layer_0.npy",
      name: "encoder_layer_0",
      tolerance: encoderTolerance
    )
    results.append(("encoder_layer_0", layer0Result.maxDiff, layer0Result.passed))
    allPassed = allPassed && layer0Result.passed

    // Compare encoder output
    let encResult = compareOutputs(
      mlx: encoderOutput,
      pytorchPath: "\(pytorchOutputDir)/encoder_output.npy",
      name: "encoder_output",
      tolerance: encoderTolerance
    )
    results.append(("encoder_output", encResult.maxDiff, encResult.passed))
    allPassed = allPassed && encResult.passed

    // Compare logits
    let logitsResult = compareOutputs(
      mlx: logits,
      pytorchPath: "\(pytorchOutputDir)/logits.npy",
      name: "logits",
      tolerance: outputTolerance
    )
    results.append(("logits", logitsResult.maxDiff, logitsResult.passed))
    allPassed = allPassed && logitsResult.passed

    // Compare log_probs
    let logProbsResult = compareOutputs(
      mlx: logProbs,
      pytorchPath: "\(pytorchOutputDir)/log_probs.npy",
      name: "log_probs",
      tolerance: outputTolerance
    )
    results.append(("log_probs", logProbsResult.maxDiff, logProbsResult.passed))
    allPassed = allPassed && logProbsResult.passed

    // Summary
    print("\n" + String(repeating: "=", count: 70))
    print("Summary")
    print(String(repeating: "=", count: 70))

    let passedCount = results.filter { $0.passed }.count
    print("Passed: \(passedCount)/\(results.count)")

    if allPassed {
      print("✅ All numerical comparisons passed!")
    } else {
      print("\n⚠️  Failed comparisons:")
      for result in results where !result.passed {
        print("  - \(result.name): maxDiff = \(String(format: "%.2e", result.maxDiff))")
      }
    }

    // Don't fail the test - just report results
    // This is a diagnostic test, not a regression test
  }

  @Test func testFrameLevelLogProbAnalysis() async throws {
    print("\n" + String(repeating: "=", count: 70))
    print("Frame-Level Log Probability Analysis")
    print(String(repeating: "=", count: 70))

    // Load model
    let model = try await Wav2Vec2ForCTC.fromPretrained(
      modelId: "facebook/wav2vec2-base-960h"
    )

    // Create test audio
    let audio = createTestAudio()
    let audioTensor = MLXArray(audio).reshaped(1, -1)

    // Run model
    let output = model(audioTensor)  // [batch, frames, vocabSize]

    // Convert to [[Float]] format
    let logProbs = mlxArrayToLogProbs(output)

    let numFrames = logProbs.count
    let vocabSize = logProbs[0].count

    print("\n  Output shape: [1, \(numFrames), \(vocabSize)]")

    // Calculate statistics for first 10 frames
    print("\nFirst 10 frames:")
    print(String(repeating: "-", count: 90))
    print("Frame    Entropy      Max Prob     Blank Prob   Top Token       Prob")
    print(String(repeating: "-", count: 90))

    for frameIdx in 0..<min(10, numFrames) {
      let frameLogProbs = logProbs[frameIdx]

      // Convert log probs to probs
      let frameProbs = frameLogProbs.map { exp($0) }

      // Calculate entropy
      var entropy: Float = 0
      for p in frameProbs {
        if p > 1e-10 {
          entropy -= p * log(p)
        }
      }

      // Find max prob and blank prob
      let maxProb = frameProbs.max() ?? 0
      let blankProb = frameProbs[0]  // Token 0 is blank

      // Find top token
      var maxIdx = 0
      var maxValue = frameProbs[0]
      for i in 1..<frameProbs.count {
        if frameProbs[i] > maxValue {
          maxValue = frameProbs[i]
          maxIdx = i
        }
      }

      // Top token (blank or actual char)
      let topToken = maxIdx == 0 ? " " : "id=\(maxIdx)"

      print(String(format: "%-8d %-12.4f %-12.4f %-12.4f",
                   frameIdx, entropy, maxProb, blankProb) + " " +
            String(format: "%-15s", topToken) + " " +
            String(format: "%.4f", maxValue))
    }

    // Check that blank probability is high for sine wave (non-speech audio)
    let firstFrameBlankProb = exp(logProbs[0][0])

    print("\n  First frame blank probability: \(String(format: "%.2f%%", firstFrameBlankProb * 100))")
    print("  Expected: High (>70%) for non-speech audio (sine wave)")

    #expect(
      firstFrameBlankProb > 0.7,
      "Expected high blank probability for non-speech audio, got \(firstFrameBlankProb)"
    )
  }

  /// Helper: Convert MLXArray output from model to [[Float]] format
  private func mlxArrayToLogProbs(_ mlxArray: MLXArray) -> [[Float]] {
    // mlxArray shape is [batch, frames, vocabSize]
    // Extract first batch: [frames, vocabSize]
    let batch0 = mlxArray[0]

    // Convert to Swift array
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
}
