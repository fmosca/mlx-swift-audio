// Copyright © Anthony DePasquale

import CoreML
import Foundation
import XCTest
@testable import Wav2Vec2Aligner

final class Wav2Vec2AlignerTests: XCTestCase {
  var projectRoot: URL!

  override func setUp() async throws {
    try await super.setUp()
    // Find project root by searching upward from test bundle
    // During xcodebuild test, the test bundle is in DerivedData,
    // so we need to find the actual source directory
    if let envPath = ProcessInfo.processInfo.environment["MLX_SWIFT_AUDIO_PROJECT_ROOT"] {
      projectRoot = URL(fileURLWithPath: envPath)
    } else {
      // Use a hardcoded path for now (assumes running from the project)
      projectRoot = URL(fileURLWithPath: "/Users/francesco.mosca/Work/mlx-swift-audio")
    }
    FileManager.default.changeCurrentDirectoryPath(projectRoot.path)
  }

  func testModelLoad() async throws {
    let modelsDir = projectRoot.appendingPathComponent("models/wav2vec2")
    let aligner = try Wav2Vec2Aligner(
      modelDirectory: modelsDir,
      modelName: "Wav2Vec2CTC",
      vocabName: "vocab.json",
      computeUnits: .all
    )
    XCTAssertNotNil(aligner)
    print("[TEST] Model loaded successfully")
    print("[TEST] Vocab size: \(aligner.tokenize("test").count)")
  }

  func testShortAudioAlignment() async throws {
    let modelsDir = projectRoot.appendingPathComponent("models/wav2vec2")
    let aligner = try Wav2Vec2Aligner(
      modelDirectory: modelsDir,
      modelName: "Wav2Vec2CTC",
      vocabName: "vocab.json",
      computeUnits: .all
    )

    // Create 1 second of silence audio at 16kHz
    let samples16k = [Float](repeating: 0.0, count: 16000)

    print("[TEST] Testing short audio alignment (\(samples16k.count) samples)")

    // Get frame log probs
    let logProbs = try aligner.getFrameLogProbs(samples16k)
    print("[TEST] Log probs shape: \(logProbs.count) frames x \(logProbs.first?.count ?? 0) vocab")
    XCTAssertFalse(logProbs.isEmpty, "Log probs should not be empty")
    XCTAssertFalse(logProbs.first?.isEmpty ?? true, "First frame should have values")
  }

  func testTextAlignment() async throws {
    let modelsDir = projectRoot.appendingPathComponent("models/wav2vec2")
    let aligner = try Wav2Vec2Aligner(
      modelDirectory: modelsDir,
      modelName: "Wav2Vec2CTC",
      vocabName: "vocab.json",
      computeUnits: .all
    )

    // Create 1 second of silence audio at 16kHz
    let samples16k = [Float](repeating: 0.0, count: 16000)
    let text = "hello world"

    print("[TEST] Testing text alignment")

    let alignments = try aligner.align(audio: samples16k, text: text)
    print("[TEST] Got \(alignments.count) word alignments")
    for alignment in alignments {
      print("[TEST]   [\(alignment.startTime)..\(alignment.endTime)] \(alignment.word)")
    }
  }
}
