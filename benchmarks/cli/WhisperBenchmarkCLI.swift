import AVFoundation
import Foundation
import MLXAudio

@main
struct WhisperBenchmarkCLI {
  static let fixturesPath = "benchmarks/fixtures"
  static let modelSize: WhisperModelSize = .largeTurbo
  static let quantization: WhisperQuantization = .q4

  static let pythonBaselineRTF = 0.0500
  static let swiftTargetRTF = pythonBaselineRTF * 0.7

  static func main() async {
    let args = CommandLine.arguments
    let mode = args.count > 1 ? args[1] : "quick"

    print(String(repeating: "=", count: 70))
    print("Whisper Benchmark CLI - \(mode.uppercased()) mode")
    print(String(repeating: "=", count: 70))

    do {
      switch mode {
      case "quick":
        try await runBenchmark(audioFile: "ami_ES2002a_5min.wav", runs: 1)
      case "full":
        try await runBenchmark(audioFile: "ami_ES2002a_full.wav", runs: 3)
      default:
        print("Usage: WhisperBenchmark [quick|full]")
        print("  quick - 5-minute segment, 1 run")
        print("  full  - 21-minute audio, 3 runs")
      }
    } catch {
      print("Error: \(error)")
    }
  }

  @MainActor
  static func runBenchmark(audioFile: String, runs: Int) async throws {
    let audioURL = URL(fileURLWithPath: fixturesPath).appendingPathComponent(audioFile)

    guard FileManager.default.fileExists(atPath: audioURL.path) else {
      print("Error: Audio fixture not found: \(audioURL.path)")
      print("Run from the mlx-swift-audio root directory.")
      return
    }

    let audioDuration = try getAudioDuration(audioURL)
    print("\nAudio: \(audioFile)")
    print("Duration: \(String(format: "%.1f", audioDuration))s (\(String(format: "%.1f", audioDuration / 60)) minutes)")

    print("\nLoading model: \(modelSize.rawValue) [\(quantization.rawValue)]...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let engine = STT.whisper(model: modelSize, quantization: quantization)
    try await engine.load()
    let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
    print("Model load time: \(String(format: "%.2f", loadTime))s")

    print("\nWarmup run...")
    let warmupStart = CFAbsoluteTimeGetCurrent()
    _ = try await engine.transcribe(audioURL, language: .english, temperature: 0.0, timestamps: .segment)
    let warmupTime = CFAbsoluteTimeGetCurrent() - warmupStart
    print("Warmup RTF: \(String(format: "%.4f", warmupTime / audioDuration))")

    print("\nBenchmark runs (\(runs) iterations)...")
    var rtfs: [Double] = []

    for run in 1...runs {
      let startTime = CFAbsoluteTimeGetCurrent()
      let result = try await engine.transcribe(audioURL, language: .english, temperature: 0.0, timestamps: .segment)
      let processingTime = CFAbsoluteTimeGetCurrent() - startTime
      let rtf = processingTime / audioDuration

      rtfs.append(rtf)
      let wordCount = result.text.split(separator: " ").count
      print("  Run \(run): RTF=\(String(format: "%.4f", rtf)), time=\(String(format: "%.1f", processingTime))s, words=\(wordCount)")
    }

    let avgRTF = rtfs.reduce(0, +) / Double(rtfs.count)
    let minRTF = rtfs.min()!
    let maxRTF = rtfs.max()!

    print("\n" + String(repeating: "-", count: 70))
    print("RESULTS")
    print(String(repeating: "-", count: 70))
    print("  Average RTF: \(String(format: "%.4f", avgRTF))")
    if runs > 1 {
      print("  Min RTF:     \(String(format: "%.4f", minRTF))")
      print("  Max RTF:     \(String(format: "%.4f", maxRTF))")
    }

    print("\n" + String(repeating: "=", count: 70))
    print("COMPARISON")
    print(String(repeating: "=", count: 70))
    print("  Python baseline RTF: \(String(format: "%.4f", pythonBaselineRTF))")
    print("  Swift target RTF:    \(String(format: "%.4f", swiftTargetRTF)) (30% faster than Python)")
    print("  Swift actual RTF:    \(String(format: "%.4f", avgRTF))")

    let vsPython = (pythonBaselineRTF - avgRTF) / pythonBaselineRTF * 100

    if avgRTF <= swiftTargetRTF {
      print("\n✅ TARGET MET: \(String(format: "%.1f", vsPython))% faster than Python")
    } else if avgRTF <= pythonBaselineRTF {
      let needed = (avgRTF - swiftTargetRTF) / avgRTF * 100
      print("\n⚠️  \(String(format: "%.1f", vsPython))% faster than Python, need \(String(format: "%.1f", needed))% more for target")
    } else {
      print("\n❌ SLOWER than Python by \(String(format: "%.1f", -vsPython))%")
    }
  }

  static func getAudioDuration(_ url: URL) throws -> Double {
    let file = try AVAudioFile(forReading: url)
    return Double(file.length) / file.processingFormat.sampleRate
  }
}
