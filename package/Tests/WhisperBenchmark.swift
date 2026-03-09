// Whisper STT Benchmark
// For validating performance optimizations against baseline
//
// Run with: swift test --filter WhisperBenchmark
// Or for specific tests: swift test --filter WhisperBenchmark/longFormBenchmark

import AVFoundation
import Foundation
import Testing

@testable import MLXAudio

// MARK: - Memory Management

private func configureMemoryLimits() {
  MLXMemory.configure(cacheLimit: 1024 * 1024 * 1024)  // 1GB for long audio
  MLXMemory.logStats(prefix: "Initial")
}

private func clearMemoryBetweenRuns() {
  MLXMemory.clearCache()
}

// MARK: - Benchmark Results

struct BenchmarkResult {
  let rtf: Double                // Real-time factor (processing time / audio duration)
  let processingTime: Double     // Wall clock time in seconds
  let audioDuration: Double      // Audio duration in seconds
  let wordCount: Int             // Number of words in transcript
  let text: String               // Full transcript text
}

// MARK: - Text Similarity

/// Compute word-level similarity between two texts (case-insensitive, punctuation-stripped)
func computeTextSimilarity(_ text1: String, _ text2: String) -> Double {
  let punctuation = CharacterSet.punctuationCharacters.union(.symbols)
  
  func normalizeAndSplit(_ text: String) -> [String] {
    text.lowercased()
      .unicodeScalars.filter { !punctuation.contains($0) }
      .map { String($0) }.joined()
      .split(separator: " ")
      .map(String.init)
      .filter { !$0.isEmpty }
  }
  
  let words1 = normalizeAndSplit(text1)
  let words2 = normalizeAndSplit(text2)
  
  guard !words1.isEmpty && !words2.isEmpty else { return 0.0 }
  
  // Simple Jaccard similarity for quick validation
  let set1 = Set(words1)
  let set2 = Set(words2)
  let intersection = set1.intersection(set2)
  let union = set1.union(set2)
  
  return Double(intersection.count) / Double(union.count)
}

/// Compute word error rate approximation using Levenshtein distance on word sequences
func computeWordErrorRate(_ hypothesis: String, _ reference: String) -> Double {
  let punctuation = CharacterSet.punctuationCharacters.union(.symbols)
  
  func normalizeAndSplit(_ text: String) -> [String] {
    text.lowercased()
      .unicodeScalars.filter { !punctuation.contains($0) }
      .map { String($0) }.joined()
      .split(separator: " ")
      .map(String.init)
      .filter { !$0.isEmpty }
  }
  
  let hyp = normalizeAndSplit(hypothesis)
  let ref = normalizeAndSplit(reference)
  
  guard !ref.isEmpty else { return hyp.isEmpty ? 0.0 : 1.0 }
  
  // Levenshtein distance on word sequences
  var dp = [[Int]](repeating: [Int](repeating: 0, count: ref.count + 1), count: hyp.count + 1)
  
  for i in 0...hyp.count { dp[i][0] = i }
  for j in 0...ref.count { dp[0][j] = j }
  
  for i in 1...hyp.count {
    for j in 1...ref.count {
      if hyp[i-1] == ref[j-1] {
        dp[i][j] = dp[i-1][j-1]
      } else {
        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
      }
    }
  }
  
  return Double(dp[hyp.count][ref.count]) / Double(ref.count)
}

// MARK: - Whisper Benchmark Suite

@Suite(.serialized)
struct WhisperBenchmark {
  
  // MARK: - Configuration
  
  /// Path to benchmark fixtures (relative to package root)
  static let fixturesPath = "../benchmarks/fixtures"
  
  /// Model configuration for benchmarks
  static let modelSize: WhisperModelSize = .largeTurbo
  static let quantization: WhisperQuantization = .q4
  
  /// Number of benchmark runs for statistical significance
  static let numRuns = 3
  
  /// Quality thresholds
  static let minSimilarity = 0.85  // Minimum Jaccard similarity to baseline
  static let maxWER = 0.30         // Maximum word error rate vs baseline
  
  /// Performance thresholds (RTF = processing time / audio duration)
  static let pythonBaselineRTF = 0.0500  // Python mlx-whisper baseline on AMI ES2002a (21min)
  static let targetRTF = 0.035           // Target: 30% faster than Python (0.0500 * 0.7)
  static let maxRTF = 0.07               // Must be competitive with Python
  
  // MARK: - Fixture Loading
  
  /// Get URL for audio fixture and its duration
  static func getAudioFixture(_ filename: String) throws -> (url: URL, duration: Double) {
    let url = URL(fileURLWithPath: fixturesPath).appendingPathComponent(filename)
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw BenchmarkError.fixtureNotFound(filename)
    }
    
    let file = try AVAudioFile(forReading: url)
    let duration = Double(file.length) / file.processingFormat.sampleRate
    
    return (url, duration)
  }
  
  /// Load baseline transcript
  static func loadBaselineTranscript(_ filename: String) throws -> String {
    let url = URL(fileURLWithPath: fixturesPath).appendingPathComponent(filename)
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw BenchmarkError.fixtureNotFound(filename)
    }
    return try String(contentsOf: url, encoding: .utf8)
  }
  
  // MARK: - Long-Form Transcription Benchmark
  
  @Test @MainActor func longFormBenchmark() async throws {
    print(repeatString("=", 70))
    print("Whisper Long-Form Transcription Benchmark")
    print(repeatString("=", 70))
    
    configureMemoryLimits()
    
    // Load fixtures
    print("\nLoading fixtures...")
    let (audioURL, audioDuration) = try Self.getAudioFixture("ami_ES2002a_full.wav")
    let baselineText = try Self.loadBaselineTranscript("ami_ES2002a_baseline.txt")
    
    print("  Audio duration: \(String(format: "%.1f", audioDuration))s (\(String(format: "%.1f", audioDuration / 60)) minutes)")
    print("  Baseline words: \(baselineText.split(separator: " ").count)")
    
    // Load model
    print("\nLoading model: \(Self.modelSize.rawValue) [\(Self.quantization.rawValue)]...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let engine = STT.whisper(model: Self.modelSize, quantization: Self.quantization)
    try await engine.load()
    let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
    print("  Model load time: \(String(format: "%.2f", loadTime))s")
    
    // Warmup run
    print("\nWarmup run...")
    let warmupStart = CFAbsoluteTimeGetCurrent()
    _ = try await engine.transcribe(audioURL, language: .english, temperature: 0.0, timestamps: .segment)
    let warmupTime = CFAbsoluteTimeGetCurrent() - warmupStart
    print("  Warmup RTF: \(String(format: "%.3f", warmupTime / audioDuration))")
    clearMemoryBetweenRuns()
    
    // Benchmark runs
    print("\nBenchmark runs (\(Self.numRuns) iterations)...")
    var results: [BenchmarkResult] = []
    
    for run in 1...Self.numRuns {
      let startTime = CFAbsoluteTimeGetCurrent()
      
      let result = try await engine.transcribe(audioURL, language: .english, temperature: 0.0, timestamps: .segment)
      
      let processingTime = CFAbsoluteTimeGetCurrent() - startTime
      let rtf = processingTime / audioDuration
      
      let benchResult = BenchmarkResult(
        rtf: rtf,
        processingTime: processingTime,
        audioDuration: audioDuration,
        wordCount: result.text.split(separator: " ").count,
        text: result.text
      )
      results.append(benchResult)
      
      print("  Run \(run): RTF=\(String(format: "%.4f", rtf)), time=\(String(format: "%.1f", processingTime))s, words=\(benchResult.wordCount)")
      
      clearMemoryBetweenRuns()
    }
    
    // Compute statistics
    let rtfs = results.map { $0.rtf }
    let avgRTF = rtfs.reduce(0, +) / Double(rtfs.count)
    let minRTF = rtfs.min()!
    let maxRTFResult = rtfs.max()!
    
    print("\n" + repeatString("-", 70))
    print("Performance Results")
    print(repeatString("-", 70))
    print("  Average RTF: \(String(format: "%.4f", avgRTF)) (target: <\(Self.targetRTF))")
    print("  Min RTF:     \(String(format: "%.4f", minRTF))")
    print("  Max RTF:     \(String(format: "%.4f", maxRTFResult))")
    print("  Threshold:   \(String(format: "%.4f", Self.maxRTF)) (must be below)")
    
    // Quality validation
    let lastResult = results.last!
    let similarity = computeTextSimilarity(lastResult.text, baselineText)
    let wer = computeWordErrorRate(lastResult.text, baselineText)
    
    print("\n" + repeatString("-", 70))
    print("Quality Results")
    print(repeatString("-", 70))
    print("  Jaccard Similarity: \(String(format: "%.2f%%", similarity * 100)) (min: \(Self.minSimilarity * 100)%)")
    print("  Word Error Rate:    \(String(format: "%.2f%%", wer * 100)) (max: \(Self.maxWER * 100)%)")
    
    // Assertions
    #expect(avgRTF < Self.maxRTF, "RTF \(avgRTF) exceeds threshold \(Self.maxRTF)")
    #expect(similarity >= Self.minSimilarity, "Similarity \(similarity) below threshold \(Self.minSimilarity)")
    #expect(wer <= Self.maxWER, "WER \(wer) exceeds threshold \(Self.maxWER)")
    
    // Summary
    print("\n" + repeatString("=", 70))
    print("BENCHMARK SUMMARY")
    print(repeatString("=", 70))
    print("  Python baseline RTF: \(String(format: "%.4f", Self.pythonBaselineRTF))")
    print("  Swift target RTF:    \(String(format: "%.4f", Self.targetRTF)) (30% faster than Python)")
    print("  Swift actual RTF:    \(String(format: "%.4f", avgRTF))")
    
    if avgRTF <= Self.targetRTF {
      let speedup = (Self.pythonBaselineRTF - avgRTF) / Self.pythonBaselineRTF * 100
      print("✅ Swift target MET: \(String(format: "%.1f%%", speedup)) faster than Python")
    } else if avgRTF <= Self.pythonBaselineRTF {
      let speedup = (Self.pythonBaselineRTF - avgRTF) / Self.pythonBaselineRTF * 100
      let needed = (avgRTF - Self.targetRTF) / avgRTF * 100
      print("⚠️  Faster than Python (\(String(format: "%.1f%%", speedup))) but need \(String(format: "%.1f%%", needed)) more for target")
    } else {
      let slower = (avgRTF - Self.pythonBaselineRTF) / Self.pythonBaselineRTF * 100
      print("❌ SLOWER than Python by \(String(format: "%.1f%%", slower))")
    }
    
    if similarity >= Self.minSimilarity && wer <= Self.maxWER {
      print("✅ Quality validation PASSED")
    } else {
      print("❌ Quality validation FAILED")
    }
    
    MLXMemory.logStats(prefix: "Final")
  }
  
  // MARK: - Quick Sanity Check (5-minute segment)
  
  @Test @MainActor func quickSanityCheck() async throws {
    print(repeatString("=", 70))
    print("Whisper Quick Sanity Check (first 5 minutes)")
    print(repeatString("=", 70))
    
    configureMemoryLimits()
    
    // Load 5-minute fixture
    let (audioURL, audioDuration) = try Self.getAudioFixture("ami_ES2002a_5min.wav")
    
    print("  Segment duration: \(String(format: "%.1f", audioDuration))s")
    
    // Load model
    let engine = STT.whisper(model: Self.modelSize, quantization: Self.quantization)
    try await engine.load()
    
    // Single run
    let startTime = CFAbsoluteTimeGetCurrent()
    let result = try await engine.transcribe(audioURL, language: .english, temperature: 0.0, timestamps: .segment)
    let processingTime = CFAbsoluteTimeGetCurrent() - startTime
    let rtf = processingTime / audioDuration
    
    print("\n  RTF: \(String(format: "%.4f", rtf))")
    print("  Processing time: \(String(format: "%.1f", processingTime))s")
    print("  Words: \(result.text.split(separator: " ").count)")
    print("  Preview: \(String(result.text.prefix(200)))...")
    
    // Sanity check - should produce non-empty output
    #expect(!result.text.isEmpty, "Transcription should not be empty")
    #expect(result.text.split(separator: " ").count > 100, "Should have significant content")
    #expect(rtf < 0.5, "RTF should be reasonable (< 0.5)")
  }
}

// MARK: - Errors

enum BenchmarkError: Error, CustomStringConvertible {
  case fixtureNotFound(String)
  case audioLoadFailed(String)
  
  var description: String {
    switch self {
    case .fixtureNotFound(let name):
      return "Benchmark fixture not found: \(name)"
    case .audioLoadFailed(let name):
      return "Failed to load audio: \(name)"
    }
  }
}

// MARK: - Helpers

private func repeatString(_ s: String, _ count: Int) -> String {
  String(repeating: s, count: count)
}
