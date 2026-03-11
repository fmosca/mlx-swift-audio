import AVFoundation
import Foundation
import FluidAudio
import MLXAudio
import Qwen3ASR
import Wav2Vec2Aligner

@main
struct WhisperBenchmarkCLI {
  static let fixturesPath = "benchmarks/fixtures"
  static let modelSize: WhisperModelSize = .largeTurbo
  static let quantization: WhisperQuantization = .q4

  // Measured sequential baseline (clean GPU, sequential runs, Mar 2026)
  static let pythonBaselineRTF = 0.0597
  static let swiftSequentialRTF = 0.0627

  static func main() async {
    let args = CommandLine.arguments
    let mode = args.count > 1 ? args[1] : "help"

    let separator = String(repeating: "=", count: 70)
    print(separator)
    print("Whisper Benchmark CLI — \(mode.uppercased())")
    print(separator)

    do {
      switch mode {
      case "quick":
        try await runSequential(audioFile: "ami_ES2002a_5min.wav", runs: 1)
      case "full":
        try await runSequential(audioFile: "ami_ES2002a_full.wav", runs: 3)
      case "batch-quick":
        try await runBatched(audioFile: "ami_ES2002a_5min.wav", runs: 1, batchSize: 4)
      case "batch-full":
        try await runBatched(audioFile: "ami_ES2002a_full.wav", runs: 3, batchSize: 4)
      case "compare":
        try await runComparison(audioFile: "ami_ES2002a_5min.wav")
      case "compare-full":
        try await runComparison(audioFile: "ami_ES2002a_full.wav")
      case "vad-scan":
        let dir = args.count > 2 ? args[2] : "/Users/francesco.mosca/Documents/meeting-transcriber-v2"
        let maxFiles = args.count > 3 ? (Int(args[3]) ?? 10) : 10
        let scanSecs = args.count > 4 ? (Double(args[4]) ?? 300) : 300
        try await runVADScan(meetingsDir: dir, maxFiles: maxFiles, scanDurationSeconds: scanSecs)
      case "word-compare":
        let audioFile = args.count > 2 ? args[2] : "ami_ES2002a_5min.wav"
        let baselineFile = args.count > 3 ? args[3] : nil
        try await runWordCompare(audioFile: audioFile, baselineFile: baselineFile)
      case "word-compare-full":
        let audioFile = args.count > 2 ? args[2] : "ami_ES2002a_full.wav"
        let baselineFile = args.count > 3 ? args[3] : nil
        try await runWordCompare(audioFile: audioFile, baselineFile: baselineFile)
      case "parakeet-compare":
        let audioFile = args.count > 2 ? args[2] : "amicorpus-ES2002a.mp3"
        let baselineFile = args.count > 3 ? args[3] : nil
        try await runParakeetCompare(audioFile: audioFile, baselineFile: baselineFile)
      case "qwen3-align":
        let audioFile = args.count > 2 ? args[2] : "ami_ES2002a_5min.wav"
        let baselineFile = args.count > 3 ? args[3] : nil
        try await runQwen3AlignCompare(audioFile: audioFile, baselineFile: baselineFile)
      case "wav2vec2-align":
        let audioFile = args.count > 2 ? args[2] : "ami_ES2002a_5min.wav"
        let baselineFile = args.count > 3 ? args[3] : nil
        try await runWav2Vec2AlignCompare(audioFile: audioFile, baselineFile: baselineFile)
      case "wav2vec2-align-full":
        let audioFile = args.count > 2 ? args[2] : "ami_ES2002a_full.wav"
        let baselineFile = args.count > 3 ? args[3] : nil
        try await runWav2Vec2AlignCompare(audioFile: audioFile, baselineFile: baselineFile)
      default:
        printUsage()
      }
    } catch {
      print("Error: \(error)")
    }
  }

  static func printUsage() {
    print("""
    Usage: WhisperBenchmark <mode> [args]

      quick         Sequential, 5-min audio, 1 run (quick sanity check)
      full          Sequential, 21-min audio, 3 runs (stable baseline)
      batch-quick   Batched (B=4), 5-min audio, 1 run
      batch-full    Batched (B=4), 21-min audio, 3 runs
      compare       Sequential vs batched side-by-side, 5-min audio
      compare-full  Sequential vs batched side-by-side, 21-min audio
      vad-scan [dir] [max] [secs]
                    Run Silero VAD over real meeting sessions and report
                    segment-duration distribution. Non-destructive.
                    dir:  meetings root     (default: meeting-transcriber-v2)
                    max:  files to scan     (default: 10)
                    secs: seconds per file  (default: 300 = first 5 min)
      word-compare [audio] [baseline_json]
                    Compare Swift batched word timestamps against Python baseline.
                    audio:         fixture filename  (default: ami_ES2002a_5min.wav)
                    baseline_json: path to JSON from word_baseline.py
                                   (default: fixtures/<audio_stem>_word_baseline.json)
                    Generate baseline first:
                      python benchmarks/scripts/word_baseline.py benchmarks/fixtures/<audio>
      word-compare-full [audio] [baseline_json]
                    Same as word-compare using the full 21-min fixture.
      parakeet-compare [audio] [baseline_json]
                    Compare FluidAudio Parakeet-TDT (native timestamps) against
                    Python Whisper baseline. Tests whether Parakeet can replace
                    Whisper's cross-attention DTW for word alignment.
      qwen3-align [audio] [baseline_json]
                    Compare Qwen3-ForcedAligner (dedicated alignment model) against
                    Python Whisper baseline. Uses Whisper for transcription, then
                    Qwen3 for alignment. Expected to be faster and more accurate
                    than DTW-based approaches.
      wav2vec2-align [audio] [baseline_json]
                    Compare wav2vec2 CTC forced aligner (MLX-native) against Python
                    Whisper baseline. Uses Whisper for transcription, then wav2vec2
                    (MLX) for word-level alignment via CTC forced alignment.
      wav2vec2-align-full [audio] [baseline_json]
                    Same as wav2vec2-align using the full 21-min fixture.
    """)
  }

  // MARK: - Sequential path

  @MainActor
  static func runSequential(audioFile: String, runs: Int) async throws {
    let (engine, audioURL, audioDuration) = try await loadEngine(audioFile: audioFile)

    print("\nWarmup (sequential)...")
    let _ = try await engine.transcribe(audioURL, language: .english, temperature: 0.0, timestamps: .segment)

    print("Benchmark runs: \(runs)")
    var rtfs: [Double] = []
    for run in 1...runs {
      let t0 = CFAbsoluteTimeGetCurrent()
      let result = try await engine.transcribe(audioURL, language: .english, temperature: 0.0, timestamps: .segment)
      let elapsed = CFAbsoluteTimeGetCurrent() - t0
      let rtf = elapsed / audioDuration
      rtfs.append(rtf)
      let words = result.text.split(separator: " ").count
      print("  Run \(run): RTF=\(fmt4(rtf))  time=\(fmt1(elapsed))s  words=\(words)")
    }

    printRTFSummary(label: "Sequential", rtfs: rtfs, audioDuration: audioDuration)
    printComparison(avgRTF: rtfs.reduce(0, +) / Double(rtfs.count), label: "Sequential")
  }

  // MARK: - Batched path

  @MainActor
  static func runBatched(audioFile: String, runs: Int, batchSize: Int) async throws {
    let (engine, audioURL, audioDuration) = try await loadEngine(audioFile: audioFile)

    print("\nWarmup (batched B=\(batchSize))...")
    let warmupResult = try await engine.transcribeBatched(audioURL, language: .english, batchSize: batchSize)
    print("  Warmup RTF=\(fmt4(warmupResult.processingTime / audioDuration))  words=\(warmupResult.text.split(separator: " ").count)")

    print("Benchmark runs: \(runs)  batchSize=\(batchSize)")
    var rtfs: [Double] = []
    for run in 1...runs {
      let t0 = CFAbsoluteTimeGetCurrent()
      let result = try await engine.transcribeBatched(audioURL, language: .english, batchSize: batchSize)
      let elapsed = CFAbsoluteTimeGetCurrent() - t0
      let rtf = elapsed / audioDuration
      rtfs.append(rtf)
      let words = result.text.split(separator: " ").count
      print("  Run \(run): RTF=\(fmt4(rtf))  time=\(fmt1(elapsed))s  words=\(words)")
    }

    printRTFSummary(label: "Batched B=\(batchSize)", rtfs: rtfs, audioDuration: audioDuration)
    printComparison(avgRTF: rtfs.reduce(0, +) / Double(rtfs.count), label: "Batched B=\(batchSize)")
  }

  // MARK: - Side-by-side comparison

  @MainActor
  static func runComparison(audioFile: String) async throws {
    let (engine, audioURL, audioDuration) = try await loadEngine(audioFile: audioFile)

    print("\n--- Sequential (warmup + 2 runs) ---")
    _ = try await engine.transcribe(audioURL, language: .english, temperature: 0.0, timestamps: .segment)
    var seqRTFs: [Double] = []
    var seqResult: TranscriptionResult!
    for _ in 1...2 {
      let t0 = CFAbsoluteTimeGetCurrent()
      seqResult = try await engine.transcribe(audioURL, language: .english, temperature: 0.0, timestamps: .segment)
      let elapsed = CFAbsoluteTimeGetCurrent() - t0
      seqRTFs.append(elapsed / audioDuration)
    }
    let seqAvgRTF = seqRTFs.reduce(0, +) / Double(seqRTFs.count)

    print("\n--- Batched B=4 (warmup + 2 runs) ---")
    _ = try await engine.transcribeBatched(audioURL, language: .english, batchSize: 4)
    var batchRTFs: [Double] = []
    var batchResult: TranscriptionResult!
    for _ in 1...2 {
      let t0 = CFAbsoluteTimeGetCurrent()
      batchResult = try await engine.transcribeBatched(audioURL, language: .english, batchSize: 4)
      let elapsed = CFAbsoluteTimeGetCurrent() - t0
      batchRTFs.append(elapsed / audioDuration)
    }
    let batchAvgRTF = batchRTFs.reduce(0, +) / Double(batchRTFs.count)

    // Quality: word-level Jaccard similarity between outputs
    let seqWords = Set(seqResult.text.lowercased().split(separator: " ").map(String.init))
    let batchWords = Set(batchResult.text.lowercased().split(separator: " ").map(String.init))
    let intersection = seqWords.intersection(batchWords).count
    let union = seqWords.union(batchWords).count
    let jaccard = union > 0 ? Double(intersection) / Double(union) : 0.0

    let separator = String(repeating: "=", count: 70)
    print("\n\(separator)")
    print("COMPARISON RESULTS  [\(audioFile)]")
    print(separator)
    print("  Sequential avg RTF:  \(fmt4(seqAvgRTF))  (\(seqResult.text.split(separator: " ").count) words)")
    print("  Batched B=4 avg RTF: \(fmt4(batchAvgRTF))  (\(batchResult.text.split(separator: " ").count) words)")
    let speedup = seqAvgRTF / batchAvgRTF
    print("  Speedup:             \(String(format: "%.2f", speedup))×")
    print("  Word Jaccard:        \(String(format: "%.3f", jaccard))  (1.0 = identical vocabulary)")
    print("")
    print("  Python baseline RTF: \(fmt4(pythonBaselineRTF))")
    print("  Swift sequential:    \(fmt4(swiftSequentialRTF))  (measured)")
    print("  Batch vs Python:     \(String(format: "%.1f", (pythonBaselineRTF - batchAvgRTF) / pythonBaselineRTF * 100))% faster")

    let targetRTF = pythonBaselineRTF * 0.7
    if batchAvgRTF <= targetRTF {
      print("\n✅ 30% TARGET MET (RTF \(fmt4(batchAvgRTF)) ≤ \(fmt4(targetRTF)))")
    } else {
      let gap = (batchAvgRTF - targetRTF) / batchAvgRTF * 100
      print("\n⚠️  Target not yet met — need \(String(format: "%.1f", gap))% more (target: \(fmt4(targetRTF)))")
    }

    print("\n--- Sequential output (first 300 chars) ---")
    print(String(seqResult.text.prefix(300)))
    print("\n--- Batched output (first 300 chars) ---")
    print(String(batchResult.text.prefix(300)))
  }

  // MARK: - Shared helpers

  @MainActor
  static func loadEngine(audioFile: String) async throws -> (WhisperEngine, URL, Double) {
    let audioURL = URL(fileURLWithPath: fixturesPath).appendingPathComponent(audioFile)
    guard FileManager.default.fileExists(atPath: audioURL.path) else {
      throw BenchmarkError.fixtureNotFound(audioURL.path)
    }

    let audioDuration = try getAudioDuration(audioURL)
    print("\nAudio: \(audioFile)  [\(fmt1(audioDuration))s / \(String(format: "%.1f", audioDuration / 60)) min]")

    print("Loading \(modelSize.rawValue) [\(quantization.rawValue)]...")
    let t0 = CFAbsoluteTimeGetCurrent()
    let engine = STT.whisper(model: modelSize, quantization: quantization)
    try await engine.load()
    print("Load time: \(fmt2(CFAbsoluteTimeGetCurrent() - t0))s")

    return (engine, audioURL, audioDuration)
  }

  static func printRTFSummary(label: String, rtfs: [Double], audioDuration: Double) {
    let avg = rtfs.reduce(0, +) / Double(rtfs.count)
    let separator = String(repeating: "-", count: 70)
    print("\n\(separator)")
    print("RESULTS  [\(label)]")
    print("\(separator)")
    print("  Average RTF: \(fmt4(avg))")
    if rtfs.count > 1 {
      print("  Min RTF:     \(fmt4(rtfs.min()!))")
      print("  Max RTF:     \(fmt4(rtfs.max()!))")
    }
  }

  static func printComparison(avgRTF: Double, label: String) {
    let targetRTF = pythonBaselineRTF * 0.7
    let vsPython = (pythonBaselineRTF - avgRTF) / pythonBaselineRTF * 100
    print("\n  Python baseline: \(fmt4(pythonBaselineRTF))")
    print("  Swift target:    \(fmt4(targetRTF)) (30% faster than Python)")
    print("  \(label): \(fmt4(avgRTF))")
    if avgRTF <= targetRTF {
      print("\n✅ TARGET MET: \(String(format: "%.1f", vsPython))% faster than Python")
    } else if avgRTF <= pythonBaselineRTF {
      let gap = (avgRTF - targetRTF) / avgRTF * 100
      print("\n⚠️  \(String(format: "%.1f", vsPython))% faster than Python, need \(String(format: "%.1f", gap))% more for target")
    } else {
      print("\n❌ SLOWER than Python by \(String(format: "%.1f", -vsPython))%")
    }
  }

  static func getAudioDuration(_ url: URL) throws -> Double {
    let file = try AVAudioFile(forReading: url)
    return Double(file.length) / file.processingFormat.sampleRate
  }

  // MARK: - Word timestamp comparison

  struct PythonWord: Decodable {
    let word: String
    let start: Double
    let end: Double
    let probability: Double
  }

  /// Compare Swift batched word timestamps against a Python mlx-whisper baseline.
  ///
  /// Metrics reported:
  ///   - Word text Jaccard similarity (vocabulary overlap, ignoring order/timing)
  ///   - Match rate: fraction of Python words found in Swift output (case-insensitive)
  ///   - Start-time MAE: mean |swift_start − python_start| for matched words
  ///   - End-time MAE: mean |swift_end − python_end| for matched words
  ///
  /// The timing MAE is the most informative metric: < 0.1 s is good,
  /// < 0.2 s is acceptable, > 0.5 s suggests alignment is off.
  @MainActor
  static func runWordCompare(audioFile: String, baselineFile: String?) async throws {
    let (engine, audioURL, audioDuration) = try await loadEngine(audioFile: audioFile)

    // Locate baseline JSON
    let stem = audioURL.deletingPathExtension().lastPathComponent
    let baselineURL: URL
    if let bf = baselineFile {
      baselineURL = URL(fileURLWithPath: bf)
    } else {
      baselineURL = URL(fileURLWithPath: fixturesPath).appendingPathComponent("\(stem)_word_baseline.json")
    }
    guard FileManager.default.fileExists(atPath: baselineURL.path) else {
      print("""
        Baseline not found: \(baselineURL.path)

        Generate it first:
          python benchmarks/scripts/word_baseline.py \(audioURL.path)
        """)
      throw BenchmarkError.fixtureNotFound(baselineURL.path)
    }

    let pythonWords = try JSONDecoder().decode([PythonWord].self, from: Data(contentsOf: baselineURL))
    print("Python baseline: \(pythonWords.count) words loaded from \(baselineURL.lastPathComponent)")

    // Run Swift batched with word timestamps (warmup then timed)
    print("\nWarmup (batched, word timestamps)…")
    _ = try await engine.transcribeBatched(audioURL, language: .english, timestamps: .word, batchSize: 4)

    print("Timed run…")
    let t0 = CFAbsoluteTimeGetCurrent()
    let result = try await engine.transcribeBatched(audioURL, language: .english, timestamps: .word, batchSize: 4)
    let elapsed = CFAbsoluteTimeGetCurrent() - t0
    let rtf = elapsed / audioDuration

    let swiftWords: [(word: String, start: Double, end: Double)] = result.segments.flatMap { seg in
      (seg.words ?? []).map { ($0.word.trimmingCharacters(in: .whitespaces), Double($0.start), Double($0.end)) }
    }
    print("Swift output:    \(swiftWords.count) words  RTF=\(fmt4(rtf))")

    // ── Text quality: word Jaccard ──────────────────────────────────────────
    let pySet = Set(pythonWords.map { $0.word.lowercased() })
    let swSet = Set(swiftWords.map { $0.word.lowercased() })
    let intersection = pySet.intersection(swSet).count
    let union = pySet.union(swSet).count
    let jaccard = union > 0 ? Double(intersection) / Double(union) : 0.0

    // ── Timing accuracy: one-to-one greedy matching with 1.5 s search window ──
    // Build a mutable index: lowercase word → [(start, end, used)].
    // The 1.5 s window prevents matching words that appear in both transcripts
    // but at completely different positions (e.g. common words like "I", "the").
    struct SwiftWordEntry { var start, end: Double; var used: Bool }
    var swiftIndex: [String: [SwiftWordEntry]] = [:]
    for w in swiftWords {
      let key = w.word.lowercased()
      swiftIndex[key, default: []].append(SwiftWordEntry(start: w.start, end: w.end, used: false))
    }

    var signedStartErrors: [Double] = []  // signed for bias analysis
    var absEndErrors: [Double] = []
    var matchedPairs: [(py: PythonWord, swStart: Double, swEnd: Double)] = []

    for pw in pythonWords {
      let key = pw.word.lowercased()
      guard var candidates = swiftIndex[key] else { continue }
      // Find nearest unused candidate within the search window.
      var bestIdx: Int? = nil
      var bestDist = 1.5  // seconds — spurious matches beyond this are excluded
      for (i, c) in candidates.enumerated() where !c.used {
        let d = abs(c.start - pw.start)
        if d < bestDist { bestDist = d; bestIdx = i }
      }
      guard let idx = bestIdx else { continue }
      candidates[idx].used = true
      swiftIndex[key] = candidates
      signedStartErrors.append(candidates[idx].start - pw.start)
      absEndErrors.append(abs(candidates[idx].end - pw.end))
      matchedPairs.append((pw, candidates[idx].start, candidates[idx].end))
    }

    let matchRate = pythonWords.isEmpty ? 0.0 : Double(signedStartErrors.count) / Double(pythonWords.count)
    let startMAE = signedStartErrors.isEmpty ? 0.0 : signedStartErrors.map(abs).reduce(0, +) / Double(signedStartErrors.count)
    let endMAE = absEndErrors.isEmpty ? 0.0 : absEndErrors.reduce(0, +) / Double(absEndErrors.count)
    let startBias = signedStartErrors.isEmpty ? 0.0 : signedStartErrors.reduce(0, +) / Double(signedStartErrors.count)
    let sortedAbsStart = signedStartErrors.map(abs).sorted()
    let startP90 = sortedAbsStart.isEmpty ? 0.0 : sortedAbsStart[Int(Double(sortedAbsStart.count) * 0.9)]

    // ── Report ──────────────────────────────────────────────────────────────
    let sep = String(repeating: "=", count: 70)
    print("\n\(sep)")
    print("WORD TIMESTAMP COMPARISON  [\(audioFile)]")
    print(sep)
    print("  RTF (Swift batched, word timestamps): \(fmt4(rtf))")
    print("")
    print("  Text quality")
    print(String(format: "    Python words:   %ld", pythonWords.count))
    print(String(format: "    Swift words:    %ld", swiftWords.count))
    print(String(format: "    Word Jaccard:   %.3f  (1.0 = identical vocabulary)", jaccard))
    print("")
    print("  Timing accuracy  (\(signedStartErrors.count) matched word pairs, ±1.5 s window)")
    print(String(format: "    Match rate:     %.1f%%", matchRate * 100))
    print(String(format: "    Start MAE:      %.3f s", startMAE))
    print(String(format: "    End MAE:        %.3f s", endMAE))
    print(String(format: "    Start p90 err:  %.3f s", startP90))
    print(String(format: "    Start bias:     %+.3f s  (%@)",
                 startBias, (startBias > 0.01 ? "Swift runs late" : startBias < -0.01 ? "Swift runs early" : "no systematic offset") as NSString))

    let timingGrade: String
    if startMAE < 0.05 { timingGrade = "excellent (< 50 ms)" }
    else if startMAE < 0.10 { timingGrade = "good (< 100 ms)" }
    else if startMAE < 0.20 { timingGrade = "acceptable (< 200 ms)" }
    else { timingGrade = "poor (> 200 ms) — check alignment head config" }
    print("    Assessment:     \(timingGrade)")

    // ── Sample output side-by-side (first 10 matched pairs) ────────────────
    print("\n  First 10 matched word pairs (python → swift):")
    for (pw, swStart, swEnd) in matchedPairs.prefix(10) {
      print(String(format: "    %-20@  py [%.2f–%.2f]  sw [%.2f–%.2f]  Δ%+.3fs",
                   pw.word as NSString, pw.start, pw.end, swStart, swEnd,
                   swStart - pw.start))
    }
  }

  // MARK: - Parakeet comparison

  /// Compare FluidAudio Parakeet-TDT (native timestamps) against Python Whisper baseline.
  /// Tests whether Parakeet's TDT-based timestamps are better than Whisper's DTW.
  static func runParakeetCompare(audioFile: String, baselineFile: String?) async throws {
    let audioURL = URL(fileURLWithPath: fixturesPath).appendingPathComponent(audioFile)
    guard FileManager.default.fileExists(atPath: audioURL.path) else {
      throw BenchmarkError.fixtureNotFound(audioURL.path)
    }

    let audioDuration = try getAudioDuration(audioURL)
    print("\nAudio: \(audioFile)  [\(fmt1(audioDuration))s / \(String(format: "%.1f", audioDuration / 60)) min]")

    // Locate baseline JSON
    let stem = audioURL.deletingPathExtension().lastPathComponent
    let baselineURL: URL
    if let bf = baselineFile {
      baselineURL = URL(fileURLWithPath: bf)
    } else {
      baselineURL = URL(fileURLWithPath: fixturesPath).appendingPathComponent("\(stem)_word_baseline.json")
    }
    guard FileManager.default.fileExists(atPath: baselineURL.path) else {
      print("""
        Baseline not found: \(baselineURL.path)

        Generate it first:
          python benchmarks/scripts/word_baseline.py \(audioURL.path)
        """)
      throw BenchmarkError.fixtureNotFound(baselineURL.path)
    }

    let pythonWords = try JSONDecoder().decode([PythonWord].self, from: Data(contentsOf: baselineURL))
    print("Python baseline: \(pythonWords.count) words loaded from \(baselineURL.lastPathComponent)")

    // Load audio as 16kHz mono Float32
    print("\nLoading audio for Parakeet...")
    let samples = try load16kMono(url: audioURL, maxSeconds: nil)
    print("Loaded \(samples.count) samples (\(fmt1(Double(samples.count) / 16000))s)")

    // Initialize Parakeet
    print("\nInitializing Parakeet-TDT (CoreML)...")
    let asrModels = try await AsrModels.loadFromCache()
    let asrManager = AsrManager()
    try await asrManager.initialize(models: asrModels)
    print("Parakeet ready.")

    // Transcribe
    print("\nTranscribing with Parakeet-TDT...")
    let t0 = CFAbsoluteTimeGetCurrent()
    let result = try await asrManager.transcribe(samples, source: .system)
    let elapsed = CFAbsoluteTimeGetCurrent() - t0
    let rtf = elapsed / audioDuration

    // Extract word timings from token timings
    let tokenTimings = result.tokenTimings ?? []
    let parakeetWords: [(word: String, start: Double, end: Double)] = tokenTimings.map {
      ($0.token.trimmingCharacters(in: .whitespaces), $0.startTime, $0.endTime)
    }.filter { !$0.word.isEmpty }

    print("Parakeet output: \(parakeetWords.count) tokens  RTF=\(fmt4(rtf))")
    print("Text preview: \(result.text.prefix(200))...")

    // ── Text quality: word Jaccard ──────────────────────────────────────────
    let pySet = Set(pythonWords.map { $0.word.lowercased() })
    let pkSet = Set(parakeetWords.map { $0.word.lowercased() })
    let intersection = pySet.intersection(pkSet).count
    let union = pySet.union(pkSet).count
    let jaccard = union > 0 ? Double(intersection) / Double(union) : 0.0

    // ── Timing accuracy: one-to-one greedy matching ──
    struct ParakeetEntry { var start, end: Double; var used: Bool }
    var parakeetIndex: [String: [ParakeetEntry]] = [:]
    for w in parakeetWords {
      let key = w.word.lowercased()
      parakeetIndex[key, default: []].append(ParakeetEntry(start: w.start, end: w.end, used: false))
    }

    var signedStartErrors: [Double] = []
    var absEndErrors: [Double] = []

    for pw in pythonWords {
      let key = pw.word.lowercased()
      guard var candidates = parakeetIndex[key] else { continue }
      var bestIdx: Int? = nil
      var bestDist = 1.5
      for (i, c) in candidates.enumerated() where !c.used {
        let d = abs(c.start - pw.start)
        if d < bestDist { bestDist = d; bestIdx = i }
      }
      guard let idx = bestIdx else { continue }
      candidates[idx].used = true
      parakeetIndex[key] = candidates
      signedStartErrors.append(candidates[idx].start - pw.start)
      absEndErrors.append(abs(candidates[idx].end - pw.end))
    }

    let matchRate = pythonWords.isEmpty ? 0.0 : Double(signedStartErrors.count) / Double(pythonWords.count)
    let startMAE = signedStartErrors.isEmpty ? 0.0 : signedStartErrors.map(abs).reduce(0, +) / Double(signedStartErrors.count)
    let endMAE = absEndErrors.isEmpty ? 0.0 : absEndErrors.reduce(0, +) / Double(absEndErrors.count)
    let startBias = signedStartErrors.isEmpty ? 0.0 : signedStartErrors.reduce(0, +) / Double(signedStartErrors.count)
    let sortedAbsStart = signedStartErrors.map(abs).sorted()
    let startP90 = sortedAbsStart.isEmpty ? 0.0 : sortedAbsStart[Int(Double(sortedAbsStart.count) * 0.9)]

    // ── Report ──────────────────────────────────────────────────────────────
    let sep = String(repeating: "=", count: 70)
    print("\n\(sep)")
    print("PARAKEET-TDT vs PYTHON WHISPER  [\(audioFile)]")
    print(sep)
    print("  RTF (Parakeet-TDT): \(fmt4(rtf))")
    print("")
    print("  Text quality")
    print(String(format: "    Python words:     %ld", pythonWords.count))
    print(String(format: "    Parakeet tokens:  %ld", parakeetWords.count))
    print(String(format: "    Word Jaccard:     %.3f  (1.0 = identical vocabulary)", jaccard))
    print("")
    print("  Timing accuracy  (\(signedStartErrors.count) matched pairs, ±1.5 s window)")
    print(String(format: "    Match rate:       %.1f%%", matchRate * 100))
    print(String(format: "    Start MAE:        %.3f s", startMAE))
    print(String(format: "    End MAE:          %.3f s", endMAE))
    print(String(format: "    Start p90 err:    %.3f s", startP90))
    print(String(format: "    Start bias:       %+.3f s", startBias))

    let timingGrade: String
    if startMAE < 0.05 { timingGrade = "excellent (< 50 ms)" }
    else if startMAE < 0.10 { timingGrade = "good (< 100 ms)" }
    else if startMAE < 0.20 { timingGrade = "acceptable (< 200 ms)" }
    else { timingGrade = "poor (> 200 ms)" }
    print("    Assessment:       \(timingGrade)")

    print("\n  Parakeet text (first 500 chars):")
    print("    \(result.text.prefix(500))")
  }

  // MARK: - Qwen3 ForcedAligner comparison

  /// Compare Qwen3-ForcedAligner (dedicated alignment model) against Python Whisper baseline.
  /// Uses Whisper for transcription, then Qwen3 for word-level alignment.
  static func runQwen3AlignCompare(audioFile: String, baselineFile: String?) async throws {
    let audioURL = URL(fileURLWithPath: fixturesPath).appendingPathComponent(audioFile)
    guard FileManager.default.fileExists(atPath: audioURL.path) else {
      throw BenchmarkError.fixtureNotFound(audioURL.path)
    }

    let audioDuration = try getAudioDuration(audioURL)
    print("\nAudio: \(audioFile)  [\(fmt1(audioDuration))s / \(String(format: "%.1f", audioDuration / 60)) min]")

    // Locate baseline JSON
    let stem = audioURL.deletingPathExtension().lastPathComponent
    let baselineURL: URL
    if let bf = baselineFile {
      baselineURL = URL(fileURLWithPath: bf)
    } else {
      baselineURL = URL(fileURLWithPath: fixturesPath).appendingPathComponent("\(stem)_word_baseline.json")
    }
    guard FileManager.default.fileExists(atPath: baselineURL.path) else {
      print("""
        Baseline not found: \(baselineURL.path)

        Generate it first:
          python benchmarks/scripts/word_baseline.py \(audioURL.path)
        """)
      throw BenchmarkError.fixtureNotFound(baselineURL.path)
    }

    let pythonWords = try JSONDecoder().decode([PythonWord].self, from: Data(contentsOf: baselineURL))
    print("Python baseline: \(pythonWords.count) words loaded from \(baselineURL.lastPathComponent)")

    // Step 1: Transcribe with Whisper (batched, no word timestamps)
    print("\n[Step 1] Transcribing with Whisper (batched)...")
    let (whisperEngine, _, _) = try await loadEngine(audioFile: audioFile)
    let t0 = CFAbsoluteTimeGetCurrent()
    let transcription = try await whisperEngine.transcribeBatched(
      audioURL,
      timestamps: .none,
      batchSize: 4
    )
    let whisperTime = CFAbsoluteTimeGetCurrent() - t0
    let whisperRTF = whisperTime / audioDuration
    print("Whisper transcription: \(fmt1(whisperTime))s  RTF=\(fmt4(whisperRTF))")

    // Extract text from segments
    let whisperText = transcription.segments.map { $0.text.trimmingCharacters(in: CharacterSet.whitespaces) }.joined(separator: " ")
    print("Whisper text (\(whisperText.count) chars): \(whisperText.prefix(200))...")

    // Load audio for Qwen3 (16kHz mono)
    print("\nLoading audio for alignment...")
    let samples16k = try load16kMono(url: audioURL, maxSeconds: nil)
    print("Loaded \(samples16k.count) samples (\(fmt1(Double(samples16k.count) / 16000))s @ 16kHz)")

    // Step 2: Align with Qwen3-ForcedAligner
    print("\n[Step 2] Aligning with Qwen3-ForcedAligner...")
    let aligner = try await Qwen3ForcedAligner.fromPretrained(
      modelId: "aufklarer/Qwen3-ForcedAligner-0.6B-4bit"
    ) { progress, status in
      if progress < 1.0 { print("  [\(Int(progress * 100))%] \(status)") }
    }

    let t1 = CFAbsoluteTimeGetCurrent()
    let alignedWords = aligner.align(
      audio: samples16k,
      text: whisperText,
      sampleRate: 16000,
      language: "English"
    )
    let alignTime = CFAbsoluteTimeGetCurrent() - t1
    let alignRTF = alignTime / audioDuration
    print("Qwen3 alignment: \(fmt1(alignTime))s  RTF=\(fmt4(alignRTF))")
    print("Aligned \(alignedWords.count) words")

    // Total time (Whisper + Qwen3)
    let totalTime = whisperTime + alignTime
    let totalRTF = totalTime / audioDuration

    // ── Timing accuracy: one-to-one greedy matching ──
    struct QwenEntry { var start, end: Double; var used: Bool }
    var qwenIndex: [String: [QwenEntry]] = [:]
    for w in alignedWords {
      let key = w.text.lowercased().trimmingCharacters(in: CharacterSet.punctuationCharacters)
      qwenIndex[key, default: []].append(QwenEntry(start: Double(w.startTime), end: Double(w.endTime), used: false))
    }

    var signedStartErrors: [Double] = []
    var absEndErrors: [Double] = []

    for pw in pythonWords {
      let key = pw.word.lowercased().trimmingCharacters(in: CharacterSet.punctuationCharacters)
      guard var candidates = qwenIndex[key] else { continue }
      var bestIdx: Int? = nil
      var bestDist = 1.5
      for (i, c) in candidates.enumerated() where !c.used {
        let d = abs(c.start - pw.start)
        if d < bestDist { bestDist = d; bestIdx = i }
      }
      guard let idx = bestIdx else { continue }
      candidates[idx].used = true
      qwenIndex[key] = candidates
      signedStartErrors.append(candidates[idx].start - pw.start)
      absEndErrors.append(abs(candidates[idx].end - pw.end))
    }

    let matchRate = pythonWords.isEmpty ? 0.0 : Double(signedStartErrors.count) / Double(pythonWords.count)
    let startMAE = signedStartErrors.isEmpty ? 0.0 : signedStartErrors.map(abs).reduce(0, +) / Double(signedStartErrors.count)
    let endMAE = absEndErrors.isEmpty ? 0.0 : absEndErrors.reduce(0, +) / Double(absEndErrors.count)
    let startBias = signedStartErrors.isEmpty ? 0.0 : signedStartErrors.reduce(0, +) / Double(signedStartErrors.count)
    let sortedAbsStart = signedStartErrors.map(abs).sorted()
    let startP90 = sortedAbsStart.isEmpty ? 0.0 : sortedAbsStart[Int(Double(sortedAbsStart.count) * 0.9)]

    // ── Report ──────────────────────────────────────────────────────────────
    let sep = String(repeating: "=", count: 70)
    print("\n\(sep)")
    print("QWEN3-FORCEDALIGNER vs PYTHON WHISPER  [\(audioFile)]")
    print(sep)
    print("  Timing breakdown")
    print(String(format: "    Whisper ASR:      %.1fs  RTF=%.4f", whisperTime, whisperRTF))
    print(String(format: "    Qwen3 align:      %.1fs  RTF=%.4f", alignTime, alignRTF))
    print(String(format: "    TOTAL:            %.1fs  RTF=%.4f", totalTime, totalRTF))
    print("")
    print("  Word counts")
    print(String(format: "    Python baseline:  %ld", pythonWords.count))
    print(String(format: "    Qwen3 aligned:    %ld", alignedWords.count))
    print("")
    print("  Timing accuracy  (\(signedStartErrors.count) matched pairs, ±1.5 s window)")
    print(String(format: "    Match rate:       %.1f%%", matchRate * 100))
    print(String(format: "    Start MAE:        %.3f s  (%.0f ms)", startMAE, startMAE * 1000))
    print(String(format: "    End MAE:          %.3f s  (%.0f ms)", endMAE, endMAE * 1000))
    print(String(format: "    Start p90 err:    %.3f s", startP90))
    print(String(format: "    Start bias:       %+.3f s", startBias))

    let timingGrade: String
    if startMAE < 0.05 { timingGrade = "excellent (< 50 ms)" }
    else if startMAE < 0.10 { timingGrade = "good (< 100 ms)" }
    else if startMAE < 0.20 { timingGrade = "acceptable (< 200 ms)" }
    else { timingGrade = "poor (> 200 ms)" }
    print("    Assessment:       \(timingGrade)")

    // Compare with DTW baseline (our current implementation)
    print("\n  Comparison with DTW-based approach")
    print("    DTW Start MAE:    ~91 ms (from word-compare-full)")
    print(String(format: "    Qwen3 Start MAE:  %.0f ms", startMAE * 1000))
    if startMAE < 0.091 {
      print("    → Qwen3 is MORE accurate than DTW")
    } else {
      print("    → Qwen3 is LESS accurate than DTW")
    }

    // Sample output
    print("\n  First 10 aligned words:")
    for word in alignedWords.prefix(10) {
      print(String(format: "    [%.2f–%.2f] %@", word.startTime, word.endTime, word.text as NSString))
    }
  }

  // MARK: - wav2vec2 CTC Forced Aligner comparison (MLX)

  /// Compare wav2vec2 CTC forced aligner (MLX) against Python Whisper baseline.
  /// Uses Whisper for transcription, then wav2vec2 for word-level alignment.
  static func runWav2Vec2AlignCompare(audioFile: String, baselineFile: String?) async throws {
    let audioURL = URL(fileURLWithPath: fixturesPath).appendingPathComponent(audioFile)
    guard FileManager.default.fileExists(atPath: audioURL.path) else {
      throw BenchmarkError.fixtureNotFound(audioURL.path)
    }

    let audioDuration = try getAudioDuration(audioURL)
    print("\nAudio: \(audioFile)  [\(fmt1(audioDuration))s / \(String(format: "%.1f", audioDuration / 60)) min]")

    // Locate baseline JSON
    let stem = audioURL.deletingPathExtension().lastPathComponent
    let baselineURL: URL
    if let bf = baselineFile {
      baselineURL = URL(fileURLWithPath: bf)
    } else {
      baselineURL = URL(fileURLWithPath: fixturesPath).appendingPathComponent("\(stem)_word_baseline.json")
    }
    guard FileManager.default.fileExists(atPath: baselineURL.path) else {
      print("""
        Baseline not found: \(baselineURL.path)

        Generate it first:
          python benchmarks/scripts/word_baseline.py \(audioURL.path)
        """)
      throw BenchmarkError.fixtureNotFound(baselineURL.path)
    }

    let pythonWords = try JSONDecoder().decode([PythonWord].self, from: Data(contentsOf: baselineURL))
    print("Python baseline: \(pythonWords.count) words loaded from \(baselineURL.lastPathComponent)")

    // Step 1: Transcribe with Whisper (batched, no word timestamps)
    print("\n[Step 1] Transcribing with Whisper (batched)...")
    let (whisperEngine, _, _) = try await loadEngine(audioFile: audioFile)
    let t0 = CFAbsoluteTimeGetCurrent()
    let transcription = try await whisperEngine.transcribeBatched(
      audioURL,
      timestamps: .none,
      batchSize: 4
    )
    let whisperTime = CFAbsoluteTimeGetCurrent() - t0
    let whisperRTF = whisperTime / audioDuration
    print("Whisper transcription: \(fmt1(whisperTime))s  RTF=\(fmt4(whisperRTF))")

    // Extract text from segments (for reference only)
    let whisperText = transcription.segments.map { $0.text.trimmingCharacters(in: CharacterSet.whitespaces) }.joined(separator: " ")
    print("Swift Whisper text (\(whisperText.count) chars): \(whisperText.prefix(200))...")

    // IMPORTANT: Use baseline transcript for fair comparison
    // This ensures we're evaluating wav2vec2 alignment quality, not transcript differences
    let baselineText = pythonWords.map { $0.word }.joined(separator: " ")
    print("\nUsing Python baseline transcript (\(baselineText.count) chars) for alignment")
    print("Baseline text: \(baselineText.prefix(200))...")

    // Load audio for wav2vec2 (16kHz mono)
    print("\nLoading audio for alignment...")
    let samples16k = try load16kMono(url: audioURL, maxSeconds: nil)
    print("Loaded \(samples16k.count) samples (\(fmt1(Double(samples16k.count) / 16000))s @ 16kHz)")

    // Step 2: Run VAD to strip leading/trailing silence
    print("\n[Step 2] Running Silero VAD to strip silence...")
    // Use two-pass approach:
    // 1. Strict parameters to find true first speech start (filters out leading noise)
    // 2. Standard parameters to detect all speech segments
    let strictVadConfig = VadSegmentationConfig(
      minSpeechDuration: 0.5,   // Filter out brief noises at start
      minSilenceDuration: 0.5,
      maxSpeechDuration: 600.0,
      speechPadding: 0.1
    )
    let standardVadConfig = VadSegmentationConfig(
      minSpeechDuration: 0.15,
      minSilenceDuration: 0.5,
      maxSpeechDuration: 600.0,
      speechPadding: 0.1
    )
    let vad = try await VadManager(config: VadConfig(computeUnits: .cpuAndNeuralEngine))

    // Get strict segments for accurate first speech detection
    let strictSegments = try await vad.segmentSpeech(samples16k, config: strictVadConfig)
    // Get standard segments for complete speech coverage
    let vadSegments = try await vad.segmentSpeech(samples16k, config: standardVadConfig)

    guard !vadSegments.isEmpty else {
      print("ERROR: No speech segments detected by VAD")
      throw BenchmarkError.fixtureNotFound("No speech detected in audio")
    }

    // Use strict segments to find true first speech start, but standard segments for coverage
    let firstSpeechStart = strictSegments.isEmpty ? vadSegments.first!.startTime : strictSegments.first!.startTime
    let lastSpeechEnd = vadSegments.last!.endTime
    let originalDuration = Double(samples16k.count) / 16000.0
    let trimmedDuration = lastSpeechEnd - firstSpeechStart
    let leadingSilence = firstSpeechStart
    let trailingSilence = originalDuration - lastSpeechEnd

    print("VAD detected \(vadSegments.count) speech segments (standard)")
    if !strictSegments.isEmpty {
      print("VAD detected \(strictSegments.count) speech segments (strict)")
    }
    print(String(format: "  First speech (strict): %.2fs  Last speech ends: %.2fs", firstSpeechStart, lastSpeechEnd))
    print(String(format: "  Leading silence: %.2fs  Trailing silence: %.2fs", leadingSilence, trailingSilence))
    print(String(format: "  Trimmed audio: %.2fs → %.2fs (%.1f%% reduction)",
      originalDuration, trimmedDuration,
      (1.0 - trimmedDuration / originalDuration) * 100))

    // Trim audio to speech region only
    let startSample = Int(firstSpeechStart * 16000)
    let endSample = Int(lastSpeechEnd * 16000)
    let trimmedAudio = Array(samples16k[startSample..<endSample])
    print("Trimmed to \(trimmedAudio.count) samples (\(fmt1(Double(trimmedAudio.count) / 16000))s @ 16kHz)")

    // Step 3: Align with wav2vec2 MLX using BASELINE transcript
    print("\n[Step 3] Aligning with wav2vec2 CTC forced aligner (MLX)...")
    print("[Step 3] Loading Wav2Vec2AlignerMLX from Hugging Face...")

    let aligner = try await Wav2Vec2AlignerMLX.fromPretrained(
      modelId: "facebook/wav2vec2-base-960h"
    ) { progress in
      if progress.fractionCompleted < 1.0 {
        print("  [\(Int(progress.fractionCompleted * 100))%] Downloading \(progress.localizedDescription ?? "")")
      }
    }
    print("Model loaded successfully")

    let t1 = CFAbsoluteTimeGetCurrent()
    let alignedWords = try aligner.align(
      audio: trimmedAudio,
      text: baselineText  // Use baseline transcript, not Swift Whisper
    )
    let alignTime = CFAbsoluteTimeGetCurrent() - t1

    // Adjust timestamps back by adding the original offset (leading silence)
    let offset = Float(firstSpeechStart)
    let adjustedWords = alignedWords.map { word -> WordAlignment in
      WordAlignment(
        word: word.word,
        startTime: word.startTime + offset,
        endTime: word.endTime + offset
      )
    }
    let alignRTF = alignTime / audioDuration
    print("wav2vec2 alignment: \(fmt1(alignTime))s  RTF=\(fmt4(alignRTF))")
    print("Aligned \(adjustedWords.count) words (timestamps adjusted for VAD trim)")

    // Total time (Whisper + wav2vec2)
    let totalTime = whisperTime + alignTime
    let totalRTF = totalTime / audioDuration

    // ── Timing accuracy: one-to-one greedy matching ──
    struct Wav2Vec2Entry { var start, end: Float; var used: Bool }
    var wav2vec2Index: [String: [Wav2Vec2Entry]] = [:]
    for w in adjustedWords {
      let key = w.word.lowercased().trimmingCharacters(in: CharacterSet.punctuationCharacters)
      wav2vec2Index[key, default: []].append(Wav2Vec2Entry(start: w.startTime, end: w.endTime, used: false))
    }

    var signedStartErrors: [Double] = []
    var absEndErrors: [Double] = []

    for pw in pythonWords {
      let key = pw.word.lowercased().trimmingCharacters(in: CharacterSet.punctuationCharacters)
      guard var candidates = wav2vec2Index[key] else { continue }
      var bestIdx: Int? = nil
      var bestDist = 1.5  // Max time difference to match (seconds)

      for (i, c) in candidates.enumerated() {
        let dist = max(abs(Double(c.start) - pw.start), abs(Double(c.end) - pw.end))
        if dist < bestDist {
          bestDist = dist
          bestIdx = i
        }
      }

      guard let idx = bestIdx else { continue }
      candidates[idx].used = true
      wav2vec2Index[key] = candidates
      signedStartErrors.append(Double(candidates[idx].start) - pw.start)
      absEndErrors.append(abs(Double(candidates[idx].end) - pw.end))
    }

    let matchRate = pythonWords.isEmpty ? 0.0 : Double(signedStartErrors.count) / Double(pythonWords.count)
    let startMAE = signedStartErrors.isEmpty ? 0.0 : signedStartErrors.map(abs).reduce(0, +) / Double(signedStartErrors.count)
    let endMAE = absEndErrors.isEmpty ? 0.0 : absEndErrors.reduce(0, +) / Double(absEndErrors.count)
    let startBias = signedStartErrors.isEmpty ? 0.0 : signedStartErrors.reduce(0, +) / Double(signedStartErrors.count)
    let sortedAbsStart = signedStartErrors.map(abs).sorted()
    let startP90 = sortedAbsStart.isEmpty ? 0.0 : sortedAbsStart[Int(Double(sortedAbsStart.count) * 0.9)]

    // ── Report ──────────────────────────────────────────────────────────────
    let sep = String(repeating: "=", count: 70)
    print("\n\(sep)")
    print("WAV2VEC2-MLX vs PYTHON WHISPER  [\(audioFile)]")
    print(sep)
    print("  Timing breakdown")
    print(String(format: "    Whisper ASR:    %.1fs  RTF=%.4f", whisperTime, whisperRTF))
    print(String(format: "    wav2vec2 align: %.1fs  RTF=%.4f", alignTime, alignRTF))
    print(String(format: "    TOTAL:          %.1fs  RTF=%.4f", totalTime, totalRTF))
    print("")
    print("  Word counts")
    print(String(format: "    Python baseline: %ld", pythonWords.count))
    print(String(format: "    wav2vec2 aligned: %ld", adjustedWords.count))
    print("")
    print("  Timing accuracy  (\(signedStartErrors.count) matched pairs, ±1.5 s window)")
    print(String(format: "    Match rate:     %.1f%%", matchRate * 100))
    print(String(format: "    Start MAE:      %.3f s  (%.0f ms)", startMAE, startMAE * 1000))
    print(String(format: "    End MAE:        %.3f s  (%.0f ms)", endMAE, endMAE * 1000))
    print(String(format: "    Start p90 err:  %.3f s", startP90))
    print(String(format: "    Start bias:     %+.3f s", startBias))

    let timingGrade: String
    if startMAE < 0.05 { timingGrade = "excellent (< 50 ms)" }
    else if startMAE < 0.10 { timingGrade = "good (< 100 ms)" }
    else if startMAE < 0.20 { timingGrade = "acceptable (< 200 ms)" }
    else { timingGrade = "poor (> 200 ms)" }
    print("    Assessment:     \(timingGrade)")

    // Sample output
    print("\n  First 10 aligned words:")
    for word in adjustedWords.prefix(10) {
      print(String(format: "    [%.2f–%.2f] %@", word.startTime, word.endTime, word.word as NSString))
    }
  }

  // MARK: - VAD scan

  /// Run Silero VAD over a sample of real meeting recordings (read-only) and report
  /// segment-duration distribution statistics. Useful for validating that the chosen
  /// VAD parameters produce Whisper-compatible chunks (≤ 29 s) across real audio.
  /// scanDurationSeconds: how many seconds of each file to analyse (nil = full file).
  /// Five minutes captures enough speech variety to validate segment sizes.
  static func runVADScan(meetingsDir: String, maxFiles: Int, scanDurationSeconds: Double = 300) async throws {
    let root = URL(fileURLWithPath: meetingsDir)

    // Collect *_combined.mp3 files on a background thread (FileManager enumerator
    // is unavailable from async contexts in Swift 6).
    let allFiles: [URL] = try await Task.detached(priority: .userInitiated) {
      var found: [URL] = []
      let fm = FileManager.default
      guard let enumerator = fm.enumerator(at: root, includingPropertiesForKeys: nil) else {
        return found
      }
      while let item = enumerator.nextObject() as? URL {
        if item.lastPathComponent.hasSuffix("_combined.mp3") {
          found.append(item)
        }
      }
      return found
    }.value

    var shuffled = allFiles
    shuffled.shuffle()
    let files = Array(shuffled.prefix(maxFiles))

    let separator = String(repeating: "=", count: 70)
    print("\n\(separator)")
    print("VAD SCAN — \(meetingsDir)")
    print("  \(files.count) of \(allFiles.count) available files  ·  first \(Int(scanDurationSeconds))s per file")
    print(separator)
    print("""
      Silero VAD params:
        minSpeechDuration : 0.15 s
        minSilenceDuration: 0.50 s
        maxSpeechDuration : 29.0 s   ← Whisper context window
        speechPadding     : 0.10 s
      """)

    let vadConfig = VadSegmentationConfig(
      minSpeechDuration: 0.15,
      minSilenceDuration: 0.5,
      maxSpeechDuration: 29.0,
      speechPadding: 0.1,
      silenceThresholdForSplit: 0.3
    )

    print("Initialising Silero VAD (CoreML)...")
    let vad = try await VadManager(config: VadConfig(computeUnits: .cpuAndNeuralEngine))
    print("VAD ready.\n")

    struct FileStats {
      let name: String
      let audioDuration: Double
      let segments: [VadSegment]
      var shortCount: Int { segments.filter { $0.duration <= 29.0 }.count }
      var longCount: Int { segments.filter { $0.duration > 29.0 }.count }
      var coverage: Double {
        guard audioDuration > 0 else { return 0 }
        return segments.reduce(0) { $0 + $1.duration } / audioDuration
      }
    }

    var allStats: [FileStats] = []

    for (i, url) in files.enumerated() {
      let name = url.lastPathComponent
      print("[\(i + 1)/\(files.count)]  \(name)")

      let audioDuration: Double
      do {
        audioDuration = try getAudioDuration(url)
      } catch {
        print("  SKIP — cannot read duration: \(error)")
        continue
      }
      print("  Duration: \(fmt1(audioDuration / 60)) min")

      var samples: [Float]
      do {
        samples = try load16kMono(url: url, maxSeconds: scanDurationSeconds)
      } catch {
        print("  SKIP — cannot load audio: \(error)")
        continue
      }
      let scannedDuration = Double(samples.count) / 16_000
      let isPartial = scannedDuration < audioDuration - 1
      if isPartial {
        print(String(format: "  (scanning first %.0fs of %.0fs)", scannedDuration, audioDuration))
      }

      let t0 = CFAbsoluteTimeGetCurrent()
      let rawSegments = try await vad.segmentSpeech(samples, config: vadConfig)
      let elapsed = CFAbsoluteTimeGetCurrent() - t0

      // Mirror the merge step used in transcribeBatched()
      let segments = mergeForScan(rawSegments)

      let durations = segments.map { $0.duration }
      let short = segments.filter { $0.duration <= 29.0 }
      let long = segments.filter { $0.duration > 29.0 }
      let effectiveDuration = isPartial ? scannedDuration : audioDuration
      let coverage = effectiveDuration > 0 ? segments.reduce(0) { $0 + $1.duration } / effectiveDuration : 0

      let mergeNote = rawSegments.count != segments.count
        ? "  (\(rawSegments.count) raw → \(segments.count) after merge)"
        : ""
      print(String(format: "  VAD: %.2fs  →  %ld segments%@  (short: %ld  long: %ld)",
                   elapsed, segments.count, mergeNote as NSString, short.count, long.count))

      if !durations.isEmpty {
        let sorted = durations.sorted()
        let avg = durations.reduce(0, +) / Double(durations.count)
        let p50 = sorted[sorted.count / 2]
        let p90 = sorted[Int(Double(sorted.count) * 0.9)]
        let veryShort = durations.filter { $0 < 1.0 }.count
        print(String(format: "  Durations: min=%.1fs  avg=%.1fs  p50=%.1fs  p90=%.1fs  max=%.1fs",
                     sorted.first ?? 0, avg, p50, p90, sorted.last ?? 0))
        print(String(format: "  Coverage:  %.1f%% of scanned audio in segments", coverage * 100))
        if veryShort > 0 {
          print("  Note: \(veryShort) merged segment(s) < 1s (may yield poor Whisper output)")
        }
      }
      print("")

      allStats.append(FileStats(name: name, audioDuration: scannedDuration, segments: segments))
    }

    // Aggregate
    let allSegments = allStats.flatMap { $0.segments }
    guard !allSegments.isEmpty else {
      print("No segments collected.")
      return
    }

    let totalAudioMinutes = allStats.reduce(0) { $0 + $1.audioDuration } / 60.0
    let buckets: [(label: String, min: Double, max: Double)] = [
      (" <  5s", 0, 5),
      (" 5–10s", 5, 10),
      ("10–15s", 10, 15),
      ("15–20s", 15, 20),
      ("20–25s", 20, 25),
      ("25–29s", 25, 29),
      ("  >29s", 29, Double.infinity),
    ]

    print(separator)
    print("AGGREGATE  [\(allStats.count) files · \(allSegments.count) segments · \(fmt1(totalAudioMinutes)) min scanned]")
    print(separator)
    print("  Segment duration histogram:")
    for bucket in buckets {
      let count = allSegments.filter { $0.duration >= bucket.min && $0.duration < bucket.max }.count
      let pct = Double(count) / Double(allSegments.count) * 100
      let bar = String(repeating: "█", count: max(1, Int(pct / 2)))
      print(String(format: "    %@  %4ld  (%5.1f%%)  %@", bucket.label as NSString, count, pct, bar as NSString))
    }

    let longCount = allSegments.filter { $0.duration > 29.0 }.count
    let allDurations = allSegments.map { $0.duration }.sorted()
    let avg = allDurations.reduce(0, +) / Double(allDurations.count)
    let p50 = allDurations[allDurations.count / 2]
    let p90 = allDurations[Int(Double(allDurations.count) * 0.9)]
    let totalCoverage = allStats.reduce(0.0) { $0 + $1.segments.reduce(0) { $0 + $1.duration } }
    let totalAudioSec = allStats.reduce(0.0) { $0 + $1.audioDuration }

    print(String(format: "\n  Overall durations: avg=%.1fs  p50=%.1fs  p90=%.1fs  max=%.1fs",
                 avg, p50, p90, allDurations.last ?? 0))
    print(String(format: "  Coverage: %.1f%% of audio captured in segments", totalCoverage / totalAudioSec * 100))
    print(String(format: "  Long-path exposure: %ld / %ld segments (%.1f%%)",
                 longCount, allSegments.count, Double(longCount) / Double(allSegments.count) * 100))
    print(String(format: "  Avg segments/min: %.1f", Double(allSegments.count) / totalAudioMinutes))

    if longCount == 0 {
      print("\n  All segments fit within the 29 s Whisper window — params look good.")
    } else {
      print("\n  \(longCount) segment(s) exceed 29 s → will use the sequential fallback path.")
    }
  }

  // MARK: - Audio loading helper (16 kHz mono, AVAssetReader)

  /// Reads a compressed audio file (MP3, M4A, WAV…) and returns 16 kHz mono Float32
  /// samples.  Uses AVAssetReader so the OS handles decompression and resampling
  /// chunk-by-chunk without ever materialising the full high-res PCM in memory.
  /// Load and resample audio to 16 kHz mono Float32 using AVAssetReader (streaming, low memory).
  /// `maxSeconds` limits how much of the file is decoded — pass `nil` for the full file.
  static func load16kMono(url: URL, maxSeconds: Double? = nil) throws -> [Float] {
    let asset = AVURLAsset(url: url, options: [AVURLAssetPreferPreciseDurationAndTimingKey: false])
    let tracks = asset.tracks(withMediaType: .audio)
    guard let track = tracks.first else { throw BenchmarkError.audioLoadFailed }

    let outputSettings: [String: Any] = [
      AVFormatIDKey: kAudioFormatLinearPCM,
      AVSampleRateKey: 16_000.0,
      AVNumberOfChannelsKey: 1,
      AVLinearPCMBitDepthKey: 32,
      AVLinearPCMIsFloatKey: true,
      AVLinearPCMIsBigEndianKey: false,
      AVLinearPCMIsNonInterleaved: false,
    ]

    guard let reader = try? AVAssetReader(asset: asset) else { throw BenchmarkError.audioLoadFailed }

    if let limit = maxSeconds {
      reader.timeRange = CMTimeRange(start: .zero, duration: CMTime(seconds: limit, preferredTimescale: 16_000))
    }

    let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
    output.alwaysCopiesSampleData = false
    reader.add(output)

    guard reader.startReading() else {
      throw BenchmarkError.audioLoadFailed
    }

    let capacity = Int((maxSeconds ?? 3600) * 16_000) + 16_000
    var samples: [Float] = []
    samples.reserveCapacity(capacity)

    while let sampleBuffer = output.copyNextSampleBuffer() {
      guard let block = CMSampleBufferGetDataBuffer(sampleBuffer) else { continue }
      let byteCount = CMBlockBufferGetDataLength(block)
      let floatCount = byteCount / MemoryLayout<Float>.size
      samples.append(contentsOf: [Float](unsafeUninitializedCapacity: floatCount) { buf, count in
        CMBlockBufferCopyDataBytes(block, atOffset: 0, dataLength: byteCount, destination: buf.baseAddress!)
        count = floatCount
      })
    }

    if reader.status == .failed {
      throw reader.error ?? BenchmarkError.audioLoadFailed
    }
    return samples
  }

  // MARK: - Segment merge (mirrors WhisperSTT.mergeAdjacentSegments)

  static func mergeForScan(_ segments: [VadSegment], maxDuration: TimeInterval = 29.0) -> [VadSegment] {
    guard !segments.isEmpty else { return [] }
    var merged: [VadSegment] = []
    var mergeStart = segments[0].startTime
    var mergeEnd = segments[0].endTime
    for seg in segments.dropFirst() {
      if seg.endTime - mergeStart <= maxDuration {
        mergeEnd = seg.endTime
      } else {
        merged.append(VadSegment(startTime: mergeStart, endTime: mergeEnd))
        mergeStart = seg.startTime
        mergeEnd = seg.endTime
      }
    }
    merged.append(VadSegment(startTime: mergeStart, endTime: mergeEnd))
    return merged
  }

  // MARK: - Formatting helpers

  static func fmt1(_ v: Double) -> String { String(format: "%.1f", v) }
  static func fmt2(_ v: Double) -> String { String(format: "%.2f", v) }
  static func fmt4(_ v: Double) -> String { String(format: "%.4f", v) }
}

enum BenchmarkError: Error {
  case fixtureNotFound(String)
  case audioLoadFailed
}
