import AVFoundation
import FluidAudio
import Foundation
import MLXAudio

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
