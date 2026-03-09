#!/usr/bin/env python3
"""
Whisper Benchmark Tool

Measures transcription performance and quality against baseline.
Run from the benchmarks directory:
    python benchmark.py quick      # 5-minute segment
    python benchmark.py full       # Full 30-minute benchmark
    python benchmark.py compare    # Compare last two results
"""

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import mlx_whisper


@dataclass
class BenchmarkResult:
    timestamp: str
    audio_file: str
    audio_duration: float
    processing_time: float
    rtf: float
    word_count: int
    model: str
    engine: str  # "python" or "swift"
    text: str


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def compute_word_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts."""
    import re

    def normalize(text: str) -> set[str]:
        words = re.sub(r"[^\w\s]", "", text.lower()).split()
        return set(words)

    words1 = normalize(text1)
    words2 = normalize(text2)

    if not words1 or not words2:
        return 0.0

    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def compute_wer(hypothesis: str, reference: str) -> float:
    """Compute word error rate using Levenshtein distance on words."""
    import re

    def normalize(text: str) -> list[str]:
        return [w for w in re.sub(r"[^\w\s]", "", text.lower()).split() if w]

    hyp = normalize(hypothesis)
    ref = normalize(reference)

    if not ref:
        return 0.0 if not hyp else 1.0

    # Levenshtein distance on word sequences
    dp = [[0] * (len(ref) + 1) for _ in range(len(hyp) + 1)]
    for i in range(len(hyp) + 1):
        dp[i][0] = i
    for j in range(len(ref) + 1):
        dp[0][j] = j

    for i in range(1, len(hyp) + 1):
        for j in range(1, len(ref) + 1):
            if hyp[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[len(hyp)][len(ref)] / len(ref)


def run_python_whisper(
    audio_path: Path, model: str = "mlx-community/whisper-large-v3-turbo-q4"
) -> BenchmarkResult:
    """Run Python mlx-whisper and return results."""
    print(f"\n[Python MLX-Whisper] Model: {model}")
    print(f"  Audio: {audio_path.name}")

    duration = get_audio_duration(audio_path)
    print(f"  Duration: {duration:.1f}s ({duration/60:.1f} minutes)")

    print("  Transcribing...")
    start = time.time()
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model,
        language="en",
        task="transcribe",
    )
    elapsed = time.time() - start

    text = result["text"].strip()
    word_count = len(text.split())
    rtf = elapsed / duration

    print(f"  RTF: {rtf:.4f}")
    print(f"  Processing time: {elapsed:.1f}s")
    print(f"  Words: {word_count}")

    return BenchmarkResult(
        timestamp=datetime.now().isoformat(),
        audio_file=audio_path.name,
        audio_duration=duration,
        processing_time=elapsed,
        rtf=rtf,
        word_count=word_count,
        model=model,
        engine="python",
        text=text,
    )


def save_result(result: BenchmarkResult, results_dir: Path) -> Path:
    """Save benchmark result to JSON file."""
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{result.engine}_{timestamp}.json"
    filepath = results_dir / filename

    with open(filepath, "w") as f:
        json.dump(asdict(result), f, indent=2)

    print(f"  Saved: {filepath}")
    return filepath


def load_baseline(fixtures_dir: Path, mode: str = "full") -> str:
    """Load baseline transcript text for the given mode."""
    if mode == "quick":
        # For quick mode, there's no separate baseline - skip comparison
        return ""
    baseline_file = fixtures_dir / "ami_ES2002a_baseline.txt"
    if baseline_file.exists():
        return baseline_file.read_text()
    return ""


def compare_results(results_dir: Path, fixtures_dir: Path) -> None:
    """Compare recent benchmark results."""
    result_files = sorted(results_dir.glob("*.json"), reverse=True)

    if len(result_files) < 1:
        print("No benchmark results found.")
        return

    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON")
    print("=" * 70)

    baseline_text = load_baseline(fixtures_dir)

    # Load the most recent result
    latest = json.loads(result_files[0].read_text())
    print(f"\nLatest result: {result_files[0].name}")
    print(f"  Engine: {latest['engine']}")
    print(f"  RTF: {latest['rtf']:.4f}")
    print(f"  Processing time: {latest['processing_time']:.1f}s")
    print(f"  Words: {latest['word_count']}")

    if baseline_text:
        similarity = compute_word_similarity(latest["text"], baseline_text)
        wer = compute_wer(latest["text"], baseline_text)
        print(f"  Similarity to baseline: {similarity*100:.1f}%")
        print(f"  WER vs baseline: {wer*100:.1f}%")

    # Compare with previous if available
    if len(result_files) > 1:
        previous = json.loads(result_files[1].read_text())
        print(f"\nPrevious result: {result_files[1].name}")
        print(f"  Engine: {previous['engine']}")
        print(f"  RTF: {previous['rtf']:.4f}")

        rtf_change = (latest["rtf"] - previous["rtf"]) / previous["rtf"] * 100
        print(f"\n  RTF change: {rtf_change:+.1f}%")
        if rtf_change < 0:
            print("  ✓ Faster")
        else:
            print("  ✗ Slower")


def main():
    parser = argparse.ArgumentParser(description="Whisper Benchmark Tool")
    parser.add_argument(
        "mode",
        choices=["quick", "full", "compare", "baseline"],
        default="quick",
        nargs="?",
        help="Benchmark mode: quick (5min), full (30min), compare results, baseline (save Python result as baseline)",
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="Number of benchmark runs (default: 1)"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    fixtures_dir = script_dir / "fixtures"
    results_dir = script_dir / "results"

    if args.mode == "compare":
        compare_results(results_dir, fixtures_dir)
        return

    # Select audio file based on mode
    if args.mode == "quick":
        audio_file = fixtures_dir / "ami_ES2002a_5min.wav"
    else:  # full or baseline
        audio_file = fixtures_dir / "ami_ES2002a_full.wav"

    if not audio_file.exists():
        print(f"Error: Audio fixture not found: {audio_file}")
        print("Please ensure the benchmark fixtures are set up.")
        return 1

    print("=" * 70)
    print(f"Whisper Benchmark - {args.mode.upper()} mode")
    print("=" * 70)

    baseline_text = load_baseline(fixtures_dir, args.mode)

    # Run benchmark(s)
    results = []
    for run in range(args.runs):
        if args.runs > 1:
            print(f"\n--- Run {run + 1}/{args.runs} ---")
        result = run_python_whisper(audio_file)
        results.append(result)

    # Compute stats
    if args.runs > 1:
        rtfs = [r.rtf for r in results]
        avg_rtf = sum(rtfs) / len(rtfs)
        print(f"\n{'='*70}")
        print("STATISTICS")
        print(f"{'='*70}")
        print(f"  Average RTF: {avg_rtf:.4f}")
        print(f"  Min RTF: {min(rtfs):.4f}")
        print(f"  Max RTF: {max(rtfs):.4f}")

    # Quality check against baseline
    if baseline_text and results:
        latest = results[-1]
        similarity = compute_word_similarity(latest.text, baseline_text)
        wer = compute_wer(latest.text, baseline_text)

        print(f"\n{'-'*70}")
        print("QUALITY METRICS")
        print(f"{'-'*70}")
        print(f"  Jaccard Similarity: {similarity*100:.1f}%")
        print(f"  Word Error Rate: {wer*100:.1f}%")

    # Save result
    if results:
        save_result(results[-1], results_dir)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    python_baseline_rtf = 0.0500  # Baseline from Python mlx-whisper on AMI ES2002a (21min)
    swift_target_rtf = python_baseline_rtf * 0.7  # 30% faster than Python = 0.035
    current_rtf = results[-1].rtf if results else 0

    print(f"  Python baseline RTF: {python_baseline_rtf:.4f}")
    print(f"  Swift target RTF: {swift_target_rtf:.4f} (30% faster than Python)")
    print(f"  Current RTF: {current_rtf:.4f}")

    if current_rtf <= swift_target_rtf:
        speedup = (python_baseline_rtf - current_rtf) / python_baseline_rtf * 100
        print(f"✓ Swift target MET: {speedup:.1f}% faster than Python baseline")
    else:
        improvement_needed = (current_rtf - swift_target_rtf) / current_rtf * 100
        print(f"⚠ Swift target NOT MET: need {improvement_needed:.1f}% improvement")
        if current_rtf <= python_baseline_rtf:
            speedup = (python_baseline_rtf - current_rtf) / python_baseline_rtf * 100
            print(f"  Currently {speedup:.1f}% faster than Python")


if __name__ == "__main__":
    main()
