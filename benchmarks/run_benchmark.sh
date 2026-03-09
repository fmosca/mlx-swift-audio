#!/bin/bash
# Whisper Benchmark Runner
# Usage: ./run_benchmark.sh [quick|full|all]

set -e

cd "$(dirname "$0")/.."

MODE="${1:-quick}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="benchmarks/results"
mkdir -p "$LOG_DIR"

case "$MODE" in
  quick)
    echo "Running quick sanity check (5 minutes)..."
    swift test --filter WhisperBenchmark/quickSanityCheck 2>&1 | tee "$LOG_DIR/quick_${TIMESTAMP}.log"
    ;;
  full)
    echo "Running full benchmark (30 minutes)..."
    swift test --filter WhisperBenchmark/longFormBenchmark 2>&1 | tee "$LOG_DIR/full_${TIMESTAMP}.log"
    ;;
  all)
    echo "Running all benchmarks..."
    swift test --filter WhisperBenchmark 2>&1 | tee "$LOG_DIR/all_${TIMESTAMP}.log"
    ;;
  compare)
    # Compare last two full benchmark results
    LATEST=$(ls -t "$LOG_DIR"/full_*.log 2>/dev/null | head -1)
    PREVIOUS=$(ls -t "$LOG_DIR"/full_*.log 2>/dev/null | head -2 | tail -1)
    if [ -n "$LATEST" ] && [ -n "$PREVIOUS" ] && [ "$LATEST" != "$PREVIOUS" ]; then
      echo "Comparing:"
      echo "  Previous: $PREVIOUS"
      echo "  Latest:   $LATEST"
      echo ""
      echo "=== Previous Results ==="
      grep -E "(Average RTF|Jaccard Similarity|Word Error Rate)" "$PREVIOUS" || true
      echo ""
      echo "=== Latest Results ==="
      grep -E "(Average RTF|Jaccard Similarity|Word Error Rate)" "$LATEST" || true
    else
      echo "Need at least 2 benchmark results to compare"
      echo "Available: $(ls "$LOG_DIR"/full_*.log 2>/dev/null | wc -l | tr -d ' ') results"
    fi
    ;;
  *)
    echo "Usage: $0 [quick|full|all|compare]"
    echo "  quick   - Run 5-minute sanity check"
    echo "  full    - Run full 30-minute benchmark"
    echo "  all     - Run all benchmark tests"
    echo "  compare - Compare last two full benchmark results"
    exit 1
    ;;
esac

echo ""
echo "Log saved to: $LOG_DIR/"
