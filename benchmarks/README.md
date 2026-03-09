# Whisper Performance Benchmarks

This directory contains fixtures and tooling for validating Whisper transcription performance and quality.

## Fixtures

Uses the AMI Corpus ES2002a meeting recording - a public dataset for reproducibility.

- `ami_ES2002a_full.wav` - Full 21-minute meeting recording (16kHz mono WAV)
- `ami_ES2002a_5min.wav` - 5-minute segment for quick testing
- `ami_ES2002a_baseline.txt` - Baseline transcription from Python mlx-whisper

Source: [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) ES2002a session

## Running Benchmarks

### Python Benchmark (recommended for development)
```bash
cd ~/Work/mlx-swift-audio/benchmarks

# Quick sanity check (5 minutes)
~/Work/meeting-transcriber/.venv/bin/python benchmark.py quick

# Full benchmark (21 minutes)
~/Work/meeting-transcriber/.venv/bin/python benchmark.py full

# Compare recent results
~/Work/meeting-transcriber/.venv/bin/python benchmark.py compare
```

### Swift Benchmark (requires Xcode for Metal support)
```bash
# Via xcodebuild (if scheme is configured for testing)
xcodebuild test -scheme MLXAudio -only-testing:WhisperBenchmark/quickSanityCheck
```

## Performance Targets

| Metric | Python Baseline | Swift Target | Notes |
|--------|-----------------|--------------|-------|
| RTF (21min) | 0.0500 | 0.035 | 30% faster than Python |
| Similarity | - | 85% min | Jaccard word overlap vs baseline |
| WER | - | 30% max | Word error rate vs baseline |

Python baseline established on 2026-03-09 with mlx-community/whisper-large-v3-turbo-q4.

## Validating Optimizations

1. Run baseline before changes:
   ```bash
   python benchmark.py full
   ```

2. Make optimization changes to Swift code

3. Run benchmark again and compare:
   ```bash
   python benchmark.py full
   python benchmark.py compare
   ```

## Fixture Source

AMI Corpus ES2002a:
- Public meeting corpus from University of Edinburgh
- Characteristics: 4 speakers, project kickoff meeting, ~21 minutes
- Multiple overlapping speakers, natural conversation
- Baseline generated with Python mlx-whisper large-v3-turbo-q4
