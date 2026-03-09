#!/usr/bin/env python3
"""
Generate a word-level timestamp baseline using Python mlx-whisper.

Usage:
    python benchmarks/scripts/word_baseline.py <audio_file> [output_json]

Outputs a JSON array of word objects:
    [{"word": "hello", "start": 0.0, "end": 0.3, "probability": 0.95}, ...]

The output file defaults to <audio_file_stem>_word_baseline.json in the same
directory as the audio file.  The Swift benchmark CLI reads this file for the
word-compare mode.
"""

import json
import sys
from pathlib import Path

try:
    import mlx_whisper
except ImportError:
    print("mlx_whisper not found. Activate the meeting-transcriber venv:", file=sys.stderr)
    print("  source /Users/francesco.mosca/Work/meeting-transcriber/.venv/bin/activate", file=sys.stderr)
    sys.exit(1)

MODEL = "mlx-community/whisper-large-v3-turbo"
QUANTIZATION = "q4"  # matches Swift benchmark default


def run(audio_path: Path, output_path: Path) -> None:
    print(f"Audio:  {audio_path}", file=sys.stderr)
    print(f"Model:  {MODEL} [{QUANTIZATION}]", file=sys.stderr)
    print("Running mlx-whisper with word_timestamps=True …", file=sys.stderr)

    result = mlx_whisper.transcribe(
                str(audio_path),
                path_or_hf_repo=f"mlx-community/whisper-large-v3-turbo-{QUANTIZATION}",
                language="en",
                word_timestamps=True,
                temperature=0.0,
                condition_on_previous_text=False,
            )

    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            words.append(
                {
                    "word": w["word"].strip(),
                    "start": round(float(w["start"]), 4),
                    "end": round(float(w["end"]), 4),
                    "probability": round(float(w.get("probability", 0.0)), 4),
                }
            )

    output_path.write_text(json.dumps(words, ensure_ascii=False, indent=2))
    print(f"Wrote {len(words)} words → {output_path}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    audio = Path(sys.argv[1]).expanduser().resolve()
    if not audio.exists():
        print(f"File not found: {audio}", file=sys.stderr)
        sys.exit(1)

    default_out = audio.parent / f"{audio.stem}_word_baseline.json"
    out = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 else default_out

    run(audio, out)
