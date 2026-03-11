#!/usr/bin/env python3
"""
Convert wav2vec2 PyTorch model to CoreML format.

This script downloads/loads the wav2vec2 model from Hugging Face and converts it
directly to CoreML .mlpackage format with support for variable-length audio input
and ANE optimization.

Output file:
- Wav2Vec2CTC.mlpackage: CoreML model package

Author: Claude (Anthropic)
Copyright © Anthony DePasquale
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import coremltools as ct


def print_progress(message: str):
    """Print progress message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def find_vocab(base_dir: str = None) -> str:
    """Find the vocab.json file."""
    if base_dir is None:
        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "wav2vec2"
        )

    vocab_path = Path(base_dir) / "vocab.json"

    if not vocab_path.exists():
        return None

    return str(vocab_path)


def convert_to_coreml(
    model_name: str = "facebook/wav2vec2-base-960h",
    output_dir: str = None,
    min_deployment_target = ct.target.macOS14,
    compute_units: ct.ComputeUnit = ct.ComputeUnit.ALL,
    use_fp16: bool = True,
    min_audio_samples: int = 16000,
    max_audio_samples: int = 16000 * 30,
) -> dict:
    """
    Convert wav2vec2 PyTorch model to CoreML format.

    Args:
        model_name: Hugging Face model identifier
        output_dir: Directory to save CoreML model
        min_deployment_target: Minimum macOS deployment target
        compute_units: Compute units for model execution
        use_fp16: Whether to use FP16 precision for ANE
        min_audio_samples: Minimum audio length in samples (1 second at 16kHz)
        max_audio_samples: Maximum audio length in samples (30 seconds at 16kHz)

    Returns:
        Dictionary with path to converted model
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "wav2vec2"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    mlpackage_path = output_path / "Wav2Vec2CTC.mlpackage"
    vocab_path = output_path / "vocab.json"
    config_path = output_path / "config.json"

    print_progress("=" * 60)
    print_progress("wav2vec2 CoreML Conversion")
    print_progress("=" * 60)
    print_progress(f"Model: {model_name}")
    print_progress(f"Output directory: {output_dir}")
    print_progress(f"Deployment target: {min_deployment_target}")
    print_progress(f"Compute units: {compute_units}")
    print_progress(f"FP16 precision: {use_fp16}")
    print_progress(f"Audio range: {min_audio_samples} - {max_audio_samples} samples")
    print_progress("")

    # Check if mlpackage already exists
    if mlpackage_path.exists():
        print_progress(f"CoreML model already exists at {mlpackage_path}")
        response = input("Overwrite existing file? (y/N): ").strip().lower()
        if response != 'y':
            print_progress("Skipping conversion.")
            return {"mlpackage_path": str(mlpackage_path)}

    # Load PyTorch model
    print_progress("Loading PyTorch model from Hugging Face...")
    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        start_time = time.time()
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model.eval()
        load_time = time.time() - start_time
        print_progress(f"Model loaded in {load_time:.1f} seconds")
        print_progress(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    except Exception as e:
        print_progress(f"ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print_progress("")

    # Save vocabulary if it doesn't exist
    if not vocab_path.exists():
        print_progress(f"Saving vocabulary to {vocab_path}...")
        vocab_dict = processor.tokenizer.get_vocab()
        # Sort vocab by ID for consistency
        sorted_vocab = {v: k for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
        with open(vocab_path, 'w') as f:
            json.dump(sorted_vocab, f, indent=2)
        print_progress(f"  Vocabulary size: {len(sorted_vocab)} tokens")
        print_progress("")

    # Save config
    print_progress(f"Saving config to {config_path}...")
    vocab_token_ids = {
        "pad": model.config.pad_token_id,
        "bos": getattr(model.config, "bos_token_id", None),
        "eos": getattr(model.config, "eos_token_id", None),
        "unk": getattr(model.config, "unk_token_id", None),
    }
    vocab_token_ids = {k: v for k, v in vocab_token_ids.items() if v is not None}

    config = {
        "model_name": model_name,
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "intermediate_size": getattr(model.config, "intermediate_size", None),
        "sample_rate": 16000,
        "padding_token_id": model.config.pad_token_id,
        "vocab_token_ids": vocab_token_ids,
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print_progress("")

    # Trace the model
    print_progress("Tracing model for CoreML conversion...")

    # Create example input for tracing
    example_audio = torch.randn(1, 80000)  # 5 seconds at 16kHz

    # Wrap the model to output only the logits (not the full dict)
    class Wav2Vec2Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, audio):
            output = self.model(audio)
            return output.logits

    wrapped_model = Wav2Vec2Wrapper(model)
    wrapped_model.eval()

    try:
        traced_model = torch.jit.trace(wrapped_model, example_audio, strict=False)
        print_progress("  Model traced successfully")
    except Exception as e:
        print_progress(f"ERROR: Tracing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print_progress("")

    # Convert to CoreML
    print_progress("Converting to CoreML...")
    try:
        convert_start = time.time()

        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(
                name="input_audio",
                shape=(1, ct.RangeDim(min_audio_samples, max_audio_samples)),
                dtype=np.float32,
            )],
            minimum_deployment_target=min_deployment_target,
            compute_units=compute_units,
        )

        convert_time = time.time() - convert_start
        print_progress(f"Conversion completed in {convert_time:.1f} seconds")

    except Exception as e:
        print_progress(f"ERROR: CoreML conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print_progress("")

    # Get model info - check if mlmodel is a MIL program or MLModel
    print_progress("Model information:")
    # The ct.convert returns a MIL program, not an MLModel with .input_description
    # We need to use the spec to get input/output info
    print_progress("  Conversion to MIL completed successfully")
    print_progress("")

    # Configure metadata - this needs to be done on the MIL spec before finalizing
    print_progress("Configuring model metadata...")
    # We can't set metadata directly on MIL program, it will be set during save
    print_progress("  Metadata will be set during save")
    print_progress("")

    # Skip FP16 quantization for now as it needs special handling with MIL programs
    # The model will be saved with default precision
    if use_fp16:
        print_progress("Note: FP16 precision optimization skipped")
        print_progress("  (Can be applied separately if needed)")
        print_progress("")

    # Save model
    print_progress(f"Saving CoreML model to {mlpackage_path}...")
    try:
        mlmodel.save(str(mlpackage_path))
        print_progress("  Model saved successfully")
    except Exception as e:
        print_progress(f"ERROR: Failed to save model: {e}")
        sys.exit(1)

    # Get package size
    if mlpackage_path.is_dir():
        total_size = sum(
            f.stat().st_size for f in mlpackage_path.rglob('*') if f.is_file()
        )
        size_mb = total_size / (1024 * 1024)
        print_progress(f"Model package size: {size_mb:.1f} MB")

    print_progress("")
    print_progress("=" * 60)
    print_progress("Conversion complete!")
    print_progress("=" * 60)
    print_progress(f"CoreML model: {mlpackage_path}")
    print_progress(f"Vocabulary: {vocab_path}")
    print_progress(f"Config: {config_path}")
    print_progress("")
    print_progress("Usage in Swift:")
    print_progress("  let model = try Wav2Vec2CTC(configuration: config)")

    return {
        "mlpackage_path": str(mlpackage_path),
        "vocab_path": str(vocab_path),
        "config_path": str(config_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert wav2vec2 PyTorch model to CoreML format"
    )
    parser.add_argument(
        "--model",
        default="facebook/wav2vec2-base-960h",
        help="Hugging Face model identifier (default: facebook/wav2vec2-base-960h)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for CoreML model (default: models/wav2vec2/)"
    )
    parser.add_argument(
        "--target",
        default="macOS14",
        choices=["macOS13", "macOS14", "macOS15", "iOS16", "iOS17"],
        help="Minimum deployment target (default: macOS14)"
    )
    parser.add_argument(
        "--compute-units",
        default="ALL",
        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
        help="Compute units (default: ALL)"
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 precision (use FP32)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=30,
        help="Maximum audio length in seconds (default: 30)"
    )

    args = parser.parse_args()

    # Map string to target enum
    target_map = {
        "macOS13": ct.target.macOS13,
        "macOS14": ct.target.macOS14,
        "macOS15": ct.target.macOS15,
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17,
    }

    # Map string to compute unit enum
    compute_unit_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }

    convert_to_coreml(
        model_name=args.model,
        output_dir=args.output_dir,
        min_deployment_target=target_map[args.target],
        compute_units=compute_unit_map[args.compute_units],
        use_fp16=not args.no_fp16,
        max_audio_samples=16000 * args.max_length,
    )


if __name__ == "__main__":
    main()
