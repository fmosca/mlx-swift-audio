#!/usr/bin/env python3
"""
Export wav2vec2 model to ONNX format for CoreML conversion.

This script downloads the facebook/wav2vec2-base-960h model and exports it
to ONNX format with dynamic axes for variable-length audio input.

Output files:
- wav2vec2_base_960h.onnx: ONNX model file
- vocab.json: Vocabulary mapping for CTC decoding
- config.json: Model configuration

Author: Claude (Anthropic)
Copyright © Anthony DePasquale
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import torch
import torch.onnx
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def print_progress(message: str):
    """Print progress message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def export_onnx(
    model_name: str = "facebook/wav2vec2-base-960h",
    output_dir: str = None,
    opset_version: int = 14,
    verify_export: bool = True,
) -> dict:
    """
    Export wav2vec2 model to ONNX format.

    Args:
        model_name: Hugging Face model identifier
        output_dir: Directory to save ONNX model and vocab
        opset_version: ONNX opset version
        verify_export: Whether to verify the exported model

    Returns:
        Dictionary with paths to exported files
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "wav2vec2"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    onnx_path = output_path / "wav2vec2_base_960h.onnx"
    vocab_path = output_path / "vocab.json"
    config_path = output_path / "config.json"

    print_progress("=" * 60)
    print_progress("wav2vec2 ONNX Export")
    print_progress("=" * 60)
    print_progress(f"Model: {model_name}")
    print_progress(f"Output directory: {output_dir}")
    print_progress(f"ONNX opset version: {opset_version}")
    print_progress("")

    # Check if ONNX file already exists
    if onnx_path.exists():
        print_progress(f"ONNX model already exists at {onnx_path}")
        response = input("Overwrite existing file? (y/N): ").strip().lower()
        if response != 'y':
            print_progress("Skipping export.")
            return {
                "onnx_path": str(onnx_path),
                "vocab_path": str(vocab_path),
                "config_path": str(config_path),
            }

    # Load model and processor
    print_progress("Loading model and processor from Hugging Face...")
    print_progress("(This may take a few minutes on first run)")

    try:
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        processor = Wav2Vec2Processor.from_pretrained(model_name)
    except Exception as e:
        print_progress(f"ERROR: Failed to load model: {e}")
        sys.exit(1)

    print_progress(f"Model loaded successfully")
    print_progress(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print_progress("")

    # Set model to evaluation mode
    model.eval()

    # Save vocabulary
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
    # Wav2Vec2 config doesn't have all token IDs, only what's available
    vocab_token_ids = {
        "pad": model.config.pad_token_id,
        "bos": getattr(model.config, "bos_token_id", None),
        "eos": getattr(model.config, "eos_token_id", None),
        "unk": getattr(model.config, "unk_token_id", None),
    }
    # Filter out None values
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

    # Prepare dummy input for export
    # Use a reasonable length: 5 seconds at 16kHz = 80000 samples
    dummy_audio_length = 80000
    dummy_audio = torch.randn(1, dummy_audio_length)

    print_progress("Exporting model to ONNX...")
    print_progress(f"  Dummy input shape: {tuple(dummy_audio.shape)}")
    print_progress("")

    # Define dynamic axes for variable-length audio
    dynamic_axes = {
        "input_audio": {0: "batch_size", 1: "audio_length"},
        "output_logits": {0: "batch_size", 1: "sequence_length"},
    }

    try:
        start_time = time.time()

        torch.onnx.export(
            model,
            dummy_audio,
            str(onnx_path),
            opset_version=opset_version,
            input_names=["input_audio"],
            output_names=["output_logits"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )

        export_time = time.time() - start_time
        print_progress(f"Export completed in {export_time:.1f} seconds")

    except Exception as e:
        print_progress(f"ERROR: ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Get file size
    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print_progress(f"ONNX model size: {file_size_mb:.1f} MB")
    print_progress("")

    # Verify export
    if verify_export:
        print_progress("Verifying ONNX export...")

        try:
            import onnx
            from onnx import checker, numpy_helper

            # Load and check model
            onnx_model = onnx.load(str(onnx_path))
            checker.check_model(onnx_model)
            print_progress("  ONNX model structure: VALID")

            # Check graph inputs
            graph_input = onnx_model.graph.input[0]
            print_progress(f"  Input: {graph_input.name}")
            for dim in graph_input.type.tensor_type.shape.dim:
                dim_name = dim.dim_param if dim.dim_param else (dim.dim_name if dim.dim_name else "?")
                if dim.dim_value:
                    print_progress(f"    - {dim_name}: {dim.dim_value}")
                else:
                    print_progress(f"    - {dim_name}: dynamic")

            # Check graph outputs
            graph_output = onnx_model.graph.output[0]
            print_progress(f"  Output: {graph_output.name}")

            # Test with onnxruntime
            print_progress("  Testing with ONNX Runtime...")
            import onnxruntime as ort

            ort_session = ort.InferenceSession(str(onnx_path))

            # Test with different input lengths
            test_lengths = [16000, 80000, 160000]  # 1s, 5s, 10s
            for length in test_lengths:
                test_audio = torch.randn(1, length).numpy()
                outputs = ort_session.run(None, {"input_audio": test_audio})
                print_progress(f"    Input length {length}: Output shape {outputs[0].shape}")

            print_progress("  ONNX Runtime inference: SUCCESS")

        except ImportError as e:
            print_progress(f"  WARNING: Verification skipped - {e}")
            print_progress("  Install with: pip install onnx onnxruntime")
        except Exception as e:
            print_progress(f"  WARNING: Verification failed - {e}")
            print_progress("  The model may still work, but verification could not complete.")

    print_progress("")
    print_progress("=" * 60)
    print_progress("Export complete!")
    print_progress("=" * 60)
    print_progress(f"ONNX model: {onnx_path}")
    print_progress(f"Vocabulary: {vocab_path}")
    print_progress(f"Config: {config_path}")

    return {
        "onnx_path": str(onnx_path),
        "vocab_path": str(vocab_path),
        "config_path": str(config_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export wav2vec2 model to ONNX format"
    )
    parser.add_argument(
        "--model",
        default="facebook/wav2vec2-base-960h",
        help="Hugging Face model identifier (default: facebook/wav2vec2-base-960h)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for ONNX model and vocab (default: models/wav2vec2/)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip ONNX verification step"
    )

    args = parser.parse_args()

    export_onnx(
        model_name=args.model,
        output_dir=args.output_dir,
        opset_version=args.opset,
        verify_export=not args.no_verify,
    )


if __name__ == "__main__":
    main()
