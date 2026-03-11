#!/usr/bin/env python3
"""
Numerical verification of wav2vec2 MLX Swift port vs PyTorch reference.

This script runs the PyTorch wav2vec2 model on test audio and saves
intermediate outputs for numerical comparison with the MLX Swift implementation.
"""

import torch
import transformers
import numpy as np
import json
import sys
from pathlib import Path

def create_test_audio(sample_rate=16000, duration=1.0):
    """Create simple test audio (440Hz sine wave)."""
    t = torch.linspace(0, duration, int(sample_rate * duration))
    audio = torch.sin(2 * np.pi * 440 * t) * 0.5
    return audio.unsqueeze(0)  # [1, samples]

def analyze_frame_log_probs(log_probs, tokenizer, vocab_size=32):
    """Analyze frame-level log probability distribution."""
    # log_probs shape: [batch, frames, vocab_size]
    probs = torch.exp(log_probs[0])  # [frames, vocab_size]

    # For each frame, analyze the probability distribution
    frame_analysis = []
    for frame_idx in range(min(10, probs.shape[0])):  # First 10 frames
        frame_probs = probs[frame_idx]

        # Find top tokens
        top_probs, top_ids = torch.topk(frame_probs, min(5, vocab_size))

        # Calculate distribution statistics
        entropy = -(frame_probs * torch.log(frame_probs + 1e-10)).sum().item()
        max_prob = top_probs[0].item()
        blank_prob = frame_probs[0].item()  # blank is typically token 0

        frame_info = {
            'frame': frame_idx,
            'entropy': entropy,
            'max_prob': max_prob,
            'blank_prob': blank_prob,
            'top_tokens': [
                {'id': top_ids[i].item(), 'prob': top_probs[i].item()}
                for i in range(len(top_probs))
            ]
        }
        frame_analysis.append(frame_info)

    return frame_analysis

def main():
    print("=" * 70)
    print("wav2vec2 Numerical Verification (PyTorch Reference)")
    print("=" * 70)

    # Load model
    model_id = "facebook/wav2vec2-base-960h"
    print(f"\nLoading model: {model_id}")

    device = torch.device("cpu")  # Use CPU for numerical consistency
    model = transformers.Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
    model.eval()

    tokenizer = transformers.Wav2Vec2CTCTokenizer.from_pretrained(model_id)
    vocab_size = model.config.vocab_size

    print(f"  Vocab size: {vocab_size}")
    print(f"  Model loaded successfully")

    # Create test audio
    print("\nCreating test audio (440Hz sine wave, 1 second @ 16kHz)")
    audio = create_test_audio()
    print(f"  Audio shape: {audio.shape}")

    # Run forward pass with hooks for intermediate outputs
    print("\nRunning forward pass...")

    outputs = {}

    def make_hook(name):
        def hook(module, input, output):
            # Handle different output types
            if hasattr(output, 'last_hidden_state'):
                # BaseModelOutput or similar
                tensor = output.last_hidden_state
            elif isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            if isinstance(tensor, torch.Tensor):
                outputs[name] = tensor.detach().cpu()
        return hook

    # Register hooks for key intermediate outputs
    hooks = []

    # Feature extractor output
    hooks.append(model.wav2vec2.feature_extractor.conv_layers[-1].register_forward_hook(
        make_hook('feature_extractor_last_conv')
    ))

    # Positional conv embedding (weight_v result)
    hooks.append(model.wav2vec2.encoder.pos_conv_embed.conv.register_forward_hook(
        make_hook('positional_conv')
    ))

    # After first encoder layer
    hooks.append(model.wav2vec2.encoder.layers[0].register_forward_hook(
        make_hook('encoder_layer_0')
    ))

    # After all encoder layers
    hooks.append(model.wav2vec2.encoder.register_forward_hook(
        make_hook('encoder_output')
    ))

    # Run forward pass
    with torch.no_grad():
        result = model(audio)

    logits = result.logits  # [1, frames, vocab_size]
    log_probs = torch.log_softmax(logits, dim=-1)

    print(f"  Logits shape: {logits.shape}")
    print(f"  Frames: {logits.shape[1]}")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Analyze frame-level log probabilities
    print("\n" + "=" * 70)
    print("Frame-Level Log Probability Analysis")
    print("=" * 70)

    frame_analysis = analyze_frame_log_probs(log_probs, tokenizer, vocab_size)

    print("\nFirst 10 frames:")
    print(f"{'Frame':<8} {'Entropy':<12} {'Max Prob':<12} {'Blank Prob':<12} {'Top Token':<15} {'Prob'}")
    print("-" * 90)
    for info in frame_analysis:
        top_token = tokenizer.decode([info['top_tokens'][0]['id']])
        top_prob = info['top_tokens'][0]['prob']
        print(f"{info['frame']:<8} {info['entropy']:<12.4f} {info['max_prob']:<12.4f} "
              f"{info['blank_prob']:<12.4f} {top_token:<15} {top_prob:.4f}")

    # Calculate aggregate statistics
    avg_entropy = np.mean([f['entropy'] for f in frame_analysis])
    avg_max_prob = np.mean([f['max_prob'] for f in frame_analysis])
    avg_blank_prob = np.mean([f['blank_prob'] for f in frame_analysis])

    print(f"\nAggregate statistics (first 10 frames):")
    print(f"  Average entropy: {avg_entropy:.4f}")
    print(f"  Average max probability: {avg_max_prob:.4f}")
    print(f"  Average blank probability: {avg_blank_prob:.4f}")

    if avg_blank_prob > 0.8:
        print("  ⚠️  WARNING: High blank probability suggests model is over-predicting blanks")
    if avg_max_prob > 0.95:
        print("  ⚠️  WARNING: Very peaky distribution suggests low confidence in non-blank tokens")

    # Save intermediate outputs for numerical comparison
    output_dir = Path("/tmp/wav2vec2_numerical")
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("Saving Intermediate Outputs")
    print("=" * 70)

    # Save each intermediate output as numpy array
    for name, tensor in outputs.items():
        np.save(output_dir / f"{name}.npy", tensor.numpy())
        print(f"  {name}: {tensor.shape} -> {name}.npy")

    # Save logits and log_probs
    np.save(output_dir / "logits.npy", logits.numpy())
    np.save(output_dir / "log_probs.npy", log_probs.numpy())
    print(f"  logits: {logits.shape} -> logits.npy")
    print(f"  log_probs: {log_probs.shape} -> log_probs.npy")

    # Save frame analysis as JSON
    with open(output_dir / "frame_analysis.json", 'w') as f:
        json.dump({
            'frame_analysis': frame_analysis,
            'aggregate_stats': {
                'avg_entropy': float(avg_entropy),
                'avg_max_prob': float(avg_max_prob),
                'avg_blank_prob': float(avg_blank_prob)
            }
        }, f, indent=2)
    print(f"  frame_analysis.json")

    # Save model weights for key components
    print("\n" + "=" * 70)
    print("Saving Key Model Weights")
    print("=" * 70)

    # Positional conv (ParametrizedConv1d stores weights differently)
    pos_conv = model.wav2vec2.encoder.pos_conv_embed.conv
    weights_to_save = {
        # Feature extractor conv layers (last one)
        'feature_extractor_last_conv_weight': model.wav2vec2.feature_extractor.conv_layers[-1].conv.weight,
        'feature_extractor_last_conv_bias': model.wav2vec2.feature_extractor.conv_layers[-1].conv.bias,

        # Positional conv (access via parametrization)
        'pos_conv_weight': pos_conv.weight if hasattr(pos_conv, 'weight') else None,
        'pos_conv_bias': pos_conv.bias if hasattr(pos_conv, 'bias') else None,

        # First encoder layer attention
        'encoder_layer_0_q_proj_weight': model.wav2vec2.encoder.layers[0].attention.q_proj.weight,
        'encoder_layer_0_k_proj_weight': model.wav2vec2.encoder.layers[0].attention.k_proj.weight,
        'encoder_layer_0_v_proj_weight': model.wav2vec2.encoder.layers[0].attention.v_proj.weight,
        'encoder_layer_0_out_proj_weight': model.wav2vec2.encoder.layers[0].attention.out_proj.weight,

        # First encoder layer feed-forward
        'encoder_layer_0_intermediate_dense_weight': model.wav2vec2.encoder.layers[0].feed_forward.intermediate_dense.weight,
        'encoder_layer_0_output_dense_weight': model.wav2vec2.encoder.layers[0].feed_forward.output_dense.weight,

        # Layer norm
        'encoder_layer_0_layer_norm_weight': model.wav2vec2.encoder.layers[0].layer_norm.weight,
        'encoder_layer_0_layer_norm_bias': model.wav2vec2.encoder.layers[0].layer_norm.bias,

        # CTC head
        'lm_head_weight': model.lm_head.weight if hasattr(model, 'lm_head') else None,
        'lm_head_bias': model.lm_head.bias if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'bias') else None,
    }

    for name, weight in weights_to_save.items():
        if weight is not None:
            np.save(output_dir / f"weight_{name}.npy", weight.detach().cpu().numpy())
            print(f"  {name}: {weight.shape} -> weight_{name}.npy")

    print(f"\nAll outputs saved to: {output_dir}")
    print("\nTo compare with MLX Swift, run:")
    print(f"  cd {output_dir}")
    print(f"  python -c \"import numpy as np; print(np.load('feature_extractor_last_conv.npy').shape)\"")

if __name__ == "__main__":
    main()
