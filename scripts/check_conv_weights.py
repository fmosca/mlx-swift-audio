#!/usr/bin/env python3
"""
Check the actual conv1d weight values being used.
"""
import torch
import transformers
import numpy as np

model_id = "facebook/wav2vec2-base-960h"
model = transformers.Wav2Vec2ForCTC.from_pretrained(model_id)

# Get the first conv layer weight
conv0_weight = model.wav2vec2.feature_extractor.conv_layers[0].conv.weight
print("Conv0 weight shape:", conv0_weight.shape)  # [512, 1, 10]
print("\nFirst conv weight (first 10 values):")
print(conv0_weight[0, 0, :].detach().numpy())

# Save for inspection
np.save('/tmp/wav2vec2_debug/conv0_weight.npy', conv0_weight.detach().numpy())
print("\nSaved to /tmp/wav2vec2_debug/conv0_weight.npy")

# Also check what shape it should be after transposition for MLX
# HF: [512, 1, 10] -> MLX: [512, 10, 1]
print("\nAfter transposition (for MLX): [512, 10, 1]")
transposed = conv0_weight.permute(0, 2, 1).detach().numpy()
print("Transposed first 10 values:", transposed[0, :10, 0])

# Let's manually test a conv1d operation
# Input: [batch, inChannels, sequenceLength] for PyTorch
# Input: [batch, sequenceLength, inChannels] for MLX (channels-last)

# Create simple test input
test_input = torch.randn(1, 1, 100)  # PyTorch format [batch, channels, seq]
print("\nTest input shape (PyTorch format):", test_input.shape)

# Get conv layer
conv_layer = model.wav2vec2.feature_extractor.conv_layers[0].conv

# Run conv
with torch.no_grad():
    pytorch_output = conv_layer(test_input)

print("PyTorch conv output shape:", pytorch_output.shape)  # [1, 512, outSeq]
print("PyTorch conv output first 10 values of first channel:", pytorch_output[0, 0, :10].detach().numpy())

# Now let's test what MLX would do with the transposed weight
# MLX conv1d expects:
# - Input: [batch, seqLen, inChannels]
# - Weight: [outChannels, kernelSize, inChannels]
# - Output: [batch, outSeqLen, outChannels]

# The transposed weight for MLX would be [512, 10, 1]
# In MLX, this means 512 output channels, kernel size 10, 1 input channel

# But wait - MLX conv1d might interpret the weight differently!
# Let me check the actual conv1d documentation

print("\n" + "=" * 70)
print("Weight Format Analysis")
print("=" * 70)
print("PyTorch Conv1d weight format: [outChannels, inChannels/groups, kernelSize]")
print("  Our weight: [512, 1, 10]")
print("")
print("MLX Conv1d weight format (from init): [outChannels, kernelSize, inChannels/groups]")
print("  Transposed: [512, 10, 1]")
print("")
print("This transposition should be correct for MLX.")
