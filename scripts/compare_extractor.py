#!/usr/bin/env python3
"""
Quick check of PyTorch feature extractor output to compare signs.
"""
import torch
import transformers
import numpy as np

model_id = "facebook/wav2vec2-base-960h"
model = transformers.Wav2Vec2ForCTC.from_pretrained(model_id)
model.eval()

# Create test audio (440Hz sine wave, 1 second @ 16kHz)
sample_rate = 16000
duration = 1.0
import numpy as np
t = np.linspace(0, duration, int(sample_rate * duration))
audio = np.sin(2 * np.pi * 440 * t) * 0.5
audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()

# Get feature extractor output
with torch.no_grad():
    hidden_states = model.wav2vec2.feature_extractor(audio_tensor)

# Output is [batch, channels, frames]
print(f"PyTorch feature extractor output shape: {hidden_states.shape}")

# Get first channel, first frame values
first_channel_first_frame = hidden_states[0, 0, :].numpy()
print(f"PyTorch first 10 values of first channel: {first_channel_first_frame[:10]}")
print(f"Min/Max of first channel: {hidden_states[0, 0, :].min().item():.6f} / {hidden_states[0, 0, :].max().item():.6f}")
