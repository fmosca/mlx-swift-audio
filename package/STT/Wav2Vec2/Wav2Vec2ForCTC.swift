// Copyright © 2021 Facebook AI Research (original model implementation)
// Ported to MLX from https://github.com/huggingface/transformers
// Copyright © Anthony DePasquale
// License: licenses/apache-2.0.txt

import Foundation
import MLX
import MLXNN

/// Wav2Vec2 model for CTC (Connectionist Temporal Classification) speech recognition.
///
/// This model combines the feature extractor, encoder, and CTC output head
/// to produce log-softmax probabilities over the vocabulary for each time step.
///
/// Architecture:
/// 1. Feature Extractor: Conv layers that downsample audio by ~320x
/// 2. Encoder: Transformer encoder with positional embeddings
/// 3. CTC Head: Linear projection + log_softmax for token probabilities
///
/// Input: [batch, samples] - Raw audio waveform
/// Output: [batch, frames, vocabSize] - Log probabilities for each token at each frame
public class Wav2Vec2ForCTC: Module {
  @ModuleInfo var featureExtractor: Wav2Vec2FeatureExtractor
  @ModuleInfo var encoder: Wav2Vec2Encoder
  @ModuleInfo(key: "lm_head") var ctcHead: Linear
  @ModuleInfo var dropout: Dropout

  public let config: Wav2Vec2Config

  init(config: Wav2Vec2Config) {
    self.config = config

    _featureExtractor.wrappedValue = Wav2Vec2FeatureExtractor(config: config)
    _encoder.wrappedValue = Wav2Vec2Encoder(config: config)
    _ctcHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize)
    _dropout.wrappedValue = Dropout(p: Float(config.finalDropout))
  }

  /// Forward pass through the model.
  ///
  /// - Parameter audio: Input audio tensor of shape [batch, samples]
  /// - Returns: Log probabilities tensor of shape [batch, frames, vocabSize]
  public func callAsFunction(_ audio: MLXArray) -> MLXArray {
    // Step 1: Extract features from raw audio
    // Input: [batch, samples]
    // Output: [batch, frames, 512]
    var hiddenStates = featureExtractor(audio)

    // Step 2: Pass through encoder
    // Input: [batch, frames, 512]
    // Output: [batch, frames, hiddenSize]
    hiddenStates = encoder(hiddenStates)

    // Step 4: Apply final dropout if configured
    hiddenStates = dropout(hiddenStates)

    // Step 5: Project to vocabulary size through CTC head
    // Output: [batch, frames, vocabSize]
    let logits = ctcHead(hiddenStates)

    // Step 6: Apply log_softmax for CTC loss computation
    // Returns log probabilities over vocabulary dimension
    return logSoftmax(logits, axis: -1)
  }

  /// Calculate the number of output frames for a given number of input samples.
  ///
  /// This accounts for the downsampling performed by the convolutional
  /// feature extractor layers.
  ///
  /// - Parameter samples: Number of audio samples (e.g., 16000 for 1 second at 16kHz)
  /// - Returns: Number of output frames from the model
  public func outputFrames(forSamples samples: Int) -> Int {
    let totalStride = config.convStride.reduce(1, *)
    // Account for kernel sizes in conv layers (approximately)
    return max(1, (samples - 1) / totalStride)
  }
}
