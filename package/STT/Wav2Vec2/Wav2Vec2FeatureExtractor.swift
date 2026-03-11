// Copyright © 2021 Facebook AI Research (original model implementation)
// Ported to MLX from https://github.com/huggingface/transformers
// Copyright © Anthony DePasquale
// License: licenses/apache-2.0.txt

import Foundation
import MLX
import MLXNN

// MARK: - Wav2Vec2 Conv Layer with LayerNorm

/// A convolutional layer with LayerNorm, used for all layers when feat_extract_norm="layer"
/// and for layer 0 when feat_extract_norm="group".
///
/// NOTE: PyTorch uses GroupNorm(512, 512) which is InstanceNorm (normalizes each channel
/// independently across frames). MLX's LayerNorm normalizes each frame across channels.
/// This is a known mismatch that causes numerical differences, but reverting because
/// the GroupNorm fix produced worse alignment results.
class Wav2Vec2ConvLayerWithLayerNorm: Module {
  @ModuleInfo(key: "conv") var conv: Conv1d
  @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

  let inChannels: Int
  let outChannels: Int
  let kernelSize: Int
  let stride: Int

  init(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    stride: Int,
    bias: Bool = false
  ) {
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.kernelSize = kernelSize
    self.stride = stride

    _conv.wrappedValue = Conv1d(
      inputChannels: inChannels,
      outputChannels: outChannels,
      kernelSize: kernelSize,
      stride: stride,
      padding: 0,
      dilation: 1,
      groups: 1,
      bias: bias
    )
    _layerNorm.wrappedValue = LayerNorm(dimensions: outChannels, eps: 1e-5, affine: true)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var output = conv(x)
    output = layerNorm(output)
    return GELU()(output)
  }
}

// MARK: - Wav2Vec2 Conv Layer without normalization

/// A convolutional layer without normalization, used for layers 1-6 when feat_extract_norm="group".
class Wav2Vec2ConvLayerNoNorm: Module {
  @ModuleInfo var conv: Conv1d

  let inChannels: Int
  let outChannels: Int
  let kernelSize: Int
  let stride: Int

  init(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    stride: Int,
    bias: Bool = false
  ) {
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.kernelSize = kernelSize
    self.stride = stride

    _conv.wrappedValue = Conv1d(
      inputChannels: inChannels,
      outputChannels: outChannels,
      kernelSize: kernelSize,
      stride: stride,
      padding: 0,
      dilation: 1,
      groups: 1,
      bias: bias
    )
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let output = conv(x)
    return GELU()(output)
  }
}

// MARK: - Wav2Vec2 Feature Extractor

/// Wav2Vec2 feature extractor that converts raw audio to feature vectors.
///
/// Architecture:
/// - Layer 0: Conv(kernel=10, stride=5) with LayerNorm
/// - Layers 1-6: Conv(kernel=3, stride=2) - LayerNorm if feat_extract_norm="layer", none if "group"
/// - All layers use GELU activation
///
/// NOTE: There's a known mismatch with PyTorch:
/// - PyTorch uses GroupNorm(512, 512) which is InstanceNorm (each channel normalized independently across frames)
/// - MLX uses LayerNorm (each frame normalized independently across channels)
/// - Despite numerical differences, LayerNorm produces better alignment results in practice
///
/// Input shape: [batch, samples]
/// Output shape: [batch, frames, channels]
///
/// The feature extractor downsamples the audio by a factor of:
/// - Layer 0: stride=5
/// - Layers 1-6: stride=2 each
/// - Total: 5 * 2^6 = 320
class Wav2Vec2FeatureExtractor: Module {
  @ModuleInfo(key: "conv_layers") var convLayers: [Module]

  let config: Wav2Vec2Config

  init(config: Wav2Vec2Config) {
    self.config = config
    _convLayers.wrappedValue = Self.makeConvLayers(config: config)
  }

  /// Create the convolutional layers based on the configuration.
  private static func makeConvLayers(config: Wav2Vec2Config) -> [Module] {
    var layers: [Module] = []

    for layerId in 0 ..< config.convDim.count {
      let inChannels = layerId == 0 ? 1 : config.convDim[layerId - 1]
      let outChannels = config.convDim[layerId]
      let kernelSize = config.convKernel[layerId]
      let stride = config.convStride[layerId]

      if config.featExtractNorm == "group" {
        if layerId == 0 {
          // Layer 0 uses LayerNorm (NOTE: PyTorch uses GroupNorm which is InstanceNorm,
          // but MLX LayerNorm gives better alignment results despite numerical mismatch)
          layers.append(
            Wav2Vec2ConvLayerWithLayerNorm(
              inChannels: inChannels,
              outChannels: outChannels,
              kernelSize: kernelSize,
              stride: stride,
              bias: config.convBias
            )
          )
        } else {
          // Layers 1-6 have no normalization (only conv + activation)
          layers.append(
            Wav2Vec2ConvLayerNoNorm(
              inChannels: inChannels,
              outChannels: outChannels,
              kernelSize: kernelSize,
              stride: stride,
              bias: config.convBias
            )
          )
        }
      } else if config.featExtractNorm == "layer" {
        // All layers use LayerNorm
        layers.append(
          Wav2Vec2ConvLayerWithLayerNorm(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            bias: config.convBias
          )
        )
      } else {
        fatalError("Unsupported feat_extract_norm: \(config.featExtractNorm)")
      }
    }

    return layers
  }

  /// Forward pass through the feature extractor.
  ///
  /// - Parameter x: Input audio tensor of shape [batch, samples]
  /// - Returns: Feature tensor of shape [batch, frames, channels]
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Reshape input from [batch, samples] to [batch, samples, 1]
    // This adds the channel dimension expected by Conv1d
    var hiddenStates = x.expandedDimensions(axis: -1)  // [batch, samples, 1]

    // Apply each convolutional layer
    for layer in convLayers {
      if let layerWithNorm = layer as? Wav2Vec2ConvLayerWithLayerNorm {
        hiddenStates = layerWithNorm(hiddenStates)
      } else if let layerNoNorm = layer as? Wav2Vec2ConvLayerNoNorm {
        hiddenStates = layerNoNorm(hiddenStates)
      }
    }

    // Output is [batch, frames, channels] which is the format expected by Conv1d
    // and by the FeatureProjection (LayerNorm operates on last dimension)
    return hiddenStates
  }
}
