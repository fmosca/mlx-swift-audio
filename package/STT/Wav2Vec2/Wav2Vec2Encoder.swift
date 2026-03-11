// Copyright © 2021 Facebook AI Research (original model implementation)
// Ported to MLX from https://github.com/huggingface/transformers
// Copyright © Anthony DePasquale
// License: licenses/apache-2.0.txt

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Weight Normalization Utilities

/// Computes the p-norm of a tensor along specified dimensions.
fileprivate func computeNorm(
  x: MLXArray,
  p: Int,
  dim: [Int]? = nil,
  keepdim: Bool = false
) -> MLXArray {
  guard p == 1 || p == 2 else {
    fatalError("Only p-norms with p of 1 or 2 are supported")
  }

  let dimensions: [Int] = if let dim {
    dim
  } else {
    Array(0 ..< x.ndim)
  }

  if p == 1 {
    // L1 norm
    return MLX.sum(MLX.abs(x), axes: dimensions, keepDims: keepdim)
  } else {
    // L2 norm
    return MLX.sqrt(MLX.sum(x * x, axes: dimensions, keepDims: keepdim))
  }
}

/// Applies weight normalization to a tensor.
///
/// Weight normalization reparameterizes weight vectors (or tensors) in terms of
/// their magnitude and direction. This is commonly used in convolutional layers.
///
/// - Parameters:
///   - weightV: The direction (v) parameter
///   - weightG: The magnitude (g) parameter
///   - dim: The dimension(s) over which to compute the norm (nil = all dims)
/// - Returns: The normalized weight tensor
fileprivate func weightNorm(
  weightV: MLXArray,
  weightG: MLXArray,
  dim: Int? = nil
) -> MLXArray {
  let rank = weightV.shape.count

  var axes: [Int]

  if let dim {
    var adjustedDim = dim
    if dim < 0 {
      adjustedDim += rank
    }

    axes = Array(0 ..< rank)
    if adjustedDim != -1 {
      axes.removeAll(where: { $0 == adjustedDim })
    }
  } else {
    axes = Array(0 ..< rank)
  }

  let normV = computeNorm(x: weightV, p: 2, dim: axes, keepdim: true)

  let normalizedWeight = weightV / (normV + 1e-7) // Add epsilon for numerical stability
  return normalizedWeight * weightG
}

/// Conv1d with weight normalization for Wav2Vec2.
///
/// This class implements a 1D convolution with weight normalization,
/// which is used in the Wav2Vec2 positional convolutional embedding.
///
/// Weight format:
/// - For groups=1: weight_v has shape [outChannels, inChannels, kernelSize]
/// - For groups>1: weight_v has shape [outChannels, inChannels/groups, kernelSize]
/// This matches MLX conv1d's expected weight format.
class Wav2Vec2ConvWeighted: Module {
  @ParameterInfo(key: "weight_g") var weightG: MLXArray
  @ParameterInfo(key: "weight_v") var weightV: MLXArray
  @ParameterInfo var bias: MLXArray?

  let stride: Int
  let padding: Int
  let dilation: Int
  let groups: Int
  let inChannelsPerGroup: Int

  init(
    inChannels: Int = 0,
    outChannels: Int = 0,
    kernelSize: Int = 1,
    stride: Int = 1,
    padding: Int = 1,
    dilation: Int = 1,
    groups: Int = 1,
    bias: Bool = false
  ) {
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
    self.inChannelsPerGroup = inChannels / groups

    // Initialize with zeros - will be replaced by weight loading
    // HF provides weight_g as [1, 1, kernelSize] (per-kernel-position gain)
    // MLX conv1d weight format: [outChannels, inChannels, kernelSize] for groups=1
    //                        or [outChannels, inChannels/groups, kernelSize] for groups>1
    _weightG.wrappedValue = MLXArray.zeros([1, 1, kernelSize])
    // MLX Python conv1d weight format: [outChannels, kernelSize, inChannels/groups]
    _weightV.wrappedValue = MLXArray.zeros([outChannels, kernelSize, inChannelsPerGroup])
    _bias.wrappedValue = bias ? MLXArray.zeros([outChannels]) : nil
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Apply weight normalization along dim=1 (kernelSize dimension)
    // This computes the norm over [outChannels, inChannels/groups] dimensions
    // Weight format is [outChannels, kernelSize, inChannels/groups] after weight loading
    // weightG format from HF is [1, 1, kernelSize] - per-kernel-position gain shared across all channels
    let outChannels = weightV.shape[0]
    let kernelSize = weightV.shape[1]
    // weightG is [1, 1, kernelSize] = [1, 1, 128], need to expand to [outChannels, kernelSize, 1] = [768, 128, 1]
    // Extract kernel size from weightG shape (dim 2) and expand
    let weightGKernelSize = weightG.shape[2]
    let weightGExpanded = MLX.repeated(weightG.reshaped([weightGKernelSize]).expandedDimensions(axis: 0), count: outChannels, axis: 0).reshaped([outChannels, kernelSize, 1])
    let weight = weightNorm(weightV: weightV, weightG: weightGExpanded, dim: 1)

    // For groups > 1, MLX Swift's conv1d has limited support
    // Implement grouped convolution manually by splitting input and weights
    if groups == 1 {
      // Standard convolution
      var result = MLX.conv1d(
        x,
        weight,
        stride: stride,
        padding: padding,
        dilation: dilation,
        groups: 1
      )

      // Add bias if present
      if let biasValue = bias {
        result = result + biasValue.reshaped([1, -1, 1])
      }

      return result
    } else {
      // Manual grouped convolution implementation
      // Weight format is [outChannels, kernelSize, inChannels/groups]
      let batchSize = x.shape[0]
      let inChannels = x.shape[1]
      let seqLength = x.shape[2]
      let inChannelsPerGroup = inChannels / groups

      // Reshape input to split channels: [batch, inChannels, seq] -> [batch, groups, inChannels/groups, seq]
      let xReshaped = x.reshaped([batchSize, groups, inChannelsPerGroup, seqLength])

      // Weight format: [outChannels, kernelSize, inChannels/groups]
      let outChannels = weight.shape[0]
      let kernelSize = weight.shape[1]
      let outChannelsPerGroup = outChannels / groups

      // Reshape weight: [outChannels, kernelSize, inChannels/groups] -> [groups, outChannels/groups, kernelSize, inChannels/groups]
      let weightReshaped = weight.reshaped([groups, outChannelsPerGroup, kernelSize, inChannelsPerGroup])

      // Process each group separately
      var groupOutputs: [MLXArray] = []
      for g in 0..<groups {
        // Get input for this group: [batch, inChannels/groups, seq]
        let xGroup = xReshaped[0..<batchSize, g..<g+1, 0..<inChannelsPerGroup, 0..<seqLength]
        let xGroupSqueezed = xGroup.reshaped([batchSize, inChannelsPerGroup, seqLength])

        // Get weight for this group: [outChannels/groups, kernelSize, inChannels/groups]
        let wGroup = weightReshaped[g..<g+1, 0..<outChannelsPerGroup, 0..<kernelSize, 0..<inChannelsPerGroup]
        let wGroupSqueezed = wGroup.reshaped([outChannelsPerGroup, kernelSize, inChannelsPerGroup])

        // MLX conv1d expects input in [batch, seq, inChannels] format (NLC)
        // Our input is [batch, inChannels, seq], so transpose
        let xGroupTransposed = xGroupSqueezed.transposed(0, 2, 1)

        // MLX conv1d expects weight in [outChannels, kernelSize, inChannels] format
        // wGroupSqueezed is already in this format: [outChannelsPerGroup, kernelSize, inChannelsPerGroup]

        // Apply convolution
        var groupResult = MLX.conv1d(
          xGroupTransposed,
          wGroupSqueezed,
          stride: stride,
          padding: padding,
          dilation: dilation,
          groups: 1
        )

        // MLX conv1d outputs [batch, seq, outChannels], transpose back to [batch, outChannels, seq]
        groupResult = groupResult.transposed(0, 2, 1)

        // Add bias portion if present
        if let biasValue = bias {
          let biasStart = g * outChannelsPerGroup
          let biasEnd = biasStart + outChannelsPerGroup
          let biasSlice = biasValue[biasStart..<biasEnd]
          groupResult = groupResult + biasSlice.reshaped([1, -1, 1])
        }

        groupOutputs.append(groupResult)
      }

      // Concatenate outputs along channel dimension: [groups] * [batch, outChannels/groups, seq] -> [batch, outChannels, seq]
      let concatenated = MLX.concatenated(groupOutputs, axis: 1)
      return concatenated
    }
  }
}

// MARK: - Wav2Vec2 Positional Convolutional Embedding

/// Convolutional positional encoding for Wav2Vec2 encoder.
///
/// Uses Conv1d with weight normalization (kernel=128, groups=16) followed by GELU activation.
/// This provides positional information to the transformer encoder.
///
/// Input/Output shapes:
/// - Input: [batch, channels, sequenceLength]
/// - Output: [batch, sequenceLength, channels]
class Wav2Vec2PositionalConvEmbedding: Module {
  @ModuleInfo var conv: Wav2Vec2ConvWeighted
  let numChannels: Int

  init(
    embeddingSize: Int = 768,
    numEmbeddings: Int = 128,
    numGroups: Int = 16
  ) {
    self.numChannels = embeddingSize

    // Wav2Vec2ConvWeighted expects MLX conv1d weight format:
    // [outChannels, inChannels/groups, kernelSize]
    // Kernel size is numEmbeddings (128)
    // Padding is set to preserve sequence length with stride=1
    _conv.wrappedValue = Wav2Vec2ConvWeighted(
      inChannels: embeddingSize,
      outChannels: embeddingSize,
      kernelSize: numEmbeddings,
      stride: 1,
      padding: (numEmbeddings - 1) / 2,  // Center the convolution
      groups: numGroups,
      bias: true
    )
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // x shape: [batch, channels, sequenceLength]
    let batchSize = x.shape[0]
    let channels = x.shape[1]
    let inputLength = x.shape[2]

    // Debug: verify input shape
    if channels != self.numChannels {
      fatalError("Wav2Vec2PositionalConvEmbedding: Expected \(self.numChannels) input channels but got \(channels). Input shape: \(x.shape)")
    }

    // Apply convolution with weight normalization
    // With padding=(kernel-1)/2 and stride=1, output length = input length
    var convOut = conv(x)

    // Ensure output sequence length matches input length
    // (handle any edge cases with rounding)
    let outputLength = convOut.shape[2]
    if outputLength < inputLength {
      // Pad right side to restore original sequence length
      let padAmount = inputLength - outputLength
      let zeroPadding = MLXArray.zeros([batchSize, channels, padAmount])
      convOut = MLX.concatenated([convOut, zeroPadding], axis: 2)
    } else if outputLength > inputLength {
      // Truncate if output is longer (shouldn't happen with correct padding)
      convOut = convOut[0..<batchSize, 0..<channels, 0..<inputLength]
    }

    // Apply GELU activation
    let activated = GELU()(convOut)

    // Transpose to [batch, sequenceLength, channels] for output
    return activated.transposed(0, 2, 1)
  }
}

// MARK: - Wav2Vec2 Feed Forward

/// Feed-forward network for Wav2Vec2 encoder layer.
///
/// Two-layer MLP with GELU activation and dropout.
/// Expands hidden size by 4x in intermediate layer.
class Wav2Vec2FeedForward: Module {
  @ModuleInfo(key: "intermediate_dense") var intermediateDense: Linear
  @ModuleInfo(key: "output_dense") var outputDense: Linear
  @ModuleInfo var dropout: Dropout

  let intermediateSize: Int
  let hiddenSize: Int

  init(
    intermediateSize: Int,
    hiddenSize: Int,
    hiddenDropout: Double = 0.1
  ) {
    self.intermediateSize = intermediateSize
    self.hiddenSize = hiddenSize

    _intermediateDense.wrappedValue = Linear(hiddenSize, intermediateSize)
    _outputDense.wrappedValue = Linear(intermediateSize, hiddenSize)
    _dropout.wrappedValue = Dropout(p: Float(hiddenDropout))
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var output = intermediateDense(x)
    output = GELU()(output)
    output = outputDense(output)
    output = dropout(output)
    return output
  }
}

// MARK: - Wav2Vec2 Attention

/// Multi-head self-attention for Wav2Vec2 encoder.
///
/// Uses MLXFast.scaledDotProductAttention for efficient computation.
/// Supports Q, K, V projections and output projection with dropout.
class Wav2Vec2Attention: Module {
  @ModuleInfo(key: "q_proj") var qProj: Linear
  @ModuleInfo(key: "k_proj") var kProj: Linear
  @ModuleInfo(key: "v_proj") var vProj: Linear
  @ModuleInfo(key: "out_proj") var outProj: Linear
  @ModuleInfo var dropout: Dropout

  let embedDim: Int
  let numHeads: Int
  let headDim: Int

  init(
    embedDim: Int,
    numHeads: Int,
    attentionDropout: Double = 0.1
  ) {
    self.embedDim = embedDim
    self.numHeads = numHeads
    self.headDim = embedDim / numHeads

    _qProj.wrappedValue = Linear(embedDim, embedDim)
    _kProj.wrappedValue = Linear(embedDim, embedDim)
    _vProj.wrappedValue = Linear(embedDim, embedDim)
    _outProj.wrappedValue = Linear(embedDim, embedDim)
    _dropout.wrappedValue = Dropout(p: Float(attentionDropout))
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    attentionMask: MLXArray? = nil
  ) -> MLXArray {
    let batchSize = hiddenStates.shape[0]
    let tgtLen = hiddenStates.shape[1]

    // Project to Q, K, V
    var q = qProj(hiddenStates)
    var k = kProj(hiddenStates)
    var v = vProj(hiddenStates)

    // Reshape for multi-head attention
    // [batch, seqLen, embedDim] -> [batch, seqLen, numHeads, headDim]
    q = q.reshaped(batchSize, tgtLen, numHeads, headDim)
    k = k.reshaped(batchSize, tgtLen, numHeads, headDim)
    v = v.reshaped(batchSize, tgtLen, numHeads, headDim)

    // Transpose to [batch, numHeads, seqLen, headDim]
    q = q.transposed(0, 2, 1, 3)
    k = k.transposed(0, 2, 1, 3)
    v = v.transposed(0, 2, 1, 3)

    // Scale factor for attention
    let scale = 1.0 / Float(headDim).squareRoot()

    // Apply attention using MLXFast
    var attnOutput = MLXFast.scaledDotProductAttention(
      queries: q,
      keys: k,
      values: v,
      scale: scale,
      mask: .none
    )

    // Transpose back: [batch, seqLen, numHeads, headDim]
    attnOutput = attnOutput.transposed(0, 2, 1, 3)

    // Reshape to [batch, seqLen, embedDim]
    attnOutput = attnOutput.reshaped(batchSize, tgtLen, embedDim)

    // Output projection
    attnOutput = outProj(attnOutput)
    attnOutput = dropout(attnOutput)

    return attnOutput
  }
}

// MARK: - Wav2Vec2 Encoder Layer

/// Single transformer encoder layer for Wav2Vec2.
///
/// Uses post-layer normalization (standard wav2vec2 architecture):
/// - attention -> residual -> layerNorm -> ffn -> residual -> layerNorm
///
/// This differs from pre-norm architectures (like some newer transformers)
/// where layer normalization is applied before the attention/ffn.
class Wav2Vec2EncoderLayer: Module {
  @ModuleInfo(key: "attention") var attention: Wav2Vec2Attention
  @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
  @ModuleInfo(key: "feed_forward") var feedForward: Wav2Vec2FeedForward
  @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm
  @ModuleInfo var dropout: Dropout

  init(config: Wav2Vec2Config) {
    _attention.wrappedValue = Wav2Vec2Attention(
      embedDim: config.hiddenSize,
      numHeads: config.numAttentionHeads,
      attentionDropout: config.attentionDropout
    )
    _layerNorm.wrappedValue = LayerNorm(
      dimensions: config.hiddenSize,
      eps: Float(config.layerNormEps)
    )
    _feedForward.wrappedValue = Wav2Vec2FeedForward(
      intermediateSize: config.intermediateSize,
      hiddenSize: config.hiddenSize,
      hiddenDropout: config.hiddenDropout
    )
    _finalLayerNorm.wrappedValue = LayerNorm(
      dimensions: config.hiddenSize,
      eps: Float(config.layerNormEps)
    )
    _dropout.wrappedValue = Dropout(p: Float(config.hiddenDropout))
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    attentionMask: MLXArray? = nil
  ) -> MLXArray {
    var output = hiddenStates

    // Post-norm architecture:
    // 1. Attention
    // 2. Residual connection
    // 3. Layer norm
    // 4. Feed forward
    // 5. Residual connection
    // 6. Final layer norm

    // Self-attention with residual connection
    let attn_residual = output
    output = attention(output, attentionMask: attentionMask)
    output = attn_residual + output
    output = layerNorm(output)

    // Feed-forward with residual connection
    output = output + feedForward(output)
    output = finalLayerNorm(output)

    return output
  }
}

// MARK: - Wav2Vec2 Encoder

/// Full Wav2Vec2 encoder with feature projection and transformer layers.
///
/// Architecture:
/// 1. Feature projection: LayerNorm(512) + Linear(512 -> hiddenSize)
/// 2. Positional convolutional embedding
/// 3. Stack of encoder layers (post-norm transformer blocks)
/// 4. Final layer normalization
class Wav2Vec2Encoder: Module {
  @ModuleInfo(key: "feature_projection") var featureProjection: FeatureProjection
  @ModuleInfo(key: "pos_conv_embed") var posConvEmbed: Wav2Vec2PositionalConvEmbedding
  @ModuleInfo(key: "layers") var layers: [Wav2Vec2EncoderLayer]
  @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
  @ModuleInfo var dropout: Dropout

  let config: Wav2Vec2Config

  init(config: Wav2Vec2Config) {
    self.config = config

    _featureProjection.wrappedValue = FeatureProjection(
      inputDim: 512,  // Output of feature extractor
      outputDim: config.hiddenSize
    )
    _posConvEmbed.wrappedValue = Wav2Vec2PositionalConvEmbedding(
      embeddingSize: config.hiddenSize,
      numEmbeddings: config.numConvPosEmbeddings,
      numGroups: config.numConvPosEmbeddingGroups
    )
    _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
      Wav2Vec2EncoderLayer(config: config)
    }
    _layerNorm.wrappedValue = LayerNorm(
      dimensions: config.hiddenSize,
      eps: Float(config.layerNormEps)
    )
    _dropout.wrappedValue = Dropout(p: Float(config.hiddenDropout))
  }

  func callAsFunction(
    _ hiddenStates: MLXArray,
    attentionMask: MLXArray? = nil
  ) -> MLXArray {
    // hiddenStates shape: [batch, frames, channels] from feature extractor
    // This is already the correct format for FeatureProjection
    var output = hiddenStates

    // Feature projection: LayerNorm + Linear
    // Input: [batch, frames, 512] -> Output: [batch, frames, hiddenSize]
    output = featureProjection(output)

    // Add positional embeddings
    // posConvEmbed expects [batch, channels, frames] and outputs [batch, frames, channels]
    let positionalEmbeddings = posConvEmbed(output.transposed(0, 2, 1))
    output = output + positionalEmbeddings  // Already in [batch, frames, channels] format

    // Apply dropout
    output = dropout(output)

    // Pass through encoder layers
    for layer in layers {
      output = layer(output, attentionMask: attentionMask)
    }

    // Final layer norm
    output = layerNorm(output)

    return output
  }
}

// MARK: - Feature Projection

/// Projects feature extractor output to encoder hidden size.
///
/// Applies LayerNorm to the 512-dimensional features then
/// projects to the model's hidden size (e.g., 768 for base model).
class FeatureProjection: Module {
  @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
  @ModuleInfo(key: "projection") var projection: Linear

  let inputDim: Int
  let outputDim: Int

  init(inputDim: Int, outputDim: Int) {
    self.inputDim = inputDim
    self.outputDim = outputDim

    _layerNorm.wrappedValue = LayerNorm(dimensions: inputDim, eps: 1e-5)
    _projection.wrappedValue = Linear(inputDim, outputDim)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // x shape: [batch, frames, inputDim]
    var output = layerNorm(x)
    output = projection(output)
    return output
  }
}
