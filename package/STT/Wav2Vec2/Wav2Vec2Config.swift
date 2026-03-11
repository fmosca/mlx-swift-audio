// Copyright © 2021 Facebook AI Research (original model implementation)
// Ported to MLX from https://github.com/huggingface/transformers
// Copyright © Anthony DePasquale
// License: licenses/apache-2.0.txt

import Foundation

/// Configuration for Wav2Vec2 models
public struct Wav2Vec2Config: Codable, Sendable {
  // MARK: - Feature Extractor Config

  /// Convolutional feature extractor dimensions (one per layer)
  public var convDim: [Int]

  /// Convolutional feature extractor kernel sizes (one per layer)
  public var convKernel: [Int]

  /// Convolutional feature extractor strides (one per layer)
  public var convStride: [Int]

  /// Whether to use bias in conv layers
  public var convBias: Bool

  /// Normalization for feature extractor: "group" or "layer"
  public var featExtractNorm: String

  /// Activation function for feature extractor
  public var featExtractActivation: String

  // MARK: - Encoder Config

  /// Hidden size of the encoder
  public var hiddenSize: Int

  /// Number of hidden layers in the encoder
  public var numHiddenLayers: Int

  /// Number of attention heads in the encoder
  public var numAttentionHeads: Int

  /// Intermediate/FFN size
  public var intermediateSize: Int

  /// Hidden activation function
  public var hiddenAct: String

  /// Hidden dropout probability
  public var hiddenDropout: Double

  /// Attention dropout probability
  public var attentionDropout: Double

  /// Layer norm epsilon
  public var layerNormEps: Double

  /// Whether to use stable layer norm (pre-norm)
  public var doStableLayerNorm: Bool

  // MARK: - Positional Encoding Config

  /// Number of convolutional positional embeddings
  public var numConvPosEmbeddings: Int

  /// Number of groups for convolutional positional embeddings
  public var numConvPosEmbeddingGroups: Int

  // MARK: - CTC/Output Config

  /// Vocabulary size
  public var vocabSize: Int

  /// Final dropout probability
  public var finalDropout: Double

  /// Whether to apply SpecAugment
  public var applySpecAugment: Bool

  // MARK: - Coding Keys

  enum CodingKeys: String, CodingKey {
    case convDim = "conv_dim"
    case convKernel = "conv_kernel"
    case convStride = "conv_stride"
    case convBias = "conv_bias"
    case featExtractNorm = "feat_extract_norm"
    case featExtractActivation = "feat_extract_activation"
    case hiddenSize = "hidden_size"
    case numHiddenLayers = "num_hidden_layers"
    case numAttentionHeads = "num_attention_heads"
    case intermediateSize = "intermediate_size"
    case hiddenAct = "hidden_act"
    case hiddenDropout = "hidden_dropout"
    case attentionDropout = "attention_dropout"
    case layerNormEps = "layer_norm_eps"
    case doStableLayerNorm = "do_stable_layer_norm"
    case numConvPosEmbeddings = "num_conv_pos_embeddings"
    case numConvPosEmbeddingGroups = "num_conv_pos_embedding_groups"
    case vocabSize = "vocab_size"
    case finalDropout = "final_dropout"
    case applySpecAugment = "apply_spec_augment"
  }

  // MARK: - Initializer

  public init(
    convDim: [Int],
    convKernel: [Int],
    convStride: [Int],
    convBias: Bool,
    featExtractNorm: String,
    featExtractActivation: String,
    hiddenSize: Int,
    numHiddenLayers: Int,
    numAttentionHeads: Int,
    intermediateSize: Int,
    hiddenAct: String,
    hiddenDropout: Double,
    attentionDropout: Double,
    layerNormEps: Double,
    doStableLayerNorm: Bool,
    numConvPosEmbeddings: Int,
    numConvPosEmbeddingGroups: Int,
    vocabSize: Int,
    finalDropout: Double,
    applySpecAugment: Bool
  ) {
    self.convDim = convDim
    self.convKernel = convKernel
    self.convStride = convStride
    self.convBias = convBias
    self.featExtractNorm = featExtractNorm
    self.featExtractActivation = featExtractActivation
    self.hiddenSize = hiddenSize
    self.numHiddenLayers = numHiddenLayers
    self.numAttentionHeads = numAttentionHeads
    self.intermediateSize = intermediateSize
    self.hiddenAct = hiddenAct
    self.hiddenDropout = hiddenDropout
    self.attentionDropout = attentionDropout
    self.layerNormEps = layerNormEps
    self.doStableLayerNorm = doStableLayerNorm
    self.numConvPosEmbeddings = numConvPosEmbeddings
    self.numConvPosEmbeddingGroups = numConvPosEmbeddingGroups
    self.vocabSize = vocabSize
    self.finalDropout = finalDropout
    self.applySpecAugment = applySpecAugment
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)

    // Feature extractor with defaults
    convDim = try container.decodeIfPresent([Int].self, forKey: .convDim) ?? [Int](
      repeating: 512, count: 7)
    convKernel = try container.decodeIfPresent([Int].self, forKey: .convKernel) ?? [10, 3, 3, 3, 3,
      3, 3]
    convStride = try container.decodeIfPresent([Int].self, forKey: .convStride) ?? [5, 2, 2, 2, 2,
      2, 2]
    convBias = try container.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
    featExtractNorm = try container.decodeIfPresent(String.self, forKey: .featExtractNorm) ?? "group"
    featExtractActivation =
      try container.decodeIfPresent(String.self, forKey: .featExtractActivation) ?? "gelu"

    // Encoder with defaults
    hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
    numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 12
    numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 12
    intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
    hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu"
    hiddenDropout = try container.decodeIfPresent(Double.self, forKey: .hiddenDropout) ?? 0.1
    attentionDropout = try container.decodeIfPresent(Double.self, forKey: .attentionDropout) ?? 0.1
    layerNormEps = try container.decodeIfPresent(Double.self, forKey: .layerNormEps) ?? 1e-5
    doStableLayerNorm = try container.decodeIfPresent(Bool.self, forKey: .doStableLayerNorm) ?? false

    // Positional encoding with defaults
    numConvPosEmbeddings =
      try container.decodeIfPresent(Int.self, forKey: .numConvPosEmbeddings) ?? 128
    numConvPosEmbeddingGroups =
      try container.decodeIfPresent(Int.self, forKey: .numConvPosEmbeddingGroups) ?? 16

    // CTC/Output with defaults
    vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 32
    finalDropout = try container.decodeIfPresent(Double.self, forKey: .finalDropout) ?? 0.1
    applySpecAugment = try container.decodeIfPresent(Bool.self, forKey: .applySpecAugment) ?? false
  }

  // MARK: - Presets

  /// Base model configuration (95M parameters)
  public static let base = Wav2Vec2Config(
    convDim: [512, 512, 512, 512, 512, 512, 512],
    convKernel: [10, 3, 3, 3, 3, 3, 3],
    convStride: [5, 2, 2, 2, 2, 2, 2],
    convBias: false,
    featExtractNorm: "group",
    featExtractActivation: "gelu",
    hiddenSize: 768,
    numHiddenLayers: 12,
    numAttentionHeads: 12,
    intermediateSize: 3072,
    hiddenAct: "gelu",
    hiddenDropout: 0.1,
    attentionDropout: 0.1,
    layerNormEps: 1e-5,
    doStableLayerNorm: false,
    numConvPosEmbeddings: 128,
    numConvPosEmbeddingGroups: 16,
    vocabSize: 32,
    finalDropout: 0.1,
    applySpecAugment: false
  )

  /// Large model configuration (317M parameters)
  public static let large = Wav2Vec2Config(
    convDim: [512, 512, 512, 512, 512, 512, 512],
    convKernel: [10, 3, 3, 3, 3, 3, 3],
    convStride: [5, 2, 2, 2, 2, 2, 2],
    convBias: false,
    featExtractNorm: "group",
    featExtractActivation: "gelu",
    hiddenSize: 1024,
    numHiddenLayers: 24,
    numAttentionHeads: 16,
    intermediateSize: 4096,
    hiddenAct: "gelu",
    hiddenDropout: 0.1,
    attentionDropout: 0.1,
    layerNormEps: 1e-5,
    doStableLayerNorm: false,
    numConvPosEmbeddings: 128,
    numConvPosEmbeddingGroups: 16,
    vocabSize: 32,
    finalDropout: 0.1,
    applySpecAugment: false
  )

  // MARK: - Computed Properties

  /// Total downsampling factor from all conv layers
  public var downsamplingFactor: Int {
    convStride.reduce(1, *)
  }

  /// Frame rate in Hz (samples per second / downsampling factor)
  public var frameRate: Double {
    16000.0 / Double(downsamplingFactor)
  }

  /// Duration of each frame in seconds
  public var frameDuration: Double {
    1.0 / frameRate
  }

  /// Head dimension for attention
  public var headDim: Int {
    hiddenSize / numAttentionHeads
  }
}
