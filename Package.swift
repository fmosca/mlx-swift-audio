// swift-tools-version:6.2
import PackageDescription

let package = Package(
  name: "mlx-audio",
  platforms: [.macOS("15.4"), .iOS("18.4")],
  products: [
    // Core library without Kokoro (no GPLv3 dependencies)
    .library(
      name: "MLXAudio",
      targets: ["MLXAudio"],
    ),
    // Separate Kokoro package (depends on GPLv3-licensed espeak-ng)
    .library(
      name: "Kokoro",
      targets: ["Kokoro"],
    ),
    // wav2vec2 forced aligner (CoreML, ANE-accelerated) for word-level timestamps
    .library(
      name: "Wav2Vec2Aligner",
      targets: ["Wav2Vec2Aligner"],
    ),
    // Benchmark CLI for performance testing
    .executable(
      name: "WhisperBenchmark",
      targets: ["WhisperBenchmark"],
    ),
    // Test executable for Wav2Vec2 weight loading
    .executable(
      name: "Wav2Vec2WeightTest",
      targets: ["Wav2Vec2WeightTest"],
    ),
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),
    .package(url: "https://github.com/ml-explore/mlx-swift", branch: "main"),
    // upToNextMajor allows FluidAudio's ≥ 1.1.6 requirement to be resolved
    .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMajor(from: "1.1.0")),
    .package(url: "https://github.com/DePasqualeOrg/swift-tiktoken", branch: "main"),
    // Silero VAD (CoreML, ANE-accelerated) for batched transcription segmentation
    .package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.12.3"),
    // Qwen3-ForcedAligner for word-level timestamp alignment (alternative to DTW)
    .package(url: "https://github.com/soniqo/speech-swift.git", branch: "main"),
    // espeak-ng is GPLv3 licensed - only linked when using Kokoro
    // TODO: Switch back to upstream after https://github.com/espeak-ng/espeak-ng/pull/2327 is merged
    .package(url: "https://github.com/DePasqualeOrg/espeak-ng-spm.git", branch: "fix-path-espeak-data-macro"),
  ],
  targets: [
    .target(
      name: "MLXAudio",
      dependencies: [
        .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXFFT", package: "mlx-swift"),
        .product(name: "Transformers", package: "swift-transformers"),
        .product(name: "SwiftTiktoken", package: "swift-tiktoken"),
        .product(name: "FluidAudio", package: "FluidAudio"),
      ],
      path: "package",
      exclude: ["TTS/Kokoro", "Wav2Vec2Aligner", "Tests"],
      resources: [
        .process("TTS/OuteTTS/default_speaker.json"), // Default speaker profile for OuteTTS
      ],
    ),
    .target(
      name: "Kokoro",
      dependencies: [
        "MLXAudio",
        .product(name: "libespeak-ng", package: "espeak-ng-spm"),
        .product(name: "espeak-ng-data", package: "espeak-ng-spm"),
      ],
      path: "package/TTS/Kokoro",
    ),
    .target(
      name: "Wav2Vec2Aligner",
      dependencies: [
        "MLXAudio",
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
      ],
      path: "package/Wav2Vec2Aligner",
    ),
    .testTarget(
      name: "MLXAudioTests",
      dependencies: ["MLXAudio", "Wav2Vec2Aligner"],
      path: "package/Tests",
    ),
    .executableTarget(
      name: "WhisperBenchmark",
      dependencies: [
        "MLXAudio",
        "Wav2Vec2Aligner",
        .product(name: "Qwen3ASR", package: "speech-swift"),
      ],
      path: "benchmarks/cli",
    ),
    .executableTarget(
      name: "Wav2Vec2WeightTest",
      dependencies: [
        "MLXAudio",
        .product(name: "Transformers", package: "swift-transformers"),
      ],
      path: "scripts",
      sources: ["verify_wav2vec2_weights.swift"],
    ),
  ],
)
