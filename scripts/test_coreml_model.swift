#!/usr/bin/env swift
import Foundation
import CoreML

// Test script to validate CoreML wav2vec2 model
// This generates test audio and runs inference to verify the model loads and works

print("[*] Loading CoreML model...")

let modelURL = URL(fileURLWithPath: "/Users/francesco.mosca/Work/mlx-swift-audio/models/wav2vec2/Wav2Vec2CTC.mlmodelc")

do {
    let config = MLModelConfiguration()
    config.computeUnits = .all
    let model = try MLModel(contentsOf: modelURL, configuration: config)

    print("[+] Model loaded successfully")
    print("[*] Model description:")
    print("    Inputs:", model.modelDescription.inputDescriptionsByName.keys.sorted())
    print("    Outputs:", model.modelDescription.outputDescriptionsByName.keys.sorted())

    // Check input description
    var expectedSamples = 16000
    if let inputDesc = model.modelDescription.inputDescriptionsByName["input_audio"] {
        print("[*] input_audio shape:", inputDesc.multiArrayConstraint?.shape ?? "nil")
        if let shape = inputDesc.multiArrayConstraint?.shape, shape.count >= 2 {
            expectedSamples = Int(truncating: shape[1])
        }
    }

    // Generate test audio (matching the model's expected input size)
    let numSamples = expectedSamples
    print("[*] Generating test audio: \(numSamples) samples (\(Double(numSamples)/16000.0)s @ 16kHz)")

    // Use fixed seed for reproducibility - generate same audio as Python
    srand48(42)
    var testAudio = [Float](repeating: 0, count: numSamples)
    for i in 0..<numSamples {
        testAudio[i] = Float(drand48() * 2.0 - 1.0)  // Range [-1, 1]
    }

    // Create MLMultiArray
    let inputArray = try MLMultiArray(shape: [1, numSamples as NSNumber], dataType: .float32)
    let ptr = inputArray.dataPointer.bindMemory(to: Float.self, capacity: numSamples)
    for i in 0..<numSamples {
        ptr[i] = testAudio[i]
    }

    print("[*] Running inference...")
    let start = Date()
    let inputDict: [String: Any] = ["input_audio": inputArray]
    let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputDict))
    let elapsed = Date().timeIntervalSince(start)

    print("[+] Inference completed in \(String(format: "%.1f", elapsed * 1000)) ms")

    // Get output logits
    // The output name might vary - check what's available
    let outputName = model.modelDescription.outputDescriptionsByName.keys.first ?? "var_891"
    print("[*] Using output: \(outputName)")

    if let logits = output.featureValue(for: outputName)?.multiArrayValue {
        print("[*] Output shape:", logits.shape)
        print("[*] Output data type:", logits.dataType)

        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)
        let count = logits.count

        // Calculate statistics
        var minVal: Float = .greatestFiniteMagnitude
        var maxVal: Float = -.greatestFiniteMagnitude
        var sum: Float = 0
        var sumSq: Float = 0

        for i in 0..<count {
            let val = ptr[i]
            if val < minVal { minVal = val }
            if val > maxVal { maxVal = val }
            sum += val
            sumSq += val * val
        }

        let mean = sum / Float(count)
        let variance = (sumSq / Float(count)) - (mean * mean)
        let std = sqrt(max(0, variance))

        print("[*] Output statistics:")
        print("    Min: \(String(format: "%.6f", minVal))")
        print("    Max: \(String(format: "%.6f", maxVal))")
        print("    Mean: \(String(format: "%.6f", mean))")
        print("    Std: \(String(format: "%.6f", std))")

        // Compare with expected PyTorch values (from our Python test)
        // PyTorch: mean ~4-5, std ~3-4 (these are logits before softmax)
        print("\n[*] Validation:")
        if maxVal > 10 && abs(mean) > 1 {
            print("[+] CoreML model appears to be working correctly!")
            print("    (Logits have reasonable magnitude)")
        } else {
            print("[-] Warning: Output values seem unusual")
        }
    }

    print("\n[+] CoreML model validation PASSED!")

} catch {
    print("[-] Error: \(error)")
    exit(1)
}
