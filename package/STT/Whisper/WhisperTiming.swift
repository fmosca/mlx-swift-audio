// Copyright © 2022 OpenAI (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/openai/whisper
// License: licenses/whisper.txt

// Word-level timestamp support for Whisper using DTW alignment

import Accelerate
import Dispatch
import Foundation
import MLX

// MARK: - Word Timing Result

/// Represents timing information for a single word
public struct WordTiming: Sendable {
  /// The word text
  public var word: String

  /// Token IDs that make up this word
  public var tokens: [Int]

  /// Start time in seconds
  public var start: Float

  /// End time in seconds
  public var end: Float

  /// Average probability/confidence for this word
  public var probability: Float
}

// MARK: - DTW Algorithm

/// Dynamic Time Warping alignment
///
/// Finds the optimal alignment path between text tokens and audio frames
/// using dynamic programming. This is the core algorithm for word-level timestamps.
///
/// - Parameters:
///   - costMatrix: Flat cost matrix of shape (N, M) where lower values indicate better alignment
///   - rows: Number of rows (N) in the cost matrix
///   - cols: Number of columns (M) in the cost matrix
/// - Returns: Tuple of aligned (textIndices, timeIndices)
@inlinable
func dtw(
  _ costMatrix: UnsafeBufferPointer<Float>,
  rows N: Int,
  cols M: Int
) -> (textIndices: [Int], timeIndices: [Int]) {
  // Flat arrays for cache efficiency (matches Numba's optimization approach)
  let costRows = N + 1
  let costCols = M + 1
  var cost = [Float](repeating: .infinity, count: costRows * costCols)
  var trace = [Int8](repeating: -1, count: costRows * costCols)

  // Helper for 2D indexing into flat array
  @inline(__always)
  func idx(_ i: Int, _ j: Int) -> Int { i * costCols + j }

  cost[idx(0, 0)] = 0

  // Main DTW loop - matches Python implementation exactly
  for j in 1 ... M {
    for i in 1 ... N {
      let c0 = cost[idx(i - 1, j - 1)] // diagonal
      let c1 = cost[idx(i - 1, j)] // vertical
      let c2 = cost[idx(i, j - 1)] // horizontal

      let (c, t): (Float, Int8)
      if c0 < c1, c0 < c2 {
        (c, t) = (c0, 0)
      } else if c1 < c0, c1 < c2 {
        (c, t) = (c1, 1)
      } else {
        (c, t) = (c2, 2)
      }

      cost[idx(i, j)] = costMatrix[(i - 1) * M + (j - 1)] + c
      trace[idx(i, j)] = t
    }
  }

  // Set boundary conditions before backtracing (matches Python exactly)
  // First row: horizontal movement (2) - when at top, can only go left
  for j in 0 ..< costCols {
    trace[idx(0, j)] = 2
  }
  // First column: vertical movement (1) - when at left edge, can only go up
  for i in 0 ..< costRows {
    trace[idx(i, 0)] = 1
  }

  return backtrace(trace, rows: costRows, cols: costCols)
}

/// Backtrace through the DTW trace matrix to recover the alignment path
///
/// - Parameters:
///   - trace: Trace matrix from DTW (flat array)
///   - rows: Number of rows in trace matrix
///   - cols: Number of columns in trace matrix
/// - Returns: Tuple of aligned (textIndices, timeIndices)
@inlinable
func backtrace(_ trace: [Int8], rows: Int, cols: Int) -> (textIndices: [Int], timeIndices: [Int]) {
  @inline(__always)
  func idx(_ i: Int, _ j: Int) -> Int { i * cols + j }

  var i = rows - 1
  var j = cols - 1
  var result: [(Int, Int)] = []
  result.reserveCapacity(i + j) // Pre-allocate for performance

  while i > 0 || j > 0 {
    result.append((i - 1, j - 1))

    switch trace[idx(i, j)] {
      case 0: i -= 1; j -= 1 // diagonal
      case 1: i -= 1 // vertical
      case 2: j -= 1 // horizontal
      default:
        // Should not happen with valid input, but handle gracefully
        if i > 0 { i -= 1 }
        if j > 0 { j -= 1 }
    }
  }

  result.reverse()
  return (result.map { $0.0 }, result.map { $0.1 })
}

// MARK: - Punctuation Merging

/// Default punctuation characters to prepend to following word
let defaultPrependPunctuations = #"\"'"¿([{-"#

/// Default punctuation characters to append to previous word
let defaultAppendPunctuations = #"\"'.。,，!！?？:：")]}、"#

/// Merge punctuation marks with adjacent words
///
/// This improves word timestamp accuracy by combining punctuation with
/// the words they belong to rather than treating them as separate words.
///
/// - Parameters:
///   - alignment: Array of word timings to modify in-place
///   - prepended: Characters that should be merged with the following word
///   - appended: Characters that should be merged with the previous word
func mergePunctuations(
  _ alignment: inout [WordTiming],
  prepended: String = defaultPrependPunctuations,
  appended: String = defaultAppendPunctuations
) {
  guard alignment.count > 1 else { return }

  // Merge prepended punctuations (iterate backwards)
  var i = alignment.count - 2
  var j = alignment.count - 1
  while i >= 0 {
    let previousWord = alignment[i].word
    if previousWord.hasPrefix(" "), prepended.contains(previousWord.trimmingCharacters(in: .whitespaces)) {
      // Prepend to the following word
      alignment[j].word = previousWord + alignment[j].word
      alignment[j].tokens = alignment[i].tokens + alignment[j].tokens
      alignment[j].start = alignment[i].start
      alignment[i].word = ""
      alignment[i].tokens = []
    } else {
      j = i
    }
    i -= 1
  }

  // Merge appended punctuations (iterate forwards)
  i = 0
  j = 1
  while j < alignment.count {
    let previousWord = alignment[i].word
    let followingWord = alignment[j].word
    if !previousWord.hasSuffix(" "), appended.contains(followingWord) {
      // Append to the previous word
      alignment[i].word = previousWord + followingWord
      alignment[i].tokens = alignment[i].tokens + alignment[j].tokens
      alignment[i].end = alignment[j].end
      alignment[j].word = ""
      alignment[j].tokens = []
    } else {
      i = j
    }
    j += 1
  }

  // Remove empty entries
  alignment.removeAll { $0.word.isEmpty && $0.tokens.isEmpty }
}

// MARK: - Constants

/// Tokens per second in Whisper (50 tokens = 1 second, i.e., 20ms per token)
let tokensPerSecond: Float = 50.0

/// Sentence-ending punctuation marks for duration clipping
private let sentenceEndMarks: Set<Character> = [".", "。", "!", "！", "?", "？"]

// MARK: - Duration Clipping

/// Calculate median and max duration for word timing clipping
///
/// - Parameter alignment: Array of word timings
/// - Returns: Tuple of (medianDuration, maxDuration) where maxDuration = min(0.7, median) * 2
func calculateDurationThresholds(_ alignment: [WordTiming]) -> (median: Float, max: Float) {
  // Get non-zero word durations
  let durations = alignment.compactMap { timing -> Float? in
    let duration = timing.end - timing.start
    return duration > 0 ? duration : nil
  }

  guard !durations.isEmpty else {
    return (median: 0.0, max: 0.0)
  }

  // Calculate median
  let sorted = durations.sorted()
  let median: Float = if sorted.count % 2 == 0 {
    (sorted[sorted.count / 2 - 1] + sorted[sorted.count / 2]) / 2
  } else {
    sorted[sorted.count / 2]
  }

  // Cap median at 0.7s and calculate max as 2x median
  let cappedMedian = min(0.7, median)
  let maxDuration = cappedMedian * 2

  return (median: cappedMedian, max: maxDuration)
}

/// Clip long word durations at sentence boundaries
///
/// Words at sentence boundaries (., 。, !, ！, ?, ？) are clipped to maxDuration
/// if they exceed it. This helps correct alignment errors at natural pause points.
///
/// - Parameters:
///   - alignment: Array of word timings to modify in-place
///   - maxDuration: Maximum allowed duration for words at sentence boundaries
func clipAtSentenceBoundaries(_ alignment: inout [WordTiming], maxDuration: Float) {
  guard alignment.count > 1, maxDuration > 0 else { return }

  // Sentence end marks as strings for exact word matching (matches Python behavior)
  let sentenceEndMarkStrings = Set(sentenceEndMarks.map { String($0) })

  for i in 1 ..< alignment.count {
    let duration = alignment[i].end - alignment[i].start
    guard duration > maxDuration else { continue }

    let currentWord = alignment[i].word
    let previousWord = alignment[i - 1].word

    // Check if current word IS a sentence-ending mark (exact match, like Python)
    if sentenceEndMarkStrings.contains(currentWord) {
      alignment[i].end = alignment[i].start + maxDuration
    }
    // Check if previous word IS a sentence-ending mark
    else if sentenceEndMarkStrings.contains(previousWord) {
      alignment[i].start = alignment[i].end - maxDuration
    }
  }
}

/// Clip long word durations at segment boundaries (after pauses)
///
/// Handles the first and second word after a pause to prevent unreasonably long durations.
/// This is a heuristic to correct alignment errors when speech resumes after silence.
///
/// - Parameters:
///   - words: Array of word dictionaries with start/end times (modified in-place)
///   - lastSpeechTimestamp: End time of the last speech segment
///   - medianDuration: Median word duration for the segment
///   - maxDuration: Maximum allowed duration (typically 2x median)
func clipAtSegmentBoundaries(
  _ words: inout [WordTiming],
  lastSpeechTimestamp: Float,
  medianDuration: Float,
  maxDuration: Float
) {
  guard !words.isEmpty, maxDuration > 0 else { return }

  let firstWord = words[0]

  // Check if there's a significant pause before the first word
  let pauseBeforeFirst = firstWord.end - lastSpeechTimestamp
  let firstDuration = firstWord.end - firstWord.start

  if pauseBeforeFirst > medianDuration * 4 {
    // Long pause detected - check if first/second words need clipping
    let needsClipping = firstDuration > maxDuration ||
      (words.count > 1 && words[1].end - firstWord.start > maxDuration * 2)

    if needsClipping {
      // Clip second word if it's also too long
      if words.count > 1 {
        let secondDuration = words[1].end - words[1].start
        if secondDuration > maxDuration {
          let boundary = max(words[1].end / 2, words[1].end - maxDuration)
          words[0].end = boundary
          words[1].start = boundary
        }
      }
      // Clip first word start
      words[0].start = max(0, words[0].end - maxDuration)
    }
  }
}

/// Adjust segment boundaries based on word timestamps
///
/// Ensures segment start/end times are consistent with the first/last word timestamps,
/// preferring segment-level timestamps when words appear misaligned.
///
/// - Parameters:
///   - words: Array of word timings for this segment
///   - segmentStart: Segment start time (may be adjusted)
///   - segmentEnd: Segment end time (may be adjusted)
///   - medianDuration: Median word duration
/// - Returns: Tuple of adjusted (segmentStart, segmentEnd, lastSpeechTimestamp)
func adjustSegmentBoundaries(
  _ words: [WordTiming],
  segmentStart: Float,
  segmentEnd: Float,
  medianDuration: Float
) -> (start: Float, end: Float, lastSpeech: Float) {
  guard !words.isEmpty else {
    return (segmentStart, segmentEnd, segmentEnd)
  }

  var adjustedStart = segmentStart
  var adjustedEnd = segmentEnd

  let firstWord = words[0]
  let lastWord = words[words.count - 1]

  // Prefer segment-level start if first word appears too early
  if segmentStart < firstWord.end, segmentStart - 0.5 > firstWord.start {
    adjustedStart = max(0, min(firstWord.end - medianDuration, segmentStart))
  } else {
    adjustedStart = firstWord.start
  }

  // Prefer segment-level end if last word appears too late
  if segmentEnd > lastWord.start, segmentEnd + 0.5 < lastWord.end {
    adjustedEnd = max(lastWord.start + medianDuration, segmentEnd)
  } else {
    adjustedEnd = lastWord.end
  }

  return (adjustedStart, adjustedEnd, adjustedEnd)
}

// MARK: - GPU Median Filter

/// GPU-accelerated median filter (width=7)
///
/// Optimized implementation using MLX primitives to avoid CPU transfer.
/// Uses a "shift-stack-sort" approach:
/// 1. Pad input with reflection (matching scipy.signal.medfilt)
/// 2. Create 7 shifted views of the data
/// 3. Stack them on a new axis
/// 4. Sort along that axis and take the middle element
///
/// - Parameters:
///   - weights: Input tensor of shape (..., frames)
/// - Returns: Filtered tensor of same shape
func medianFilterAttentionGPU(_ weights: MLXArray) -> MLXArray {
  let shape = weights.shape
  let F = shape.last! // Frames is the last dimension
  let W = 7
  let P = W / 2 // 3

  // Reflect padding:
  // Left: weights[..., 3, 2, 1] (indices 1, 2, 3 reversed)
  // Right: weights[..., F-2, F-3, F-4] (indices F-4, F-3, F-2 reversed)

  // Simplest approach: concatenate on axis -1.
  
  // Left pad: weights[..., 3:0:-1]
  let leftPad = weights[.ellipsis, stride(from: P, to: 0, by: -1)]
  
  // Right pad: weights[..., -2:-5:-1]
  let rightPad = weights[.ellipsis, stride(from: F - 2, to: F - 2 - P, by: -1)]
  
  let padded = MLX.concatenated([leftPad, weights, rightPad], axis: -1)
  
  // Create shifted views
  var stacked: [MLXArray] = []
  for i in 0 ..< W {
    // Slice frames: padded[..., i ..< i + F]
    let slice = padded[.ellipsis, i ..< (i + F)]
    stacked.append(slice)
  }
  
  // Stack -> (..., F, W)
  let stackedTensor = MLX.stacked(stacked, axis: -1)
  
  // Sort along last axis
  let sorted = MLX.sorted(stackedTensor, axis: -1)
  
  // Take median (index 3)
  return sorted[.ellipsis, P]
}

// MARK: - Find Alignment

/// Find word-level alignment using cross-attention weights and DTW
///
/// This is the core function for extracting word-level timestamps. It:
/// 1. Runs a forward pass with the text tokens to get cross-attention weights
/// 2. Extracts and normalizes attention from alignment heads
/// 3. Uses DTW to align text tokens to audio frames
/// 4. Maps token boundaries to word boundaries
///
/// - Parameters:
///   - model: Whisper model with alignment heads configured
///   - tokenizer: Whisper tokenizer
///   - textTokens: Text tokens to align (without special tokens).
///                 Can be flat [Int] (single sequence) or [[Int]] (batch).
///                 If [Int] is passed, it is treated as batch=1.
///   - mel: Mel spectrogram (n_frames, n_mels) or (batch, n_frames, n_mels)
///   - numFrames: Number of audio frames (Int for single/constant, or [Int] for batch variable lengths)
///   - language: Optional language code for SOT sequence
///   - task: Transcription task
///   - medfiltWidth: Median filter width (default 7)
///   - qkScale: QK scaling factor (default 1.0)
/// - Returns: Array of word timings (or flattened list for batch=1 to maintain compat)
func findAlignment(
  model: WhisperModel,
  tokenizer: WhisperTokenizer,
  textTokens: Any, // [Int] or [[Int]]
  mel: MLXArray,
  audioFeatures: MLXArray? = nil,
  numFrames: Any, // Int or [Int]
  language: String?,
  task: TranscriptionTask,
  medfiltWidth: Int = 7,
  qkScale: Float = 1.0
) -> [[WordTiming]] { // Always return batch of results
  
  // Normalize input to batch format
  let batchTokens: [[Int]]
  if let single = textTokens as? [Int] {
    batchTokens = [single]
  } else if let batch = textTokens as? [[Int]] {
    batchTokens = batch
  } else {
    return []
  }
  
  guard !batchTokens.isEmpty, !batchTokens[0].isEmpty else { return [] }
  let batchSize = batchTokens.count
  
  // Normalize numFrames
  let batchNumFrames: [Int]
  if let single = numFrames as? Int {
      batchNumFrames = Array(repeating: single, count: batchSize)
  } else if let batch = numFrames as? [Int] {
      batchNumFrames = batch
  } else {
      return Array(repeating: [], count: batchSize)
  }

  // Check if alignment heads are available
  guard model.alignmentHeads.size > 0 else {
    Log.model.warning("No alignment heads configured - word timestamps unavailable")
    return Array(repeating: [], count: batchSize)
  }
  
  // Prepare batch tokens with SOT/EOT
  var processedBatch: [[Int]] = []
  var maxLen = 0
  
  let sotSeq = tokenizer.sotSequence(language: language, task: task)
  let noTimestampsIndex = sotSeq.count
  let textStartIndex = sotSeq.count + 1 // +1 for no_timestamps
  
  for tokens in batchTokens {
    var t = sotSeq
    t.append(tokenizer.noTimestamps)
    t.append(contentsOf: tokens)
    t.append(tokenizer.eot)
    processedBatch.append(t)
    maxLen = max(maxLen, t.count)
  }
  
  // Create padded token array [B, MaxLen]
  var paddedTokens = [Int32]()
  for t in processedBatch {
    paddedTokens.append(contentsOf: t.map { Int32($0) })
    let padCount = maxLen - t.count
    if padCount > 0 {
      paddedTokens.append(contentsOf: Array(repeating: Int32(tokenizer.eot), count: padCount))
    }
  }
  
  let tokenArray = MLXArray(paddedTokens).reshaped([batchSize, maxLen])

  // Forward pass with cross-attention extraction.
  let (logits, crossQK): (MLXArray, [MLXArray?])
  if let features = audioFeatures {
    // features: [B, nFrames, nCh]
    (logits, crossQK) = model.decodeWithCrossQK(features, tokens: tokenArray)
  } else {
    var melBatched = mel
    if mel.ndim == 2 { melBatched = mel.expandedDimensions(axis: 0) }
    (logits, crossQK) = model.forwardWithCrossQK(melBatched, tokens: tokenArray)
  }
  eval(logits)

  // Get alignment head indices
  let alignmentHeadsArray = model.alignmentHeads.asArray(Int32.self)
  let numAlignmentHeads = alignmentHeadsArray.count / 2
  let maxFrames = batchNumFrames.max() ?? 3000
  let maxFramesLen = maxFrames / 2
  
  // Collect attention weights: [B, Heads, Tokens, Frames]
  var headWeightsArrays: [MLXArray] = []
  for h in 0 ..< numAlignmentHeads {
    let layerIdx = Int(alignmentHeadsArray[h * 2])
    let headIdx = Int(alignmentHeadsArray[h * 2 + 1])
    guard layerIdx < crossQK.count, let layerQK = crossQK[layerIdx] else { continue }
    
    // layerQK: [B, nHeads, Tokens, Frames]
    // Extract head: layerQK[0..., headIdx] -> [B, Tokens, Frames]
    let headWeights = layerQK[0..., headIdx]
    headWeightsArrays.append(headWeights)
  }
  
  guard !headWeightsArrays.isEmpty else { return Array(repeating: [], count: batchSize) }
  
  // Stack heads -> [B, H, T, F]
  var weights = MLX.stacked(headWeightsArrays, axis: 1)
  
  // Slice frames to max length: weights[..., :maxFramesLen]
  weights = weights[.ellipsis, 0 ..< maxFramesLen]
  
  // Normalize (GPU)
  weights = MLX.softmax(weights * qkScale, axis: -1, precise: true)
  let mean = MLX.mean(weights, axis: -2, keepDims: true)
  let variance = MLX.variance(weights, axis: -2, keepDims: true)
  let std = MLX.sqrt(variance + 1e-8)
  weights = (weights - mean) / std
  weights = weights.asType(.float32)
  
  // Median Filter (GPU) - [B, H, T, F]
  let filteredWeights = medianFilterAttentionGPU(weights)
  
  // Average across heads (GPU) - [B, T, F]
  let avgMatrixGPU = MLX.mean(filteredWeights, axis: 1)
  eval(avgMatrixGPU)
  
  // Transfer to CPU for DTW
  let avgMatrixFlat = avgMatrixGPU.asArray(Float.self)
  let T = avgMatrixGPU.shape[1] // Max tokens
  let F = avgMatrixGPU.shape[2] // Frames (maxFramesLen)
  let stridePerItem = T * F
  
  var batchTimings: [[WordTiming]] = []
  
  avgMatrixFlat.withUnsafeBufferPointer { avgMatrixPtr in
    guard let basePtr = avgMatrixPtr.baseAddress else {
        batchTimings = Array(repeating: [], count: batchSize)
        return
    }

    for b in 0..<batchSize {
      let textTokens = batchTokens[b]
      let actualTextLen = textTokens.count
      let itemFramesLen = batchNumFrames[b] / 2
      
      // Matrix for this batch item
      let itemOffset = b * stridePerItem
      let itemMatrix = basePtr + itemOffset
      
      // Extract text portion
      // Note: column slice is tricky because flattened. We perform row-wise copy.
      let dtwRows = 1 + actualTextLen // no_timestamps + tokens
      let dtwCols = itemFramesLen
      var dtwInput = [Float](repeating: 0, count: dtwRows * dtwCols)
      
      for r in 0 ..< dtwRows {
          // Source row index in itemMatrix (start at no_timestamps = textStartIndex-1)
          let srcIdx = (textStartIndex - 1 + r) * F
          
          // Copy `itemFramesLen` columns
          // Negate while copying for DTW
          for c in 0 ..< dtwCols {
              if c < F { // Safety check, though dtwCols <= F
                  dtwInput[r * dtwCols + c] = -itemMatrix[srcIdx + c]
              }
          }
      }
      
      let (textIndices, timeIndices) = dtwInput.withUnsafeBufferPointer { ptr in
        dtw(ptr, rows: dtwRows, cols: dtwCols)
      }
      
      if textIndices.isEmpty {
        batchTimings.append([])
        continue
      }
      
      let (words, wordTokenGroups) = tokenizer.splitToWordTokens(textTokens + [tokenizer.eot])
      
      if wordTokenGroups.count <= 1 {
         batchTimings.append([])
         continue
      }
      
      var wordBoundaries = [0]
      var cumLen = 0
      for group in wordTokenGroups.dropLast() {
          cumLen += group.count
          wordBoundaries.append(cumLen)
      }
      
      var jumpPositions = [0]
      for i in 1 ..< textIndices.count {
          if textIndices[i] != textIndices[i - 1] {
              jumpPositions.append(i)
          }
      }
      
      let jumpTimes = jumpPositions.map { idx -> Float in
          guard idx < timeIndices.count else { return 0 }
          return Float(timeIndices[idx]) / tokensPerSecond
      }
      
      var wordTimings: [WordTiming] = []
      
      // Gather token probs from logits (on GPU)
      let itemLogits = logits[b] // [MaxLen, V]
      let logitsStart = textStartIndex - 1
      let itemSampledLogits = itemLogits[logitsStart ..< (logitsStart + actualTextLen), 0 ..< tokenizer.eot]
      let itemTokenProbs = MLX.softmax(itemSampledLogits, axis: -1, precise: true)
      
      // Gather specific token probs
      let indices = MLXArray(textTokens.map { Int32($0) }).expandedDimensions(axis: 1)
      let gathered = MLX.takeAlong(itemTokenProbs, indices, axis: 1).squeezed()
      let tokenProbsArray = gathered.asArray(Float.self)
      
      for i in 0 ..< (words.count - 1) {
          let startBoundary = wordBoundaries[i]
          let endBoundary = wordBoundaries[i + 1]
          
          let startTime: Float
          let endTime: Float
          
          if startBoundary < jumpTimes.count { startTime = jumpTimes[startBoundary] }
          else if !jumpTimes.isEmpty { startTime = jumpTimes.last! }
          else { startTime = 0 }
          
          if endBoundary < jumpTimes.count { endTime = jumpTimes[endBoundary] }
          else if !jumpTimes.isEmpty { endTime = jumpTimes.last! }
          else { endTime = startTime }
          
          var avgProb: Float = 0
          let probStart = wordBoundaries[i]
          let probEnd = min(wordBoundaries[i + 1], tokenProbsArray.count)
          if probStart < probEnd {
              for j in probStart ..< probEnd { avgProb += tokenProbsArray[j] }
              avgProb /= Float(probEnd - probStart)
          }
          
          wordTimings.append(WordTiming(
              word: words[i],
              tokens: wordTokenGroups[i],
              start: startTime,
              end: max(endTime, startTime),
              probability: avgProb
          ))
      }
      
      batchTimings.append(wordTimings)
    }
  }
  
  return batchTimings
}

// Wrapper to maintain backward compatibility for single-sequence calls
func findAlignment(
  model: WhisperModel,
  tokenizer: WhisperTokenizer,
  textTokens: [Int], // Single sequence
  mel: MLXArray,
  audioFeatures: MLXArray? = nil,
  numFrames: Int,
  language: String?,
  task: TranscriptionTask,
  medfiltWidth: Int = 7,
  qkScale: Float = 1.0
) -> [WordTiming] {
  let batch = findAlignment(
    model: model,
    tokenizer: tokenizer,
    textTokens: [textTokens] as [[Int]],
    mel: mel,
    audioFeatures: audioFeatures,
    numFrames: numFrames as Any,
    language: language,
    task: task,
    medfiltWidth: medfiltWidth,
    qkScale: qkScale
  )
  return batch.first ?? []
}

// MARK: - Batched Word Timestamps

/// Add word-level timestamps to segments using batched processing
///
/// This matches Python's `add_word_timestamps()` function which processes all segments
/// in a single forward pass for efficiency. Key optimizations:
/// - Single `findAlignment` call for all segments (vs per-segment in naive approach)
/// - Batched duration calculations and clipping
///
/// - Parameters:
///   - segments: Array of transcription segments to add word timestamps to (modified in-place)
///   - model: Whisper model with alignment heads
///   - tokenizer: Whisper tokenizer
///   - mel: Mel spectrogram for the current window
///   - numFrames: Number of audio frames (segment_size). Can be [Int] for variable batched.
///   - language: Language code
///   - task: Transcription task
///   - timeOffset: Time offset for current window (used if timeOffsets is nil)
///   - timeOffsets: Optional array of time offsets, one per segment (for VAD batched mode)
///   - lastSpeechTimestamp: End time of last speech (for segment boundary clipping)
/// - Returns: Updated lastSpeechTimestamp
func addWordTimestamps(
  segments: inout [TranscriptionSegment],
  model: WhisperModel,
  tokenizer: WhisperTokenizer,
  mel: MLXArray,
  audioFeatures: MLXArray? = nil,
  numFrames: Any, // Int or [Int]
  language: String?,
  task: TranscriptionTask,
  timeOffset: Float,
  timeOffsets: [Float]? = nil,
  lastSpeechTimestamp: Float
) -> Float {
  guard !segments.isEmpty else { return lastSpeechTimestamp }

  // Extract text tokens per segment (matching Python exactly)
  let textTokensPerSegment: [[Int]] = segments.map { segment in
    segment.tokens.filter { $0 < tokenizer.eot }
  }

  guard !textTokensPerSegment.isEmpty else { return lastSpeechTimestamp }
  
  // Determine if we are in "Batched VAD Mode" (independent contexts) or "Shared Context Mode"
  // VAD Mode: audioFeatures has batch dim matching segments count, and segments > 1
  // Note: audioFeatures might be nil (if not passed), check segments count vs mel shape?
  // Assume if timeOffsets is passed, it is Batched VAD Mode.
  let isBatchedContext = timeOffsets != nil || ((audioFeatures?.shape[0] ?? 0) == segments.count && segments.count > 1)
  
  var alignment: [WordTiming] = []
  
  if isBatchedContext {
      // Batched Mode: segments have independent audio contexts (e.g. from VAD)
      // Pass tokens as batch [[Int]] to findAlignment
      let batchAlignment = findAlignment(
        model: model,
        tokenizer: tokenizer,
        textTokens: textTokensPerSegment,
        mel: mel,
        audioFeatures: audioFeatures,
        numFrames: numFrames,
        language: language,
        task: task
      )
      // Flatten the results to a single sequence of words
      alignment = batchAlignment.flatMap { $0 }
  } else {
      // Shared Context Mode: segments share the same audio context (e.g. sequential decoding)
      // Concatenate all text tokens for single-sequence alignment (matching Python)
      let allTextTokens = textTokensPerSegment.flatMap { $0 }
      guard !allTextTokens.isEmpty else { return lastSpeechTimestamp }
      
      // Single findAlignment call for the concatenated sequence
      alignment = findAlignment(
        model: model,
        tokenizer: tokenizer,
        textTokens: allTextTokens, // [Int] -> wrapper calls findAlignment batch=1
        mel: mel,
        audioFeatures: audioFeatures,
        numFrames: (numFrames as? Int) ?? (numFrames as? [Int])?.first ?? 3000,
        language: language,
        task: task
      )
  }

  guard !alignment.isEmpty else { return lastSpeechTimestamp }

  // Calculate duration thresholds across all words
  let (medianDuration, maxDuration) = calculateDurationThresholds(alignment)

  // Clip long words at sentence boundaries (Python lines 250-258)
  if maxDuration > 0 {
    clipAtSentenceBoundaries(&alignment, maxDuration: maxDuration)
  }

  // Merge punctuations
  mergePunctuations(&alignment)

  // Distribute words back to segments (Python lines 265-329)
  var wordIndex = 0
  var updatedLastSpeechTimestamp = lastSpeechTimestamp
  
  // Use provided per-segment offsets or fallback to single global offset
  let offsets = timeOffsets ?? Array(repeating: timeOffset, count: segments.count)

  for (segmentIdx, textTokens) in textTokensPerSegment.enumerated() {
    var savedTokens = 0
    var words: [Word] = []
    let currentOffset = offsets[segmentIdx]

    // Consume words until we've covered this segment's tokens
    while wordIndex < alignment.count, savedTokens < textTokens.count {
      let timing = alignment[wordIndex]

      if !timing.word.isEmpty {
        words.append(Word(
          word: timing.word,
          start: TimeInterval(currentOffset + timing.start),
          end: TimeInterval(currentOffset + timing.end),
          probability: timing.probability
        ))
      }

      savedTokens += timing.tokens.count
      wordIndex += 1
    }

    if !words.isEmpty {
      // Clip at segment boundaries (Python lines 287-303)
      if maxDuration > 0 {
        var wordTimings = words.map {
          WordTiming(word: $0.word, tokens: [], start: Float($0.start), end: Float($0.end), probability: $0.probability)
        }
        clipAtSegmentBoundaries(
          &wordTimings,
          lastSpeechTimestamp: updatedLastSpeechTimestamp,
          medianDuration: medianDuration,
          maxDuration: maxDuration
        )
        // Apply clipped times back to words
        for i in 0 ..< min(words.count, wordTimings.count) {
          words[i] = Word(
            word: words[i].word,
            start: TimeInterval(wordTimings[i].start),
            end: TimeInterval(wordTimings[i].end),
            probability: words[i].probability
          )
        }
      }

      // Adjust segment boundaries based on word timestamps (Python lines 305-325)
      let segment = segments[segmentIdx]
      let segmentStart = Float(segment.start)
      let segmentEnd = Float(segment.end)

      var adjustedStart = segmentStart
      var adjustedEnd = segmentEnd

      if let firstWord = words.first {
        // Prefer segment-level start if first word appears too early
        if segmentStart < Float(firstWord.end), segmentStart - 0.5 > Float(firstWord.start) {
          adjustedStart = max(0, min(Float(firstWord.end) - medianDuration, segmentStart))
          words[0] = Word(
            word: firstWord.word,
            start: TimeInterval(adjustedStart),
            end: firstWord.end,
            probability: firstWord.probability
          )
        } else {
          adjustedStart = Float(firstWord.start)
        }
      }

      if let lastWord = words.last {
        // Prefer segment-level end if last word appears too late
        if segmentEnd > Float(lastWord.start), segmentEnd + 0.5 < Float(lastWord.end) {
          adjustedEnd = max(Float(lastWord.start) + medianDuration, segmentEnd)
          words[words.count - 1] = Word(
            word: lastWord.word,
            start: lastWord.start,
            end: TimeInterval(adjustedEnd),
            probability: lastWord.probability
          )
        } else {
          adjustedEnd = Float(lastWord.end)
        }
        updatedLastSpeechTimestamp = adjustedEnd
      }

      // Update segment with words and adjusted boundaries
      segments[segmentIdx] = TranscriptionSegment(
        text: segment.text,
        start: TimeInterval(adjustedStart),
        end: TimeInterval(adjustedEnd),
        tokens: segment.tokens,
        avgLogProb: segment.avgLogProb,
        noSpeechProb: segment.noSpeechProb,
        words: words
      )
    }
  }

  return updatedLastSpeechTimestamp
}
