// Copyright © Anthony DePasquale
//
// WhisperX-style Trellis CTC Forced Alignment
//
// This is an alternative to the blank-interleaved CTC formulation.
// It uses a simpler trellis without explicit blank states, making it
// more robust to "peaky" model outputs where characters only get
// high confidence on single frames.
//
// Reference: https://pytorch.org/tutorials/intermediate/forced_alignment_with_torchaudio_tutorial.html
// Source: /private/tmp/whisperX/whisperx/alignment.py

import Foundation

/// Trellis-based CTC forced alignment (WhisperX style)
///
/// Key differences from blank-interleaved CTC:
/// - States represent only tokens (not interleaved with blanks)
/// - Blank emissions = "stay" on current token
/// - Token emissions = "change" to next token
/// - Produces continuous coverage even with peaky predictions
public struct TrellisForcedAligner {

  /// Perform forced alignment on log probabilities
  ///
  /// - Parameters:
  ///   - logProbs: Frame-level log probabilities [frames][vocabSize]
  ///   - tokens: Token IDs to align
  ///   - idToToken: Mapping from token ID to string representation
  ///   - blankId: ID of blank token (default 0)
  /// - Returns: Array of aligned tokens with timestamps
  public static func align(
    logProbs: [[Float]],
    tokens: [Int],
    idToToken: [Int: String],
    blankId: Int = 0
  ) -> [AlignedToken] {

    guard !logProbs.isEmpty && !tokens.isEmpty else {
      return []
    }

    let numFrames = logProbs.count
    let numTokens = tokens.count

    // Convert log probs to emissions (log domain)
    // logProbs is [frames][vocabSize], need [vocabSize][frames] for easier access
    var emissions: [[Float]] = Array(repeating: Array(repeating: 0, count: numFrames), count: logProbs[0].count)
    for frame in 0..<numFrames {
      for token in 0..<logProbs[frame].count {
        emissions[token][frame] = logProbs[frame][token]
      }
    }

    // Build trellis: [numFrames + 1, numTokens + 1]
    // Extra dim for tokens represents start-of-sentence
    // Extra dim for time axis is for code simplification
    var trellis = Array(repeating: Array(repeating: Float(-Float.infinity), count: numTokens + 1), count: numFrames + 1)
    trellis[0][0] = 0

    // Initialize first column (all blanks, no tokens emitted)
    var cumulativeBlank: Float = 0
    for t in 0..<numFrames {
      cumulativeBlank += emissions[blankId][t]
      trellis[t + 1][0] = cumulativeBlank
    }

    // Initialize first row (all tokens at time 0 - impossible except start)
    for j in 1...numTokens {
      trellis[0][j] = -Float.infinity
    }
    trellis[numFrames][0] = Float.infinity  // Marker for backtrack

    // Fill trellis
    for t in 0..<numFrames {
      for j in 1...numTokens {
        // Score for staying at the same token (emit blank)
        let stayed = trellis[t][j] + emissions[blankId][t]
        // Score for changing to the next token (emit token)
        let changed = trellis[t][j - 1] + emissions[tokens[j - 1]][t]
        trellis[t + 1][j] = max(stayed, changed)
      }
    }

    // Backtrack to find optimal path
    return backtrack(trellis: trellis, emissions: emissions, tokens: tokens, idToToken: idToToken, blankId: blankId)
  }

  /// Backtrack through trellis to find optimal path
  private static func backtrack(
    trellis: [[Float]],
    emissions: [[Float]],
    tokens: [Int],
    idToToken: [Int: String],
    blankId: Int
  ) -> [AlignedToken] {

    let numFrames = trellis.count - 1
    let numTokens = tokens.count

    // Find the best ending time (argmax of last column)
    var j = numTokens
    var tStart = 0
    var maxScore: Float = -Float.infinity
    for t in 0..<trellis.count {
      if trellis[t][j] > maxScore {
        maxScore = trellis[t][j]
        tStart = t
      }
    }

    // Backtrack from (tStart, j) to (0, 0)
    var path: [Point] = []
    for t in stride(from: tStart, through: 1, by: -1) {
      // Score for staying at same token (emit blank)
      let stayed = trellis[t - 1][j] + emissions[blankId][t - 1]
      // Score for changing to next token (emit token)
      let changed = trellis[t - 1][j - 1] + emissions[tokens[j - 1]][t - 1]

      // Get probability for the chosen path
      let tokenId = changed > stayed ? tokens[j - 1] : blankId
      let prob = exp(emissions[tokenId][t - 1])

      path.append(Point(tokenIndex: j - 1, timeIndex: t - 1, score: prob))

      // Update token index if we changed
      if changed > stayed {
        j -= 1
        if j == 0 {
          break
        }
      }
    }

    // Reverse to get forward path
    path.reverse()

    // Merge consecutive frames with same token
    return mergeRepeats(path: path, tokens: tokens, idToToken: idToToken)
  }

  /// Merge consecutive frames with same token into segments
  private static func mergeRepeats(path: [Point], tokens: [Int], idToToken: [Int: String]) -> [AlignedToken] {
    guard !path.isEmpty else {
      return []
    }

    var segments: [AlignedToken] = []
    var i1 = 0

    while i1 < path.count {
      var i2 = i1
      // Find all consecutive frames with same token
      while i2 < path.count && path[i1].tokenIndex == path[i2].tokenIndex {
        i2 += 1
      }

      let tokenIdx = path[i1].tokenIndex
      if tokenIdx >= 0 && tokenIdx < tokens.count {
        let tokenId = tokens[tokenIdx]
        let tokenString = idToToken[tokenId] ?? "?"  // Use mapping or default

        // Calculate average score
        var totalScore: Float = 0
        for k in i1..<i2 {
          totalScore += path[k].score
        }
        let avgScore = totalScore / Float(i2 - i1)

        // Create segment
        segments.append(AlignedToken(
          token: tokenString,
          tokenId: tokenId,
          startFrame: path[i1].timeIndex,
          endFrame: path[i2 - 1].timeIndex + 1  // Exclusive end
        ))
      }

      i1 = i2
    }

    return segments
  }

  /// Point in the trellis path
  private struct Point {
    var tokenIndex: Int
    var timeIndex: Int
    var score: Float
  }
}
