// Copyright © Anthony DePasquale

import Foundation

/// Result of CTC forced alignment for a single token
public struct AlignedToken {
  /// The token character/string
  public let token: String

  /// The token ID in the vocabulary
  public let tokenId: Int

  /// Starting frame index (inclusive)
  public let startFrame: Int

  /// Ending frame index (inclusive)
  public let endFrame: Int

  /// Starting time in seconds (20ms per frame)
  public var startTime: Float {
    Float(startFrame) * 0.02
  }

  /// Ending time in seconds (20ms per frame)
  public var endTime: Float {
    Float(endFrame + 1) * 0.02  // +1 to make end time exclusive
  }

  public init(token: String, tokenId: Int, startFrame: Int, endFrame: Int) {
    self.token = token
    self.tokenId = tokenId
    self.startFrame = startFrame
    self.endFrame = endFrame
  }
}

/// CTC Forced Alignment algorithm
///
/// Implements the dynamic programming algorithm for aligning a known text
/// sequence to frame-level log probabilities from an acoustic model.
public final class CTCForcedAligner {
  // MARK: - Properties

  /// The blank token ID (typically 0 for wav2vec2)
  public let blankId: Int

  // MARK: - Initialization

  /// Initialize the aligner
  ///
  /// - Parameter blankId: The CTC blank token ID (default: 0)
  public init(blankId: Int = 0) {
    self.blankId = blankId
  }

  // MARK: - Public Methods

  /// Align known text to frame-level probabilities using CTC forced alignment
  ///
  /// This implements the full CTC forward-backward algorithm with backtracing
  /// to find the optimal alignment path between the text tokens and audio frames.
  ///
  /// - Parameters:
  ///   - logProbs: Frame-level log probabilities [T][V] where T is frames, V is vocab size
  ///   - tokens: Known text as token IDs (without blanks)
  ///   - idToToken: Optional mapping from token ID to string for populating AlignedToken.token
  /// - Returns: Array of aligned tokens with frame boundaries
  /// - Throws: CTCAlignmentError if alignment is not feasible
  public func align(
    logProbs: [[Float]],
    tokens: [Int],
    idToToken: [Int: String] = [:]
  ) throws -> [AlignedToken] {
    guard !tokens.isEmpty else {
      throw Wav2Vec2Error.alignmentFailure("Token sequence is empty")
    }

    guard !logProbs.isEmpty else {
      throw Wav2Vec2Error.alignmentFailure("Log probability array is empty")
    }

    let T = logProbs.count  // Number of frames
    let S = tokens.count    // Number of non-blank tokens

    // Feasibility check: need at least as many frames as 2*S + 1 (blanks between each token)
    let minFrames = 2 * S + 1
    guard T >= minFrames else {
      throw Wav2Vec2Error.alignmentFailure(
        "Insufficient frames: \(T) frames for \(S) tokens (need at least \(minFrames))"
      )
    }

    // Build target sequence with blanks: _t1_t2_t3_..._tn_
    // This is the expanded CTC label sequence
    var targets = [blankId]
    for token in tokens {
      targets.append(token)
      targets.append(blankId)
    }
    let targetLength = targets.count

    // DP tables for forward pass
    // alpha[t][s] = best log probability to be in state s at time t
    var alpha = [[Float]](
      repeating: [Float](repeating: -.infinity, count: targetLength),
      count: T
    )

    // Backpointer table for backtracing
    // backptr[t][s] = previous state index that led to state s at time t
    var backptr = [[Int]](
      repeating: [Int](repeating: -1, count: targetLength),
      count: T
    )

    // Initialize first frame (t=0)
    // Can start in blank (s=0) or first token (s=1)
    alpha[0][0] = logProbs[0][blankId]
    if targetLength > 1 {
      alpha[0][1] = logProbs[0][targets[1]]
    }

    // Forward pass: fill DP table
    for t in 1..<T {
      for s in 0..<targetLength {
        let tokenId = targets[s]
        let emitScore = logProbs[t][tokenId]

        var candidates: [(score: Float, prev: Int)] = []

        // Transition 1: Stay in same state (repeat blank or repeat token)
        if alpha[t - 1][s] > -.infinity {
          candidates.append((alpha[t - 1][s], s))
        }

        // Transition 2: Move from previous state
        if s > 0 && alpha[t - 1][s - 1] > -.infinity {
          candidates.append((alpha[t - 1][s - 1], s - 1))
        }

        // Transition 3: Skip blank (jump from s-2 to s)
        // Only allowed when:
        // - We're far enough (s > 1)
        // - Current token is not blank
        // - Current token differs from the token two positions back
        // This prevents repeating the same token without a blank in between
        if s > 1
          && tokenId != blankId
          && targets[s] != targets[s - 2]
          && alpha[t - 1][s - 2] > -.infinity
        {
          candidates.append((alpha[t - 1][s - 2], s - 2))
        }

        // Select best transition
        if let best = candidates.max(by: { $0.score < $1.score }) {
          alpha[t][s] = best.score + emitScore
          backptr[t][s] = best.prev
        }
      }
    }

    // Find best final state
    // CTC allows ending in either the last blank or the last token
    var bestFinalScore = alpha[T - 1][targetLength - 1]
    var bestFinalState = targetLength - 1

    if targetLength > 1 {
      let secondLastScore = alpha[T - 1][targetLength - 2]
      if secondLastScore > bestFinalScore {
        bestFinalScore = secondLastScore
        bestFinalState = targetLength - 2
      }
    }

    guard bestFinalScore > -.infinity else {
      throw Wav2Vec2Error.alignmentFailure("No valid alignment path found")
    }

    // Backtrace to find the optimal path
    var path: [(frame: Int, state: Int)] = []
    path.reserveCapacity(T)

    var currentState = bestFinalState
    for t in stride(from: T - 1, through: 0, by: -1) {
      path.append((t, currentState))
      if t > 0 && currentState >= 0 {
        currentState = backptr[t][currentState]
      }
    }
    path.reverse()

    // Convert path to token alignments
    // Merge consecutive frames for the same token, skip blanks
    return convertPathToAlignments(
      path: path,
      targets: targets,
      idToToken: idToToken,
      totalFrames: T
    )
  }

  // MARK: - Private Methods

  /// Convert the backtraced path to AlignedToken objects
  ///
  /// This merges consecutive frames for the same token and skips blank frames.
  private func convertPathToAlignments(
    path: [(frame: Int, state: Int)],
    targets: [Int],
    idToToken: [Int: String],
    totalFrames: Int
  ) -> [AlignedToken] {
    var alignedTokens: [AlignedToken] = []
    var currentToken: Int? = nil
    var startFrame = 0
    var lastNonBlankFrame = 0  // Track the last frame where current token appeared

    for (frame, state) in path {
      let tokenId = targets[state]

      if tokenId != blankId {
        // Non-blank token
        if currentToken == nil {
          // Start of a new token
          currentToken = tokenId
          startFrame = frame
          lastNonBlankFrame = frame
        } else if tokenId != currentToken {
          // Token changed - end previous token, start new one
          let tokenString = idToToken[currentToken!] ?? ""
          alignedTokens.append(AlignedToken(
            token: tokenString,
            tokenId: currentToken!,
            startFrame: startFrame,
            endFrame: lastNonBlankFrame  // Use last frame where token actually appeared
          ))
          currentToken = tokenId
          startFrame = frame
          lastNonBlankFrame = frame
        } else {
          // Same token continues - update last non-blank frame
          lastNonBlankFrame = frame
        }
      } else if currentToken != nil {
        // Blank after a token - close the token span
        // The token ended at the last frame where it appeared
        let tokenString = idToToken[currentToken!] ?? ""
        alignedTokens.append(AlignedToken(
          token: tokenString,
          tokenId: currentToken!,
          startFrame: startFrame,
          endFrame: lastNonBlankFrame
        ))
        currentToken = nil  // Reset for next token
      }
    }

    // Don't forget the last token (if path ends with a non-blank)
    if let lastToken = currentToken {
      let tokenString = idToToken[lastToken] ?? ""
      alignedTokens.append(AlignedToken(
        token: tokenString,
        tokenId: lastToken,
        startFrame: startFrame,
        endFrame: lastNonBlankFrame  // Use actual last frame, not end of audio
      ))
    }

    return alignedTokens
  }
}
