#include "move_generation.h"
#include "move.h"
#include "board.h"
#include <cassert>

namespace chess
{

  // ---------------------------------------------------------
  // Small bitboard utilities
  // ---------------------------------------------------------

  // Pop least significant bit and return its index (0..63).
  // Uses GCC/Clang builtin; adapt for MSVC if needed.
  static inline int popLsb(uint64_t &bb)
  {
    assert(bb != 0);
    int idx = __builtin_ctzll(bb);
    bb &= bb - 1;
    return idx;
  }

  // Convert 0..63 bit index to 0x88 square index.
  static inline int bitIndexTo0x88(int sq)
  {
    int file = sq % 8;
    int rank = sq / 8;
    return (rank << 4) | file; // MAKE_SQUARE(file, rank)
  }

  // Convert 0x88 square index to 0..63 bit index.
  static inline int square0x88ToBitIndex(int sq88)
  {
    int file = sq88 & 7;
    int rank = sq88 >> 4;
    return rank * 8 + file;
  }

  // Get captured piece type (PieceType) on a given 0..63 square,
  // using the mailbox board. Returns EMPTY if no piece.
  static inline PieceType getCapturedPieceType(const Board &board, int toBitIndex)
  {
    int sq88 = bitIndexTo0x88(toBitIndex);
    int piece = board.squares[sq88];
    if (piece == EMPTY)
      return EMPTY;
    return static_cast<PieceType>(type_of(piece)); // piece & 7
  }

  // ---------------------------------------------------------
  // External attack tables (to be implemented elsewhere)
  // ---------------------------------------------------------

  extern uint64_t knightAttacks[64];
  extern uint64_t kingAttacks[64];

  uint64_t bishopAttacks(int sq, uint64_t occ);
  uint64_t rookAttacks(int sq, uint64_t occ);

  // ---------------------------------------------------------
  // Forward declarations for piece-type generators (Parent Job 1.2)
  // ---------------------------------------------------------

  static void generateKnightMoves(const Board &board, Side side, MoveList &pseudoMoves);
  static void generateKingMoves(const Board &board, Side side, MoveList &pseudoMoves);
  static void generateBishopMoves(const Board &board, Side side, MoveList &pseudoMoves);
  static void generateRookMoves(const Board &board, Side side, MoveList &pseudoMoves);
  static void generateQueenMoves(const Board &board, Side side, MoveList &pseudoMoves);
  static void generatePawnMoves(const Board &board, Side side, MoveList &pseudoMoves);

  // ---------------------------------------------------------
  // Parent Job 1.2: Pseudolegal move generation
  // ---------------------------------------------------------

  /**
   * Generate pseudolegal moves (ignores self-check) for a side.
   * This is the "Parent Job 1.2" umbrella, which delegates to the
   * specific piece-type generators.
   */
  static void generatePseudoLegalMoves(const Board &board,
                                       Side side,
                                       MoveList &pseudoMoves)
  {
    pseudoMoves.clear();

    generateKnightMoves(board, side, pseudoMoves);
    generateKingMoves(board, side, pseudoMoves);
    generateBishopMoves(board, side, pseudoMoves);
    generateRookMoves(board, side, pseudoMoves);
    generateQueenMoves(board, side, pseudoMoves);
    generatePawnMoves(board, side, pseudoMoves);
  }

  // ---------------------------------------------------------
  // 1.2.1 — Knights
  // ---------------------------------------------------------

  static void generateKnightMoves(const Board &board, Side side, MoveList &pseudoMoves)
  {
    uint64_t myKnights = (side == WHITE) ? board.wknights : board.bknights;
    if (!myKnights)
      return;

    uint64_t myOcc = (side == WHITE) ? board.occWhite : board.occBlack;
    uint64_t oppOcc = (side == WHITE) ? board.occBlack : board.occWhite;

    while (myKnights)
    {
      int from = popLsb(myKnights);
      uint64_t targets = knightAttacks[from] & ~myOcc;

      uint64_t quiets = targets & ~oppOcc;
      uint64_t captures = targets & oppOcc;

      // Quiet moves
      uint64_t bb = quiets;
      while (bb)
      {
        int to = popLsb(bb);
        Move m(from, to, MF_QUIET, EMPTY, EMPTY);
        pseudoMoves.add(m);
      }

      // Captures
      bb = captures;
      while (bb)
      {
        int to = popLsb(bb);
        PieceType captured = getCapturedPieceType(board, to);
        Move m(from, to, MF_CAPTURE, EMPTY, captured);
        pseudoMoves.add(m);
      }
    }
  }

  // ---------------------------------------------------------
  // 1.2.1 — King (no castling yet; just one-step moves)
  // ---------------------------------------------------------

  static void generateKingMoves(const Board &board, Side side, MoveList &pseudoMoves)
  {
    uint64_t myKing = (side == WHITE) ? board.wking : board.bking;
    if (!myKing)
      return;

    uint64_t myOcc = (side == WHITE) ? board.occWhite : board.occBlack;
    uint64_t oppOcc = (side == WHITE) ? board.occBlack : board.occWhite;

    int from = __builtin_ctzll(myKing); // only one king
    uint64_t targets = kingAttacks[from] & ~myOcc;

    uint64_t quiets = targets & ~oppOcc;
    uint64_t captures = targets & oppOcc;

    uint64_t bb = quiets;
    while (bb)
    {
      int to = popLsb(bb);
      Move m(from, to, MF_QUIET, EMPTY, EMPTY);
      pseudoMoves.add(m);
    }

    bb = captures;
    while (bb)
    {
      int to = popLsb(bb);
      PieceType captured = getCapturedPieceType(board, to);
      Move m(from, to, MF_CAPTURE, EMPTY, captured);
      pseudoMoves.add(m);
    }

    // Castling will be added later (pseudolegal: check only empty & rights)
  }

  // ---------------------------------------------------------
  // 1.2.2 — Bishops
  // ---------------------------------------------------------

  static void generateBishopMoves(const Board &board, Side side, MoveList &pseudoMoves)
  {
    uint64_t myBishops = (side == WHITE) ? board.wbishops : board.bbishops;
    if (!myBishops)
      return;

    uint64_t myOcc = (side == WHITE) ? board.occWhite : board.occBlack;
    uint64_t oppOcc = (side == WHITE) ? board.occBlack : board.occWhite;

    while (myBishops)
    {
      int from = popLsb(myBishops);
      uint64_t attacks = bishopAttacks(from, board.occAll) & ~myOcc;

      uint64_t quiets = attacks & ~oppOcc;
      uint64_t captures = attacks & oppOcc;

      uint64_t bb = quiets;
      while (bb)
      {
        int to = popLsb(bb);
        Move m(from, to, MF_QUIET, EMPTY, EMPTY);
        pseudoMoves.add(m);
      }

      bb = captures;
      while (bb)
      {
        int to = popLsb(bb);
        PieceType captured = getCapturedPieceType(board, to);
        Move m(from, to, MF_CAPTURE, EMPTY, captured);
        pseudoMoves.add(m);
      }
    }
  }

  // ---------------------------------------------------------
  // 1.2.2 — Rooks
  // ---------------------------------------------------------

  static void generateRookMoves(const Board &board, Side side, MoveList &pseudoMoves)
  {
    uint64_t myRooks = (side == WHITE) ? board.wrooks : board.brooks;
    if (!myRooks)
      return;

    uint64_t myOcc = (side == WHITE) ? board.occWhite : board.occBlack;
    uint64_t oppOcc = (side == WHITE) ? board.occBlack : board.occWhite;

    while (myRooks)
    {
      int from = popLsb(myRooks);
      uint64_t attacks = rookAttacks(from, board.occAll) & ~myOcc;

      uint64_t quiets = attacks & ~oppOcc;
      uint64_t captures = attacks & oppOcc;

      uint64_t bb = quiets;
      while (bb)
      {
        int to = popLsb(bb);
        Move m(from, to, MF_QUIET, EMPTY, EMPTY);
        pseudoMoves.add(m);
      }

      bb = captures;
      while (bb)
      {
        int to = popLsb(bb);
        PieceType captured = getCapturedPieceType(board, to);
        Move m(from, to, MF_CAPTURE, EMPTY, captured);
        pseudoMoves.add(m);
      }
    }
  }

  // ---------------------------------------------------------
  // 1.2.2 — Queens (bishop + rook)
  // ---------------------------------------------------------

  static void generateQueenMoves(const Board &board, Side side, MoveList &pseudoMoves)
  {
    uint64_t myQueens = (side == WHITE) ? board.wqueens : board.bqueens;
    if (!myQueens)
      return;

    uint64_t myOcc = (side == WHITE) ? board.occWhite : board.occBlack;
    uint64_t oppOcc = (side == WHITE) ? board.occBlack : board.occWhite;

    while (myQueens)
    {
      int from = popLsb(myQueens);
      uint64_t attacks =
          (bishopAttacks(from, board.occAll) |
           rookAttacks(from, board.occAll)) &
          ~myOcc;

      uint64_t quiets = attacks & ~oppOcc;
      uint64_t captures = attacks & oppOcc;

      uint64_t bb = quiets;
      while (bb)
      {
        int to = popLsb(bb);
        Move m(from, to, MF_QUIET, EMPTY, EMPTY);
        pseudoMoves.add(m);
      }

      bb = captures;
      while (bb)
      {
        int to = popLsb(bb);
        PieceType captured = getCapturedPieceType(board, to);
        Move m(from, to, MF_CAPTURE, EMPTY, captured);
        pseudoMoves.add(m);
      }
    }
  }

  // ---------------------------------------------------------
  // 1.2.3 — Pawns, Promotions & En Passant
  // ---------------------------------------------------------

  static void generatePawnMoves(const Board &board, Side side, MoveList &pseudoMoves)
  {
    uint64_t myPawns = (side == WHITE) ? board.wpawns : board.bpawns;
    if (!myPawns)
      return;

    uint64_t myOcc = (side == WHITE) ? board.occWhite : board.occBlack;
    uint64_t oppOcc = (side == WHITE) ? board.occBlack : board.occWhite;
    uint64_t occAll = board.occAll;

    // EP target as bit index, if any
    int epBit = -1;
    if (board.ep_square != NO_SQUARE)
    {
      epBit = square0x88ToBitIndex(board.ep_square);
    }

    while (myPawns)
    {
      int from = popLsb(myPawns);
      int rank = from / 8;
      int file = from % 8;

      if (side == WHITE)
      {
        // --- Single push ---
        int oneStep = from + 8;
        if (oneStep < 64 && !(occAll & (1ULL << oneStep)))
        {
          int toRank = oneStep / 8;

          if (toRank == 7)
          {
            // Promotion quiet (to rank 8)
            PieceType promos[4] = {QUEEN, ROOK, BISHOP, KNIGHT};
            for (PieceType pt : promos)
            {
              Move m(from, oneStep, MF_PROMOTION, pt, EMPTY);
              pseudoMoves.add(m);
            }
          }
          else
          {
            Move m(from, oneStep, MF_QUIET, EMPTY, EMPTY);
            pseudoMoves.add(m);

            // --- Double push from rank 2 (rank == 1) ---
            if (rank == 1)
            {
              int twoStep = from + 16;
              if (!(occAll & (1ULL << twoStep)))
              {
                Move dm(from, twoStep, MF_DOUBLE_PAWN_PUSH, EMPTY, EMPTY);
                pseudoMoves.add(dm);
              }
            }
          }
        }

        // --- Captures (including promo captures) ---
        // Left capture (from White's POV): +7 (file > 0)
        if (file > 0)
        {
          int to = from + 7;
          if (to < 64)
          {
            uint64_t mask = 1ULL << to;
            if (oppOcc & mask)
            {
              int toRank = to / 8;
              PieceType captured = getCapturedPieceType(board, to);
              if (toRank == 7)
              {
                // Promotion capture
                PieceType promos[4] = {QUEEN, ROOK, BISHOP, KNIGHT};
                for (PieceType pt : promos)
                {
                  Move m(from, to, static_cast<MoveFlag>(MF_CAPTURE | MF_PROMOTION),
                         pt, captured);
                  pseudoMoves.add(m);
                }
              }
              else
              {
                Move m(from, to, MF_CAPTURE, EMPTY, captured);
                pseudoMoves.add(m);
              }
            }
            // En passant capture
            if (epBit == to)
            {
              Move m(from, to, static_cast<MoveFlag>(MF_CAPTURE | MF_EN_PASSANT_CAPTURE),
                     EMPTY, PAWN);
              pseudoMoves.add(m);
            }
          }
        }

        // Right capture: +9 (file < 7)
        if (file < 7)
        {
          int to = from + 9;
          if (to < 64)
          {
            uint64_t mask = 1ULL << to;
            if (oppOcc & mask)
            {
              int toRank = to / 8;
              PieceType captured = getCapturedPieceType(board, to);
              if (toRank == 7)
              {
                PieceType promos[4] = {QUEEN, ROOK, BISHOP, KNIGHT};
                for (PieceType pt : promos)
                {
                  Move m(from, to, static_cast<MoveFlag>(MF_CAPTURE | MF_PROMOTION),
                         pt, captured);
                  pseudoMoves.add(m);
                }
              }
              else
              {
                Move m(from, to, MF_CAPTURE, EMPTY, captured);
                pseudoMoves.add(m);
              }
            }
            // En passant capture
            if (epBit == to)
            {
              Move m(from, to, static_cast<MoveFlag>(MF_CAPTURE | MF_EN_PASSANT_CAPTURE),
                     EMPTY, PAWN);
              pseudoMoves.add(m);
            }
          }
        }
      }
      else
      {
        // side == BLACK
        // --- Single push ---
        int oneStep = from - 8;
        if (oneStep >= 0 && !(occAll & (1ULL << oneStep)))
        {
          int toRank = oneStep / 8;
          if (toRank == 0)
          {
            // Promotion quiet to rank 1
            PieceType promos[4] = {QUEEN, ROOK, BISHOP, KNIGHT};
            for (PieceType pt : promos)
            {
              Move m(from, oneStep, MF_PROMOTION, pt, EMPTY);
              pseudoMoves.add(m);
            }
          }
          else
          {
            Move m(from, oneStep, MF_QUIET, EMPTY, EMPTY);
            pseudoMoves.add(m);

            // Double push from rank 7 (rank == 6)
            if (rank == 6)
            {
              int twoStep = from - 16;
              if (!(occAll & (1ULL << twoStep)))
              {
                Move dm(from, twoStep, MF_DOUBLE_PAWN_PUSH, EMPTY, EMPTY);
                pseudoMoves.add(dm);
              }
            }
          }
        }

        // --- Captures ---
        // Left capture (from Black's POV): -9 (file > 0)
        if (file > 0)
        {
          int to = from - 9;
          if (to >= 0)
          {
            uint64_t mask = 1ULL << to;
            if (oppOcc & mask)
            {
              int toRank = to / 8;
              PieceType captured = getCapturedPieceType(board, to);
              if (toRank == 0)
              {
                PieceType promos[4] = {QUEEN, ROOK, BISHOP, KNIGHT};
                for (PieceType pt : promos)
                {
                  Move m(from, to, static_cast<MoveFlag>(MF_CAPTURE | MF_PROMOTION),
                         pt, captured);
                  pseudoMoves.add(m);
                }
              }
              else
              {
                Move m(from, to, MF_CAPTURE, EMPTY, captured);
                pseudoMoves.add(m);
              }
            }
            // En passant capture
            if (epBit == to)
            {
              Move m(from, to, static_cast<MoveFlag>(MF_CAPTURE | MF_EN_PASSANT_CAPTURE),
                     EMPTY, PAWN);
              pseudoMoves.add(m);
            }
          }
        }

        // Right capture: -7 (file < 7)
        if (file < 7)
        {
          int to = from - 7;
          if (to >= 0)
          {
            uint64_t mask = 1ULL << to;
            if (oppOcc & mask)
            {
              int toRank = to / 8;
              PieceType captured = getCapturedPieceType(board, to);
              if (toRank == 0)
              {
                PieceType promos[4] = {QUEEN, ROOK, BISHOP, KNIGHT};
                for (PieceType pt : promos)
                {
                  Move m(from, to, static_cast<MoveFlag>(MF_CAPTURE | MF_PROMOTION),
                         pt, captured);
                  pseudoMoves.add(m);
                }
              }
              else
              {
                Move m(from, to, MF_CAPTURE, EMPTY, captured);
                pseudoMoves.add(m);
              }
            }
            // En passant capture
            if (epBit == to)
            {
              Move m(from, to, static_cast<MoveFlag>(MF_CAPTURE | MF_EN_PASSANT_CAPTURE),
                     EMPTY, PAWN);
              pseudoMoves.add(m);
            }
          }
        }
      }
    }
  }

  /**
   * Task 2.1.1 — Check detection.
   * Returns true if `side`'s king is currently in check.
   */
  static bool isInCheck(const Board &board, Side side);

  /**
   * Task 2.2 — Forced capture rule.
   * If captureOnly is true and at least one capture exists in `moves`,
   * remove all non-captures.
   */
  static void applyCaptureOnlyFilter(MoveList &moves, bool captureOnly);

  // --- Public API implementation ---

  void validMoveGeneration(const Board &board,
                           Side side,
                           MoveList &outMoves,
                           bool captureOnly)
  {
    outMoves.clear();

    // 1) Generate all pseudolegal moves for the side.
    MoveList pseudoMoves;
    pseudoMoves.clear();
    generatePseudoLegalMoves(board, side, pseudoMoves);

    // 2) Filter to legal moves by testing for self-check.
    //    This will eventually use makeMove/unmakeMove + isInCheck.
    for (int i = 0; i < pseudoMoves.count; ++i)
    {
      const Move &m = pseudoMoves.moves[i];

      // TODO (Task 2.1.2):
      // - Make a copy or use an undo stack with makeMove/unmakeMove.
      // - Apply the move.
      // - If the `side` is not in check in the resulting position, keep it.
      // - Undo the move.
      //
      // Pseudocode:
      //
      // Board copy = board;
      // Undo undo;
      // if (!makeMove(copy, m, undo)) {
      //     continue; // illegal for some reason
      // }
      // if (!isInCheck(copy, side)) {
      //     outMoves.add(m);
      // }

      // TEMPORARY placeholder to keep the function compiling:
      // Remove this once you implement make-move + legality.
      (void)m; // suppress unused warning
    }

    // 3) Apply the optional "capture-only" variant rule.
    applyCaptureOnlyFilter(outMoves, captureOnly);
  }

  static bool isInCheck(const Board & /*board*/, Side /*side*/)
  {
    // TODO: Task 2.1.1 — attack detection onto king square.
    // This should reuse the same attack primitives used in movegen.
    return false;
  }

  static void applyCaptureOnlyFilter(MoveList &moves, bool captureOnly)
  {
    if (!captureOnly || moves.empty())
    {
      return;
    }

    // First pass: count captures.
    MoveList captures;
    captures.clear();

    for (int i = 0; i < moves.count; ++i)
    {
      const Move &m = moves.moves[i];

      // TODO: Once Move has isCapture(), use it here.
      // if (m.isCapture()) captures.add(m);

      (void)m; // placeholder
    }

    // If any capture exists, replace moves with captures.
    if (!captures.empty())
    {
      moves = captures;
    }
  }

} // namespace chess
