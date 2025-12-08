#ifndef TRANSPOSITION_TABLE_H
#define TRANSPOSITION_TABLE_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "../nextMoveGeneration/move.h"
#include "../common/constants.h"
#include "eval.h"

/**
 * TT entry payload. Kept POD and compact so we can size the table by bytes.
 */
struct TTEntry
{
  uint64_t key = 0;    // Zobrist hash for the stored position
  int16_t depth = std::numeric_limits<int16_t>::min(); // search depth of this record
  int32_t value = 0;   // stored score (mate scores are packed with ply at store time)
  uint8_t flag = 0;    // TTFlag as its underlying value
  Move bestMove{};     // hash move to aid move ordering
};

/**
 * Meaning of a stored score relative to the original alpha/beta window.
 */
enum class TTFlag : uint8_t
{
  EXACT = 0,      // Score is exact; usable directly.
  LOWERBOUND = 1, // Score is >= beta (beta cutoff).
  UPPERBOUND = 2  // Score is <= alpha (alpha cutoff).
};

/**
 * Transposition table interface (no implementation details here).
 *
 * Usage sketch:
 *   TT.init(128);                 // allocate ~128 MB
 *   TT.clear();                   // wipe for new game
 *   if (TT.probe(...)) ...        // query
 *   TT.store(...);                // insert/update
 */
class TranspositionTable {
  public:
    TranspositionTable() = default;
    TranspositionTable(const TranspositionTable&) = delete;
    TranspositionTable& operator=(const TranspositionTable&) = delete;
    TranspositionTable(TranspositionTable&&) = delete;
    TranspositionTable& operator=(TranspositionTable&&) = delete;

    /**
     * Allocate and size the table based on requested megabytes.
     * Megabytes are interpreted as MB = 1024*1024 bytes.
     * Entry count = floor(bytes / sizeof(TTEntry)). Always at least 1 entry.
     */
    void init(std::size_t megabytes = 128);

    /**
     * Wipe all entries to an "empty" state (key=0, depth=min, flag=0, move=null).
     * Call this at the start of each new game so stale data does not leak.
     */
    void clear();

    /**
     * Probe the table.
     * - key: current position Zobrist.
     * - depth: required search depth; shallower entries are treated as misses.
     * - alpha/beta: current window; used with flags to decide if the entry is usable.
     * - ply: distance from root; needed to unpack mate scores correctly.
     * On hit with sufficient depth and compatible flag, returns true and fills outScore/outMove.
     */
    bool probe(uint64_t key, int depth, int alpha, int beta, int ply, int &outScore, Move &outMove) const;

    /**
     * Store an entry.
     * - key: position Zobrist.
     * - depth: search depth of this node.
     * - score: value to store (mate scores will be packed with ply).
     * - flag: EXACT / LOWERBOUND / UPPERBOUND meaning of the stored score.
     * - bestMove: hash move for move ordering (can be null).
     * - ply: distance from root; used to pack mate scores so they remain root-relative.
     *
     * Replacement policy: depth-preferred (replace if the incoming depth is >= stored depth).
     */
    void store(uint64_t key, int depth, int score, TTFlag flag, const Move &bestMove, int ply);

    /**
     * Number of entries currently allocated (after init()).
     */
    std::size_t size() const { return tableSize; }

  private:
    std::vector<TTEntry> table;
    std::size_t tableSize = 0;

    // Helpers (implemented in .cpp):
    static int pack_score(int score, int ply);
    static int unpack_score(int score, int ply);
    std::size_t index_for(uint64_t key) const;
};

// Global instance (defined in the .cpp).
extern TranspositionTable TT;

#endif // TRANSPOSITION_TABLE_H
