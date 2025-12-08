#include "TranspositionTable.h"

#include <algorithm>
#include <limits>

// Single global instance (declared in header)
TranspositionTable TT;

void TranspositionTable::init(std::size_t megabytes)
{
  const std::size_t bytes = megabytes * 1024ULL * 1024ULL;
  const std::size_t entrySize = sizeof(TTEntry);
  const std::size_t count = entrySize ? (bytes / entrySize) : 0;

  // Round down to nearest power-of-two entry count (for fast masking).
  std::size_t power = 1;
  while (power * 2 <= count && power <= (std::numeric_limits<std::size_t>::max() / 2))
  {
    power *= 2;
  }

  tableSize = std::max<std::size_t>(1, power);
  table.assign(tableSize, TTEntry{});
}

void TranspositionTable::clear()
{
  for (auto &e : table)
  {
    e = TTEntry{};
  }
}

int TranspositionTable::pack_score(int score, int ply)
{
  // Adjust mate scores to remain root-relative; non-mate scores unchanged.
  if (score > SCORE_MATE - MAX_PLY) {
    return score - ply;
  } else if (score < -SCORE_MATE + MAX_PLY) {
    return score + ply;
  } else {
    return score;
  }
}

int TranspositionTable::unpack_score(int score, int ply)
{

  if (score > SCORE_MATE - MAX_PLY) {
    return score + ply;
  } else if (score < -SCORE_MATE + MAX_PLY) {
    return score - ply;
  } else {
    return score;
  }
}

std::size_t TranspositionTable::index_for(uint64_t key) const
{
  if (tableSize > 1)
  {
    return key & (tableSize - 1);
  }
  return 0;
}

bool TranspositionTable::probe(uint64_t,
                               int,
                               int,
                               int,
                               int,
                               int &,
                               Move &) const
{
  // Not implemented in this task (Epic 3.2 later step).
  return false;
}

void TranspositionTable::store(uint64_t,
                               int,
                               int,
                               TTFlag,
                               const Move &,
                               int)
{
  // Not implemented in this task (Epic 3.2 later step).
}
