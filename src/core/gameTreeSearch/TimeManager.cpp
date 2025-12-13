#include "TimeManager.h"

#include <algorithm>
#include <limits>

static int safe_div(int num, int denom, int fallback) {
    if (denom <= 0) return fallback;
    return num / denom;
}

TimeBudget compute_time_budget(int remaining_ms,
                               int increment_ms,
                               int time_per_move_ms,
                               int moves_left_est,
                               bool in_check,
                               bool capture_heavy)
{
    if (remaining_ms < 0) remaining_ms = 0;
    if (increment_ms < 0) increment_ms = 0;
    if (time_per_move_ms < 0) time_per_move_ms = 0;
    if (moves_left_est <= 0) moves_left_est = 30;

    // Give a small credit for increment but don't overcount it.
    int inc_credit = std::min(increment_ms * 2, remaining_ms / 2);
    int available = remaining_ms + inc_credit;

    // Base allocation: spread remaining time over estimated moves plus a small buffer.
    const int buffer = 5;
    int denom = moves_left_est + buffer;
    int base_ms = safe_div(available, denom, 10);
    if (base_ms < 10) base_ms = 10;

    // Start with the base as soft cap.
    long long soft = base_ms;

    // Bonus for tactical/volatile roots.
    if (in_check) {
        soft += base_ms / 2; // think longer when in check
    } else if (capture_heavy) {
        soft += base_ms / 3;
    }

    // Hard cap: 35% of remaining time.
    long long hard = static_cast<long long>(remaining_ms) * 35 / 100;
    if (hard <= 0 || hard > remaining_ms) {
        hard = remaining_ms;
    }

    // Respect per-move limit if provided.
    if (time_per_move_ms > 0) {
        if (soft > time_per_move_ms) soft = time_per_move_ms;
        if (hard > time_per_move_ms) hard = time_per_move_ms;
    }

    // Clamp soft to hard.
    if (soft > hard) soft = hard;
    if (soft < 1) soft = 1;

    TimeBudget out;
    out.soft_ms = static_cast<int>(std::min(soft, static_cast<long long>(std::numeric_limits<int>::max())));
    out.hard_ms = static_cast<int>(std::min(hard, static_cast<long long>(std::numeric_limits<int>::max())));
    return out;
}
