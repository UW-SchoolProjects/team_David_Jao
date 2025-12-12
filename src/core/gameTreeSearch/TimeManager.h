#ifndef TIME_MANAGER_H
#define TIME_MANAGER_H

#include <chrono>

struct TimeBudget {
    int soft_ms;
    int hard_ms;
};

// Compute per-move soft/hard budgets from remaining time and context flags.
TimeBudget compute_time_budget(int remaining_ms,
                               int increment_ms,
                               int time_per_move_ms,
                               int moves_left_est,
                               bool pv_unstable,
                               bool in_check,
                               bool capture_heavy);

#endif // TIME_MANAGER_H
