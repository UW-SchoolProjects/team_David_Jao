/*
Basic checks for compute_time_budget behavior.
*/

#include "../src/core/gameTreeSearch/TimeManager.h"
#include <cassert>

int main() {
    // Baseline: 60s remaining, no bonuses.
    TimeBudget b1 = compute_time_budget(60000, 0, 0, 30, false, false, false);
    // base = 60000 / (30+5) = 1714ms, hard = 21000ms
    assert(b1.soft_ms > 1000 && b1.soft_ms < b1.hard_ms);
    assert(b1.hard_ms == 21000);

    // In check should increase soft cap but stay under hard cap.
    TimeBudget b2 = compute_time_budget(60000, 0, 0, 30, false, true, false);
    assert(b2.soft_ms > b1.soft_ms);
    assert(b2.soft_ms <= b2.hard_ms);

    // Capture heavy and unstable should also bump soft.
    TimeBudget b3 = compute_time_budget(60000, 0, 0, 30, true, false, true);
    assert(b3.soft_ms > b1.soft_ms);

    // Hard cap should limit runaway soft allocations.
    TimeBudget b4 = compute_time_budget(5000, 0, 0, 5, true, true, true);
    assert(b4.soft_ms <= b4.hard_ms);
    assert(b4.hard_ms <= 5000);

    return 0;
}
