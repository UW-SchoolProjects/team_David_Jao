/*
Simple clock-handling simulation test for CECP time management.
Simulates known elapsed times for both sides and asserts remaining clocks.
*/

#include "../src/interface/cecp.h"
#include "../src/core/board/zobrist.h"
#include "../src/core/nextMoveGeneration/attacks.h"

#include <cassert>
#include <chrono>

int main() {
    zobrist_init();
    initAttackTables();

    EngineSession sess;
    init_engine_session(sess);

    // Set clocks via CECP commands (centiseconds -> milliseconds)
    handle_time(sess, 1500);  // 15 seconds
    handle_otim(sess, 2000);  // 20 seconds
    assert(sess.my_time_ms == 15000);
    assert(sess.opp_time_ms == 20000);

    // Opponent spends 1.2 seconds before their move arrives
    sess.last_opp_move_ts = std::chrono::steady_clock::now() - std::chrono::milliseconds(1200);
    apply_opp_move_elapsed(sess, 1200, std::chrono::steady_clock::now());
    assert(sess.opp_time_ms == 18800);

    // We think for 2.5 seconds and have a 2-second increment
    sess.my_time_ms = 15000;
    sess.increment_ms = 2000;
    auto my_move_end = std::chrono::steady_clock::now();
    apply_my_move_elapsed(sess, 2500, my_move_end);
    // 15.0s - 2.5s + 2.0s = 14.5s
    assert(sess.my_time_ms == 14500);
    assert(sess.last_my_move_ts == my_move_end);
    assert(sess.last_opp_move_ts == my_move_end);

    // Ensure clamping at zero when elapsed exceeds remaining
    sess.opp_time_ms = 500;
    auto now = std::chrono::steady_clock::now();
    apply_opp_move_elapsed(sess, 1000, now);
    assert(sess.opp_time_ms == 0);
    assert(sess.last_opp_move_ts == now);

    return 0;
}
