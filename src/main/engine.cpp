
#include <iostream>
#include "../interface/cecp.h"
#include "../core/board/zobrist.h"

int main() {
    EngineSession sess;
    zobrist_init();
    init_engine_session(sess);
    cecp_main_loop(sess);
    return 0;
}
