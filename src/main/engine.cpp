
#include <iostream>
#include "../interface/cecp.h"

int main() {
    EngineSession sess;
    init_engine_session(sess);
    cecp_main_loop(sess);
    return 0;
}