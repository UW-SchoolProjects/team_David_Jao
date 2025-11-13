#include "board.h"

int main() {
    Board b = {};
    setup_startpos(b);
    print_board(b);

    return 0;
}