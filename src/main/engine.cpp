#include "board.h"
#include <iostream>

using namespace std;

int main() {
    Board b = {};
    setup_startpos(b);
    print_board(b);

    if (!assert_bb_consistency(b)) {
        cout << "bit board error!" << endl;
        return -1;
    }

    return 0;
}