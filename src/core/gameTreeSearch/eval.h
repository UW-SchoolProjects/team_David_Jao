#pragma once

#include "board.h" // or whatever header defines Board, Side, etc.

constexpr int SCORE_INF = 3000000;
constexpr int SCORE_MATE = 2000000;

// Function pointer type for evaluation callbacks.
using EvalFn = int (*)(const Board &);

// A simple baseline evaluation (material + bishop pair + tempo, etc.).
int basicEvaluate(const Board &board);

// You can later add more evals with the same signature:
// int pstEvaluate(const Board &board);
// int fancyEvalV2(const Board &board);
