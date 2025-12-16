# team_David_Jao

## Overview
This repo hosts the `Team_David_Jao` chess engine plus helper scripts for Epic 5.2’s evaluation tuning workstream. Build the engine via the provided `Makefile`, then:

- sample midgame forced-capture positions (`scripts/self_play_sampler.py`),
- label them with a deeper, stabilised eval (`scripts/deep_labeler.py`),
- persist complete NDJSON self-play games for opening analytics (`scripts/self_play_games.py` + `scripts/opening_book_miner.py`),
- tune PST/eval weights with Texel (`scripts/texel_tuner.py`), and
- reuse the engine via CECP/XBoard for stress testing (`scripts/blitz_gauntlet.sh`).

## Prerequisites
- Linux toolchain (`g++`, `make`).
- Python 3.8+ with standard library modules (`argparse`, `csv`, `gzip`, `math`, `subprocess`, `select`, `threading`).
- `cutechess-cli` (bundled under `scripts/`) for match harnesses.

## Building the Engine
1. Install prerequisites (Debian/Ubuntu example):
   ```bash
   sudo apt update
   sudo apt install build-essential g++ make python3 xboard
   ```
2. Build the engine:
   ```bash
   make
   ```
   This emits `./program`, a CECP-compatible binary that automatically registers the `david_*` helpers.

By default `make` now invokes `scripts/self_play_games.py` and `scripts/opening_book_miner.py` after linking so `build/selfplay_games.jsonl`, `build/opening_book_map.json`, and the accompanying stats/logs are regenerated whenever the binary changes. Override `PYTHON`/`SELFPLAY_GAMES`/`OPENING_BOOK_MAP` etc. if you need different paths or tuning parameters, or set `SKIP_POST_BUILD=1` to skip the post-build pipeline during quick iterations.

### Running with XBoard
1. (Optional) enable verbose instrumentation by rebuilding with logging:
   ```bash
   LOG=1 make
   ```
2. Launch `xboard` (or any CECP client such as `cutechess-cli`) by pointing it directly at the engine binary:
   ```bash
   xboard -fcp "./program"
   ```
3. To persist `stderr` metrics, wrap the command so the shell redirects output for you:
   ```bash
   mkdir -p build/logs
   xboard -fcp "./program 2>build/logs/engine.log" &
   tail -f build/logs/engine.log
   ```
   Use XBoard to play, watch the log for custom `METRIC` entries, and stop with `fg` + Ctrl+C or `kill`.

## Self-Play Sampling (Task 5.2.1)
Generates `build/selfplay_positions.csv(.gz)` containing columns `fen, eval_cp, phase, ply, side_to_move, game_id, move`. The sampler assumes the engine runs single-threaded—avoid enabling multi-threaded search or thread-local pools when launching `./program`.
The sampler:

1. Runs forced-capture self-play at ~0.15s/move.
2. Records midgame FENs every 2 plies once phase ≥6.
3. Logs shallow engine scores (`eval_cp`) and additional context.

Example invocation:
```bash
python3 scripts/self_play_sampler.py \
  --engine-cmd ./program \
  --positions 50000 \
  --move-time-ms 150 \
  --depth 6 \
  --output build/selfplay_positions.csv \
  --gzip
```

## Deep Labeling (Task 5.2.2)
Re-runs the single-threaded engine at `depth+2` (or ~3× the shallow time) to compute high-quality targets and stores them in `build/deep_labeled_positions.csv.gz` with two extra fields: `eval_deep_cp` and `eval_deep_norm` (tanh-normalized).

```bash
python3 scripts/deep_labeler.py \
  --input build/selfplay_positions.csv.gz \
  --output build/deep_labeled_positions.csv.gz \
  --engine-cmd ./program \
  --shallow-depth 6
```

Failures (timeouts, malformed FENs, etc.) are recorded in `build/deep_label_failures.log`.

## Self-Play Games (NDJSON)
`scripts/self_play_games.py` emits NDJSON that captures every ply plus zobrist keys, which is the preferred input for `scripts/opening_book_miner.py`.

```bash
python3 scripts/self_play_games.py \
  --engine-cmd ./program \
  --games 100 \
  --max-plies 80 \
  --move-time-ms 150 \
  --depth 6 \
  --output build/selfplay_games.jsonl
```

Set `--gzip` if you want compressed output; the JSON lines already include `ply_data` entries with zobrist keys so the miner can avoid replaying moves when possible.

### Opening Book (required)
`src/interface/cecp.cpp:212-276` loads `build/opening_book_map.json` (configurable via `ENGINE_BOOK_PATH`) and the engine refuses to start without it. To keep that book in sync with a new build, feed the NDJSON into `scripts/opening_book_miner.py` so it can recompute the map and stats:

```bash
python3 scripts/opening_book_miner.py \
  --input build/selfplay_games.jsonl \
  --output-map build/opening_book_map.json \
  --output-details build/opening_book_stats.jsonl \
  --engine-cmd ./program \
  --max-ply 8 \
  --min-samples 6 \
  --top-n 2
```

The miner also logs invalid games to `build/opening_miner_fail.log` and writes the raw stats JSONL so you can inspect the move distributions. Because both the NDJSON games and the derived map can be large and change whenever the engine is rebuilt, we intentionally do not commit these generated artifacts; rerun the two scripts whenever you need fresh opening coverage instead of checking the outputs into git.

## Additional Scripts
- `scripts/blitz_gauntlet.sh`: run 1+0 gauntlet matches between engine builds (uses the bundled `cutechess-cli` wrapper if present).
- `scripts/self_play_games.py`: produce NDJSON game records for downstream opening-book analytics.
- `scripts/opening_book_miner.py`: mine NDJSON self-play for opening W/D/L stats over the first N plies, emitting a compact book map (`hex_key -> best_move`) plus detailed per-move counts (prefers `zobrist_key` in each `ply_data` entry, otherwise replays plies through `./program`).
- `scripts/texel_tuner.py`: Texel logistic/centipawn regression that learns PST tables and small eval terms, outputting a header suitable for `src/core/gameTreeSearch/eval.cpp`.

## Tests
- `python3 -m py_compile scripts/self_play_sampler.py scripts/deep_labeler.py` ensures the helpers parse.
- `make` rebuilds the engine; it is also a proxy for linking sanity.

## References

The following sources were consulted during the design and implementation of our chess engine.  
Each reference includes a hyperlink and a short explanation of how it influenced the project.  
*Accessed November 2025.*

---

### General Search & Engine Architecture

1. **Alpha–Beta Algorithm – ChessProgramming.org**  
   https://www.chessprogramming.org/Alpha-Beta  
   Used to guide the structure of our alpha–beta search, including negamax formulation, bound propagation, and search loop design.

2. **Alpha–beta pruning – Wikipedia**  
   https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning  
   Provided background on alpha–beta pruning theory, pruning correctness, and complexity reductions.

3. **Move Ordering – ChessProgramming.org**  
   https://www.chessprogramming.org/Move_Ordering  
   Helped determine our move ordering strategy (TT move → captures → killers → history) to improve pruning efficiency.

---

### Hashing & Transposition Tables

4. **A New Hashing Method with Application for Game Playing – Albert L. Zobrist**  
   https://research.cs.wisc.edu/techreports/1970/TR88.pdf  
   The original paper introducing Zobrist hashing, which we used as the basis for our transposition table key generation.

5. **Albert Zobrist – ChessProgramming.org**  
   https://www.chessprogramming.org/Albert_Zobrist  
   Provided practical explanations of how Zobrist hashing is used in modern chess engines and informed our TT implementation.

---

### Engine Communication Protocol (WinBoard / XBoard / CECP)

6. **Chess Engine Communication Protocol (CECP) – GNU XBoard / WinBoard**  
   https://www.gnu.org/software/xboard/engine-intf.html  
   Primary protocol specification used to implement our CECP (XBoard) command loop (`xboard`, `protover`, `go`, `usermove`, `ping`, `time`, etc.).

7. **Chess Engine Communication Protocol – ChessProgramming.org**  
   https://www.chessprogramming.org/Chess_Engine_Communication_Protocol  
   Served as a concise summary of required CECP commands and expected engine behaviors.

8. **Chess-Engine Communication Protocol v2 – H.G. Muller**  
   https://hgm.nubati.net/CECP.html  
   Used to clarify subtle CECP behaviors, such as `ping/pong`, timing rules, and engine–GUI interaction details.

9. **Tim Mann – Chess Engines / CECP Guidance**  
   https://www.tim-mann.org/engines.html  
   Provided additional insights on implementing XBoard-compatible engines, especially feature negotiation and testing.

---

### Tutorial Engines / Code Examples

10. **Tom Kerrigan’s Simple Chess Program (TSCP)**  
    https://www.tckerrigan.com/Chess/TSCP/  
    Used as a conceptual reference for understanding basic engine structure (move generation, search loop, evaluation framework).

11. **TSCP – ChessProgramming.org**  
    https://www.chessprogramming.org/TSCP  
    Helped contextualize TSCP as a tutorial engine and influenced some architectural decisions.

12. **TSCP Source Code (example repository)**  
    https://github.com/terredeciels/TSCP/blob/master/main.c  
    Examined for insight into how a small chess engine wires together its main loop and search driver (no code was copied).

---

### Evaluation & Tuning

13. **Texel’s Tuning Method – ChessProgramming.org**  
    https://www.chessprogramming.org/Texel%27s_Tuning_Method  
    Provided the theoretical foundation for our evaluation tuning pipeline using logistic regression (Texel tuning).

14. **Zurichess Evaluation Improvements – Alexandru Moșoi**  
    https://medium.com/%40brtzsnr/hi-all-a73c1b7b7a73  
    Practical example showing how Texel tuning improves PST and evaluation coefficients in real engines.

15. **Rofchade – Technical Notes (Tuning)**  
    https://rofchade.nl/?page_id=116  
    Referenced for understanding realistic Elo gains and methodology for tuning classical evaluation terms.

16. **“Understanding evaluation function” – Chess StackExchange**  
    https://chess.stackexchange.com/questions/6178/understanding-evaluation-function  
    Helped reinforce our understanding of classical evaluation components (material, PSTs, mobility) and why tuning them matters.

---

### Optional / Background References  
(Include only if actually used.)

17. **Chess Engine, pt. 4: α–β pruning and better search – DogeyStamp**  
    https://www.dogeystamp.com/chess4/  
    Tutorial-style explanation of alpha–beta pruning; helpful during early search development.

18. **“Which interface protocol should I implement for my chess engine?” – Chess StackExchange**  
    https://chess.stackexchange.com/questions/6110/which-interface-protocol-should-i-implement-for-my-chess-engine  
    Helped justify CECP as the required protocol and clarified differences from UCI.

---
