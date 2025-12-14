# team_David_Jao

## Overview
This repo hosts the `Team_David_Jao` chess engine plus helper scripts for Epic 5.2’s evaluation tuning workstream. Build the engine via the provided `Makefile`, then:

- sample midgame forced-capture positions (`scripts/self_play_sampler.py`),
- label them with a deeper, stabilised eval (`scripts/deep_labeler.py`),
- and reuse the engine via CECP/XBoard for stress testing (`scripts/blitz_gauntlet.sh`).

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

### Running with XBoard
1. Enable engine logging directory:
   ```bash
   mkdir -p tmp
   ```
2. Make the helper script executable:
   ```bash
   chmod +x scripts/run_engine.sh
   ```
3. Launch `xboard` with the engine:
   ```bash
   xboard -fcp ./scripts/run_engine.sh &
   tail -f tmp/engine.log
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

## Additional Scripts
- `scripts/blitz_gauntlet.sh`: run 1+0 gauntlet matches between engine builds (uses bundled `cutechess-cli` wrapper).
- `scripts/run_engine.sh`: helper to boot the engine under XBoard/CECP (set +x and use `xboard -fcp ./scripts/run_engine.sh`).

## Tests
- `python3 -m py_compile scripts/self_play_sampler.py scripts/deep_labeler.py` ensures the helpers parse.
- `make` rebuilds the engine; it is also a proxy for linking sanity.
