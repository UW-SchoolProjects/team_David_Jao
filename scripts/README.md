## Sampling and Testing Scripts

### Self-play sampler (`self_play_sampler.py`)
- Purpose: generate 30k–100k labeled positions (FEN + shallow eval) for the forced-capture variant.
- Prereqs: build `./program` after this patch (exposes custom CECP commands `stms`, `david_fen`, `david_lastscore`, `david_moves`).
- Recommended invocation:
  ```bash
  python3 scripts/self_play_sampler.py \
    --engine-cmd ./program \
    --positions 50000 \
    --move-time-ms 150 \
    --depth 6 \
    --output build/selfplay_positions.csv \
    --gzip
  ```
- Default filters: sample every 2 plies starting at ply 8, focus on material phase 6–18 with a 20% tail outside that band, skip near-terminal (<=2 non-king pieces).
- Randomization: plays a few random legal plies at game start using `david_moves` + `usermove` to diversify openings.
- Output schema: `fen, eval_cp, phase, ply, side_to_move, game_id, move, halfmove_clock, nonking` (CSV, optional gzip).

### Deep labeler (`deep_labeler.py`)
- Purpose: run depth+2 / ~3× time over sampled positions to produce higher-quality labels.
- Recommended invocation:
  ```bash
  python3 scripts/deep_labeler.py \
    --input build/selfplay_positions.csv.gz \
    --output build/deep_labeled_positions.csv.gz \
    --engine-cmd ./program \
    --shallow-depth 6
  ```
- Defaults: deep depth = shallow+2 (or deep_time_ms = move_time_ms*3), tanh scale 400, workers = min(cpu_count, 8).
- Output schema extends the input with `eval_deep_cp` and `eval_deep_norm`; failures logged to `build/deep_label_failures.log`.

### Opening book miner (`opening_book_miner.py`)
- Purpose: mine forced-capture self-play (NDJSON) for opening W/D/L stats in the first N plies and emit a compact book map.
- Input format (one JSON per line): `{"moves": ["e2e4", ...], "result": "1-0", "start_fen": "...", "ply_data": [{"move": "...", "fen": "...", "zobrist_key": "0x..."}, ...]}`. The `ply_data` entries may include `zobrist_key` (preferred) plus `fen`/`side`; the miner skips replay and hashing when the key is supplied. Older files can still provide `fens` or rely on engine replay.
- Recommended invocation:
  ```bash
  python3 scripts/opening_book_miner.py \
    --input build/selfplay_games.jsonl.gz \
    --max-ply 8 \
    --min-samples 6 \
    --top-n 2 \
    --output-map build/opening_book_map.json \
    --output-details build/opening_book_stats.jsonl \
    --engine-cmd ./program
  ```
- Output: compact map JSON (`hex_key -> best UCI`) plus detailed JSONL with top moves and W/D/L counts; failures logged to `build/opening_miner_fail.log`.

### Blitz gauntlet (`blitz_gauntlet.sh`)
- Stresses new builds vs a reference at 1+0 (see inline comments).
