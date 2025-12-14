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
- Output schema: `fen, eval_cp, phase, ply, side_to_move, game_id, move` (CSV, optional gzip).

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

### Blitz gauntlet (`blitz_gauntlet.sh`)
- Stresses new builds vs a reference at 1+0 (see inline comments).
