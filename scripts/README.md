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

### Blitz gauntlet (`blitz_gauntlet.sh`)
- Stresses new builds vs a reference at 1+0 (see inline comments).
