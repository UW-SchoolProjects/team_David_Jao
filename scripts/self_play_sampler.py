#!/usr/bin/env python3
"""Rapid self-play sampler for the forced-capture variant.

Outputs CSV rows: fen, eval_cp, phase, ply, side_to_move, game_id, move
Requires engine exposing custom CECP commands:
  - stms <ms>            (fixed time-per-move in milliseconds)
  - david_fen            (prints 'fen <fen-string>')
  - david_lastscore      (prints 'lastscore <cp>|none')
  - david_moves          (prints 'moves <uci>...')
"""

import argparse
import csv
import gzip
import os
import random
import select
import shlex
import subprocess
import sys
import time
from typing import Callable, List, Optional


def send(proc: subprocess.Popen, cmd: str) -> bool:
    try:
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()
        return True
    except BrokenPipeError:
        return False


def read_until(proc: subprocess.Popen, timeout: float, predicate: Callable[[str], bool]) -> Optional[str]:
    end = time.time() + timeout
    while time.time() < end:
        remaining = end - time.time()
        ready, _, _ = select.select([proc.stdout], [], [], remaining)
        if proc.stdout in ready:
            line = proc.stdout.readline()
            if line == "":
                return None
            line = line.strip()
            if not line:
                continue
            if predicate(line):
                return line
    return None


def query_prefix(proc: subprocess.Popen, cmd: str, prefix: str, timeout: float = 1.0) -> Optional[str]:
    if not send(proc, cmd):
        return None
    return read_until(proc, timeout, lambda line: line.startswith(prefix))


def material_phase(fen: str) -> int:
    fields = fen.split()
    if not fields:
        return 0
    placement = fields[0]
    weights = {"p": 0, "n": 1, "b": 1, "r": 2, "q": 4, "k": 0}
    phase = 0
    for ch in placement:
        if ch in "/12345678":
            continue
        phase += weights.get(ch.lower(), 0)
    return phase


def nonking_material(fen: str) -> int:
    fields = fen.split()
    if not fields:
        return 0
    placement = fields[0]
    count = 0
    for ch in placement:
        if ch in "/12345678":
            continue
        if ch.lower() != "k":
            count += 1
    return count


def drain(proc: subprocess.Popen, timeout: float = 0.05) -> None:
    end = time.time() + timeout
    while time.time() < end:
        remaining = end - time.time()
        ready, _, _ = select.select([proc.stdout], [], [], remaining)
        if not ready:
            break
        line = proc.stdout.readline()
        if line == "":
            break


def start_engine(cmd: str, extra_env: Optional[dict] = None) -> subprocess.Popen:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(
        shlex.split(cmd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )


def handshake(proc: subprocess.Popen, args) -> bool:
    if not send(proc, "xboard"):
        return False
    send(proc, "protover 2")
    send(proc, "new")
    send(proc, "force")
    if args.depth:
        send(proc, f"sd {args.depth}")
    send(proc, f"stms {args.move_time_ms}")
    send(proc, f"time {args.clock_cs}")
    send(proc, f"otim {args.clock_cs}")
    drain(proc)
    return True


def request_fen(proc: subprocess.Popen) -> Optional[str]:
    line = query_prefix(proc, "david_fen", "fen ")
    if line is None:
        return None
    return line[len("fen ") :].strip()


def request_lastscore(proc: subprocess.Popen) -> Optional[int]:
    line = query_prefix(proc, "david_lastscore", "lastscore ")
    if line is None:
        return None
    payload = line[len("lastscore ") :].strip()
    if payload == "none":
        return None
    try:
        return int(payload)
    except ValueError:
        return None


def list_moves(proc: subprocess.Popen) -> List[str]:
    line = query_prefix(proc, "david_moves", "moves")
    if line is None:
        return []
    parts = line.split()
    return parts[1:]


def randomize_opening(proc: subprocess.Popen, plies: int, rng: random.Random) -> None:
    for _ in range(plies):
        moves = list_moves(proc)
        if not moves:
            return
        mv = rng.choice(moves)
        if not send(proc, f"usermove {mv}"):
            return


def play_game(proc: subprocess.Popen, game_id: int, args, rng: random.Random, budget: int) -> List[List]:
    rows: List[List] = []
    randomize_opening(proc, args.random_opening, rng)
    ply = 0
    while ply < args.max_plies and len(rows) < budget:
        fen = request_fen(proc)
        if not fen:
            break
        fields = fen.split()
        if len(fields) < 2:
            break
        side = fields[1]
        phase = material_phase(fen)
        sparse_tail = rng.random() < args.tail_fraction
        in_band = args.phase_min <= phase <= args.phase_max
        should_sample = (
            ply >= args.sample_start
            and ply % args.sample_stride == 0
            and (in_band or sparse_tail)
            and nonking_material(fen) > 2
        )

        # Set a fresh per-move clock to avoid draining over many plies.
        send(proc, f"time {args.clock_cs}")
        send(proc, f"otim {args.clock_cs}")

        go_cmd = "white" if side == "w" else "black"
        send(proc, go_cmd)
        move_line = read_until(proc, args.move_timeout, lambda l: l.startswith("move "))
        if move_line is None:
            break
        parts = move_line.split()
        move = parts[1] if len(parts) >= 2 else ""
        score = request_lastscore(proc)
        if should_sample and score is not None:
            rows.append([fen, score, phase, ply, side, game_id, move])
        ply += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Forced-capture self-play sampler (Texel dataset).")
    parser.add_argument("--engine-cmd", default="./program", help="Engine binary to run (CECP).")
    parser.add_argument("--positions", type=int, default=30000, help="Target number of positions to collect.")
    parser.add_argument("--max-games", type=int, default=5000, help="Fail-safe cap on number of games.")
    parser.add_argument("--max-plies", type=int, default=180, help="Maximum plies per game.")
    parser.add_argument("--sample-stride", type=int, default=2, help="Record every N plies.")
    parser.add_argument("--sample-start", type=int, default=8, help="Start sampling after this ply.")
    parser.add_argument("--phase-min", type=int, default=6, help="Preferred minimum material phase.")
    parser.add_argument("--phase-max", type=int, default=18, help="Preferred maximum material phase.")
    parser.add_argument("--tail-fraction", type=float, default=0.2, help="Fraction of out-of-band phases to keep.")
    parser.add_argument("--random-opening", type=int, default=4, help="Random plies to play before sampling.")
    parser.add_argument("--move-time-ms", type=int, default=150, help="Fixed time per move in ms.")
    parser.add_argument("--clock-cs", type=int, default=1500, help="Clock centiseconds sent before each move.")
    parser.add_argument("--depth", type=int, default=6, help="Fixed search depth for labeling (0 = engine default).")
    parser.add_argument("--output", default="build/selfplay_positions.csv", help="Output CSV path.")
    parser.add_argument("--gzip", action="store_true", help="Compress output with gzip (.gz).")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    parser.add_argument("--move-timeout", type=float, default=5.0, help="Seconds to wait for a move.")
    args = parser.parse_args()

    rng = random.Random(args.seed or int(time.time()))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    opener = gzip.open if args.gzip else open
    mode = "wt"
    with opener(args.output if not args.gzip else args.output + ".gz", mode, newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fen", "eval_cp", "phase", "ply", "side_to_move", "game_id", "move"])

        collected = 0
        game_id = 0
        while collected < args.positions and game_id < args.max_games:
            game_id += 1
            proc = start_engine(args.engine_cmd)
            try:
                if not handshake(proc, args):
                    break
                budget = args.positions - collected
                rows = play_game(proc, game_id, args, rng, budget)
                for row in rows:
                    writer.writerow(row)
                collected += len(rows)
                sys.stdout.write(f"Game {game_id}: +{len(rows)} positions (total {collected})\n")
                sys.stdout.flush()
            finally:
                send(proc, "quit")
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()

    sys.stdout.write(f"Done. Collected {collected} positions into {args.output}{'.gz' if args.gzip else ''}\n")


if __name__ == "__main__":
    main()
