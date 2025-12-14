#!/usr/bin/env python3
"""Deep labeling pipeline: run the engine at depth+2 (or 3x time) to relabel positions.

Input: CSV (optionally .gz) with columns including at least: fen, side_to_move, ply, game_id, eval_cp, phase, move.
Output: CSV (.gz optional) with the same columns plus eval_deep_cp and eval_deep_norm (tanh-scaled).
"""

import argparse
import csv
import gzip
import math
import os
import select
import shlex
import subprocess
import time
from typing import Dict, Iterable, Optional, Tuple


def send(proc: subprocess.Popen, cmd: str) -> bool:
    try:
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()
        return True
    except BrokenPipeError:
        return False


def read_until(proc: subprocess.Popen, timeout: float, predicate) -> Optional[str]:
    end = time.time() + timeout
    while time.time() < end:
        remaining = max(0.0, end - time.time())
        if proc.poll() is not None:
            return None
        try:
            ready, _, _ = select.select([proc.stdout], [], [], remaining)
        except (ValueError, OSError):
            return None
        if not ready:
            continue
        while True:
            try:
                line = proc.stdout.readline()
            except Exception:
                return None
            if line == "":
                if proc.poll() is not None:
                    return None
                break
            if isinstance(line, bytes):
                try:
                    line = line.decode(errors="ignore")
                except Exception:
                    continue
            line = line.strip()
            if not line:
                continue
            if predicate(line):
                return line
    return None


def query_prefix(proc: subprocess.Popen, cmd: str, prefix: str, timeout: float = 2.0) -> Optional[str]:
    if not send(proc, cmd):
        return None
    return read_until(proc, timeout, lambda line: line.startswith(prefix))


def start_engine(cmd: str, extra_env: Optional[dict] = None) -> subprocess.Popen:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(
        shlex.split(cmd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    try:
        import fcntl

        flags = fcntl.fcntl(proc.stdout.fileno(), fcntl.F_GETFL)
        fcntl.fcntl(proc.stdout.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)
    except Exception:
        pass
    return proc


def handshake(proc: subprocess.Popen, args) -> bool:
    if not send(proc, "xboard"):
        return False
    send(proc, "protover 2")
    feat_ok = read_until(proc, timeout=2.0, predicate=lambda l: l.startswith("feature ") or l == "done=1")
    send(proc, "new")
    send(proc, "force")
    if args.deep_depth > 0:
        send(proc, f"sd {args.deep_depth}")
    if args.deep_time_ms > 0:
        send(proc, f"stms {args.deep_time_ms}")
    send(proc, f"time {args.clock_cs}")
    send(proc, f"otim {args.clock_cs}")
    # Verify custom command responds
    probe = query_prefix(proc, "david_fen", "fen ", timeout=1.0)
    return (feat_ok is not None) and (probe is not None)


def request_lastscore(proc: subprocess.Popen) -> Optional[int]:
    line = query_prefix(proc, "david_lastscore", "lastscore ", timeout=1.0)
    if line is None:
        return None
    payload = line[len("lastscore ") :].strip()
    if not payload:
        return None
    lower = payload.lower()
    if lower == "none":
        return None
    parts = lower.split()
    val: Optional[int] = None
    if parts[0] == "mate" and len(parts) >= 2:
        sign = -1 if parts[1].startswith("-") else 1
        val = sign * 30000
    else:
        try:
            cleaned = parts[0].lstrip("+")
            val = int(cleaned)
        except ValueError:
            return None
    if val is None or val < -30000 or val > 30000:
        return None
    return val


def evaluate_position(proc: subprocess.Popen, fen: str, side: str, args, move_timeout: float) -> Optional[int]:
    if not send(proc, f"setboard {fen}"):
        return None

    stm = side.strip().lower()
    go_cmd = "white" if stm.startswith("w") else "black"
    if not send(proc, go_cmd):
        return None
    move_line = read_until(proc, move_timeout, lambda l: l.startswith("move "))
    if move_line is None:
        return None
    score = request_lastscore(proc)
    return score


def normalize(cp: int, scale: float) -> float:
    return math.tanh(cp / scale)


def open_reader(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", newline="", encoding="utf-8", errors="replace")
    return open(path, "r", newline="", encoding="utf-8", errors="replace")


def open_writer(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "wt", newline="", encoding="utf-8")
    return open(path, "w", newline="", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep label positions with depth+2 or 3x time.")
    parser.add_argument("--input", default="build/selfplay_positions.csv.gz", help="Input CSV (.gz ok) with sampled positions.")
    parser.add_argument("--output", default="build/deep_labeled_positions.csv.gz", help="Output CSV (.gz ok).")
    parser.add_argument("--engine-cmd", default="./program", help="Engine command.")
    parser.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 4, 8)), help="Number of parallel engines.")
    parser.add_argument("--shallow-depth", type=int, default=6, help="Reference shallow depth used to sample.")
    parser.add_argument("--deep-depth", type=int, default=0, help="Override deep depth; default is shallow+2.")
    parser.add_argument("--move-time-ms", type=int, default=150, help="Shallow per-move time (for time multiplier).")
    parser.add_argument("--time-mult", type=float, default=3.0, help="Time multiplier for deep eval if deep-time-ms not set.")
    parser.add_argument("--deep-time-ms", type=int, default=0, help="Override deep per-move time in ms (0 to disable).")
    parser.add_argument("--tanh-scale", type=float, default=400.0, help="Scale for tanh normalization.")
    parser.add_argument("--move-timeout", type=float, default=8.0, help="Seconds to wait for a deep move.")
    parser.add_argument("--fail-log", default="build/deep_label_failures.log", help="Path to write failures.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.deep_depth <= 0:
        args.deep_depth = args.shallow_depth + 2
    if args.deep_time_ms <= 0:
        args.deep_time_ms = int(args.move_time_ms * args.time_mult)
    args.clock_cs = max(1, args.deep_time_ms // 10)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.fail_log)), exist_ok=True)

    fail_writer = open(args.fail_log, "w")
    proc = start_engine(args.engine_cmd)
    if not handshake(proc, args):
        fail_writer.write("handshake failed\n")
        fail_writer.flush()
        proc.kill()
        fail_writer.close()
        return

    total = 0
    written = 0
    with open_reader(args.input) as fh, open_writer(args.output) as out_fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            fail_writer.write("fatal: input has no header\n")
            fail_writer.flush()
            proc.kill()
            return
        base_fields = [f for f in reader.fieldnames if f not in ("eval_deep_cp", "eval_deep_norm")]
        fieldnames = base_fields + ["eval_deep_cp", "eval_deep_norm"]
        writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(reader):
            total += 1
            fen = (row.get("fen") or "").strip()
            if not fen:
                fail_writer.write(f"{idx}: missing fen\n")
                if idx % 100 == 0:
                    fail_writer.flush()
                continue
            parts = fen.split()
            if len(parts) >= 2:
                default_side = parts[1]
            else:
                default_side = "w"
            side = row.get("side_to_move") or default_side
            score = evaluate_position(proc, fen, side, args, args.move_timeout)
            if score is None:
                fail_writer.write(f"{idx}: eval_fail\n")
                if idx % 100 == 0:
                    fail_writer.flush()
                continue
            row_out = {**row}
            row_out["eval_deep_cp"] = str(score)
            row_out["eval_deep_norm"] = f"{normalize(score, args.tanh_scale):.6f}"
            writer.writerow(row_out)
            written += 1

    send(proc, "quit")
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        proc.kill()
    fail_writer.close()

    print(f"Deep labeling done. Input rows: {total}, labeled: {written}, output: {args.output}")


if __name__ == "__main__":
    main()
