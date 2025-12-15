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
import threading
import time
from collections import deque
from queue import Queue
from typing import Dict, Iterable, Optional, Tuple


class FailLogger:
    def __init__(self, path: str):
        self._lock = threading.Lock()
        self._fh = open(path, "w", encoding="utf-8")

    def write(self, msg: str) -> None:
        with self._lock:
            self._fh.write(msg)
            if not msg.endswith("\n"):
                self._fh.write("\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def send(proc: subprocess.Popen, cmd: str) -> bool:
    try:
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()
        return True
    except BrokenPipeError:
        return False


def read_until(proc: subprocess.Popen, timeout: float, predicate, line_handler=None) -> Optional[str]:
    end = time.time() + timeout
    bytes_read = 0
    lines_read = 0
    max_bytes = 1_000_000
    max_lines = 10_000
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
        for _ in range(100):
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
            bytes_read += len(line)
            lines_read += 1
            if bytes_read > max_bytes or lines_read > max_lines:
                return None
            line = line.strip()
            if not line:
                continue
            if line_handler:
                try:
                    line_handler(line)
                except Exception:
                    pass
            if predicate(line):
                return line
    return None


def query_prefix(proc: subprocess.Popen, cmd: str, prefix: str, timeout: float = 2.0) -> Optional[str]:
    if not send(proc, cmd):
        return None
    return read_until(proc, timeout, lambda line: line.startswith(prefix))


def parse_score_token(token: str) -> Optional[int]:
    """Parse a score token (cp or mate notation) into an int, clamped to [-30000, 30000]."""
    t = token.strip()
    if not t:
        return None
    tl = t.lower()
    if tl == "none":
        return None
    val: Optional[int] = None

    parts = tl.split()
    if parts and parts[0] == "mate":
        try:
            ply = int(parts[1])
        except (IndexError, ValueError):
            return None
        sign = -1 if ply < 0 else 1
        dist = abs(ply)
        val = sign * max(1, 30000 - min(dist, 29999))
    elif tl.startswith("+m") or tl.startswith("-m"):
        sign = -1 if tl.startswith("-m") else 1
        try:
            dist = int(tl[2:])
        except ValueError:
            return None
        val = sign * max(1, 30000 - min(abs(dist), 29999))
    else:
        try:
            val = int(t.lstrip("+"))
        except ValueError:
            return None

    if val is None:
        return None
    return max(-30000, min(30000, val))


def start_engine(cmd: str, extra_env: Optional[dict] = None) -> subprocess.Popen:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(
        shlex.split(cmd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    return proc


def handshake(proc: subprocess.Popen, args, fail_logger: Optional[FailLogger] = None) -> bool:
    if not send(proc, "xboard"):
        return False
    if not send(proc, "protover 2"):
        return False
    done = False
    deadline = time.time() + 2.0
    while time.time() < deadline:
        line = read_until(
            proc,
            timeout=max(0.0, deadline - time.time()),
            predicate=lambda l: l.startswith("feature") or l.startswith("done"),
        )
        if line is None:
            break
        if "done=1" in line or line.strip() == "done=1":
            done = True
            break
    if not done:
        if fail_logger:
            fail_logger.write("handshake failed: feature negotiation timeout")
        return False
    if not send(proc, "new"):
        return False
    if not send(proc, "force"):
        return False
    if args.deep_depth > 0:
        send(proc, f"sd {args.deep_depth}")
    if args.deep_time_ms > 0:
        send(proc, f"stms {args.deep_time_ms}")
    send(proc, f"time {args.clock_cs}")
    send(proc, f"otim {args.clock_cs}")
    probe = query_prefix(proc, "david_fen", "fen ", timeout=1.5)
    return probe is not None


def restart_engine(args, fail_logger: FailLogger):
    proc = start_engine(args.engine_cmd)
    if not handshake(proc, args, fail_logger):
        fail_logger.write("handshake failed")
        proc.kill()
        return None
    return proc


def request_lastscore(proc: subprocess.Popen) -> Optional[int]:
    line = query_prefix(proc, "david_lastscore", "lastscore ", timeout=1.0)
    if line is None:
        return None
    payload = line[len("lastscore ") :].strip()
    return parse_score_token(payload)


def drain_stderr(proc: subprocess.Popen, fail_logger: Optional[FailLogger] = None) -> None:
    if not proc.stderr:
        return
    try:
        while True:
            ready, _, _ = select.select([proc.stderr], [], [], 0)
            if not ready:
                break
            line = proc.stderr.readline()
            if not line:
                break
            if fail_logger:
                fail_logger.write(f"stderr: {line.strip()}")
    except Exception:
        pass


def drain_stdout(proc: subprocess.Popen):
    if not proc.stdout:
        return
    try:
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], 0)
            if not ready:
                break
            line = proc.stdout.readline()
            if not line:
                break
    except Exception:
        pass


def dump_engine_output(proc: subprocess.Popen, fail_logger: FailLogger, header: str) -> None:
    """Reads any buffered stdout/stderr from the engine without blocking (for diagnostics)."""
    fail_logger.write(f"--- {header} ---")
    def drain_stream(stream, label: str):
        if not stream:
            fail_logger.write(f"   [{label}: closed]")
            return
        try:
            while True:
                ready, _, _ = select.select([stream], [], [], 0)
                if not ready:
                    break
                line = stream.readline()
                if not line:
                    break
                fail_logger.write(f"   [{label}] {line.strip()}")
        except Exception as e:
            fail_logger.write(f"   [{label} read error: {e}]")

    drain_stream(proc.stdout, "stdout")
    drain_stream(proc.stderr, "stderr")
    fail_logger.write("--------------------------------")


def evaluate_position(proc: subprocess.Popen, fen: str, side: str, args, move_timeout: float, fail_logger: FailLogger) -> Optional[int]:
    drain_stdout(proc)
    if not send(proc, f"setboard {fen}"):
        fail_logger.write(f"FAIL: 'setboard' rejected for FEN: {fen}")
        return None

    if args.deep_depth > 0:
        send(proc, f"sd {args.deep_depth}")
    if args.deep_time_ms > 0:
        send(proc, f"stms {args.deep_time_ms}")
    if args.clock_cs > 0:
        send(proc, f"time {args.clock_cs}")
        send(proc, f"otim {args.clock_cs}")

    # Track fallback score from thinking output and a short log for debugging.
    fallback_score = [None]
    recent_logs = deque(maxlen=20)

    def handle_output(line: str) -> None:
        recent_logs.append(line)
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            s = parse_score_token(parts[1])
            if s is not None:
                fallback_score[0] = s

    stm = side.strip().lower()
    go_cmd = "white" if stm.startswith("w") else "black"
    if not send(proc, go_cmd):
        fail_logger.write("FAIL: 'go' command rejected.")
        return None

    start_time = time.time()
    move_line = read_until(
        proc,
        move_timeout,
        lambda l: l.startswith("move ") or l in ("1-0", "0-1", "1/2-1/2"),
        line_handler=handle_output,
    )
    elapsed = time.time() - start_time

    if move_line is None:
        exit_code = proc.poll()
        if exit_code is not None:
            fail_logger.write(f"CRASH: engine exited (code {exit_code}) after {elapsed:.2f}s on FEN: {fen}")
        elif elapsed >= move_timeout:
            fail_logger.write(f"TIMEOUT: >{elapsed:.2f}s (limit {move_timeout}s) on FEN: {fen}")
            dump_engine_output(proc, fail_logger, "Buffered Output at Timeout")
        else:
            fail_logger.write(f"ERROR: engine silent/EOF after {elapsed:.2f}s on FEN: {fen}")
            dump_engine_output(proc, fail_logger, "Buffered Output on EOF")
        # Ensure the next task doesn't reuse a wedged engine process.
        try:
            send(proc, "?")
            send(proc, "force")
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
        return None

    score = request_lastscore(proc)
    if score is None:
        score = fallback_score[0]
    send(proc, "?")
    drain_stdout(proc)
    send(proc, "force")
    if score is None:
        fail_logger.write(f"MISSING SCORE after move '{move_line}': fen={fen}")
        for l in recent_logs:
            fail_logger.write(f"  > {l}")
        dump_engine_output(proc, fail_logger, "Debug Output")
    return score


def normalize(cp: int, scale: float) -> float:
    return math.tanh(cp / scale)


def open_reader(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", newline="", encoding="utf-8", errors="replace")
    return open(path, "r", newline="", encoding="utf-8", errors="replace")


def open_writer(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8")
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
    parser.add_argument("--log-engine-stderr", action="store_true", help="Write engine stderr into the failure log (very verbose).")
    parser.add_argument("--progress-every", type=int, default=500, help="Print progress every N positions.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N rows (0 = no limit).")
    return parser.parse_args()


def worker_loop(worker_id: int, args: argparse.Namespace, rows: list, task_q: Queue, result_q: Queue, fail_logger: FailLogger) -> None:
    proc = restart_engine(args, fail_logger)
    if proc is None:
        fail_logger.write(f"worker {worker_id}: failed to start engine; marking tasks as failed")
        while True:
            idx = task_q.get()
            if idx is None:
                break
            result_q.put((idx, None))
        return

    try:
        while True:
            idx = task_q.get()
            if idx is None:
                break

            if proc.poll() is not None:
                proc = restart_engine(args, fail_logger)
                if proc is None:
                    fail_logger.write(f"{idx}: engine_restart_failed")
                    result_q.put((idx, None))
                    continue

            drain_stderr(proc, fail_logger if args.log_engine_stderr else None)

            row = rows[idx]
            fen = (row.get("fen") or "").strip()
            if not fen:
                fail_logger.write(f"{idx}: missing fen")
                result_q.put((idx, None))
                continue

            parts = fen.split()
            default_side = parts[1] if len(parts) >= 2 else "w"
            side = row.get("side_to_move") or default_side

            score = evaluate_position(proc, fen, side, args, args.move_timeout, fail_logger)
            if score is None:
                result_q.put((idx, None))
                continue

            row_out = {**row}
            row_out["eval_deep_cp"] = str(score)
            row_out["eval_deep_norm"] = f"{normalize(score, args.tanh_scale):.6f}"
            result_q.put((idx, row_out))
    finally:
        try:
            if proc.poll() is None:
                send(proc, "quit")
                proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def main():
    args = parse_args()
    if args.deep_depth <= 0:
        args.deep_depth = args.shallow_depth + 2
    if args.deep_time_ms <= 0:
        args.deep_time_ms = int(args.move_time_ms * args.time_mult)
    args.clock_cs = max(200, int(round(args.deep_time_ms * 2)))  # give a generous clock per position

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.fail_log)), exist_ok=True)

    fail_logger = FailLogger(args.fail_log)

    with open_reader(args.input) as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            fail_logger.write("fatal: input has no header")
            fail_logger.close()
            return

        base_fields = [f for f in reader.fieldnames if f not in ("eval_deep_cp", "eval_deep_norm")]
        fieldnames = base_fields + ["eval_deep_cp", "eval_deep_norm"]

        rows = list(reader)
        if args.limit and args.limit > 0:
            rows = rows[: args.limit]

    total = len(rows)
    if total == 0:
        fail_logger.write("fatal: input has no rows")
        fail_logger.close()
        return

    args.workers = max(1, int(args.workers))
    if args.workers > total:
        args.workers = total

    task_q: Queue = Queue()
    result_q: Queue = Queue()

    for idx in range(total):
        task_q.put(idx)
    for _ in range(args.workers):
        task_q.put(None)

    threads = []
    for wid in range(args.workers):
        t = threading.Thread(target=worker_loop, args=(wid, args, rows, task_q, result_q, fail_logger), daemon=True)
        t.start()
        threads.append(t)

    written = 0
    processed = 0
    pending = {}
    next_to_write = 0
    start = time.time()

    with open_writer(args.output) as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
        writer.writeheader()

        while processed < total:
            idx, row_out = result_q.get()
            processed += 1
            pending[idx] = row_out

            while next_to_write in pending:
                out_row = pending.pop(next_to_write)
                if out_row is not None:
                    writer.writerow(out_row)
                    written += 1
                next_to_write += 1

            if args.progress_every > 0 and processed % args.progress_every == 0:
                elapsed = max(1e-6, time.time() - start)
                rate = processed / elapsed
                remaining = total - processed
                eta_s = remaining / max(1e-6, rate)
                print(f"{processed}/{total} processed, {written} labeled, {rate:.1f} pos/s, ETA {eta_s/60.0:.1f} min")
                try:
                    out_fh.flush()
                except Exception:
                    pass

    for t in threads:
        t.join()

    fail_logger.close()
    print(f"Deep labeling done. Input rows: {total}, labeled: {written}, output: {args.output}")


if __name__ == "__main__":
    main()
