#!/usr/bin/env python3
"""Smoke-test the engine while stderr is piped but not drained.

Some harnesses spawn engines with stderr=PIPE but never read it. If the engine
logs verbosely to stderr by default, it can block after a few moves when the
pipe buffer fills. This test ensures the default build stays quiet on stderr and
continues to respond over CECP/XBoard.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path
from select import select
from typing import Optional


def send(proc: subprocess.Popen[str], cmd: str) -> None:
    if proc.stdin is None:
        raise RuntimeError("engine stdin closed")
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()


def read_line(proc: subprocess.Popen[str], timeout_s: float) -> Optional[str]:
    stdout = proc.stdout
    if stdout is None:
        return None
    end = time.time() + timeout_s
    while time.time() < end:
        if proc.poll() is not None:
            return None
        remaining = max(0.0, end - time.time())
        ready, _, _ = select([stdout], [], [], remaining)
        if not ready:
            continue
        line = stdout.readline()
        if not line:
            continue
        line = line.strip()
        if line:
            return line
    return ""


def query_prefix(proc: subprocess.Popen[str], cmd: str, prefix: str, timeout_s: float) -> Optional[str]:
    send(proc, cmd)
    return read_until_prefix(proc, prefix, timeout_s)


def read_until_prefix(proc: subprocess.Popen[str], prefix: str, timeout_s: float) -> Optional[str]:
    end = time.time() + timeout_s
    while time.time() < end:
        line = read_line(proc, max(0.0, end - time.time()))
        if line is None:
            return None
        if not line:
            continue
        if line.startswith(prefix):
            return line
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Ensure the engine doesn't block when stderr is not drained.")
    ap.add_argument("--engine", default=None, help="Path to engine binary (default: ./program).")
    ap.add_argument("--plies", type=int, default=400, help="Number of plies to play.")
    ap.add_argument("--depth", type=int, default=2, help="Fixed depth sent via 'sd'.")
    ap.add_argument("--move-time-ms", type=int, default=5, help="Move time sent via 'stms'.")
    ap.add_argument("--timeout-s", type=float, default=8.0, help="Overall timeout for the run.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    engine_path = Path(args.engine) if args.engine else (repo_root / "program")
    if not engine_path.exists():
        print(f"engine not found: {engine_path}", file=sys.stderr)
        return 2

    proc = subprocess.Popen(
        shlex.split(str(engine_path)),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # intentionally not drained during the run
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    try:
        try:
            # Basic CECP/XBoard handshake.
            for cmd in (
                "xboard",
                "protover 2",
                "new",
                f"sd {max(0, int(args.depth))}",
                f"stms {max(0, int(args.move_time_ms))}",
                f"time {max(0, int(args.move_time_ms)) * 10}",
                f"otim {max(0, int(args.move_time_ms)) * 10}",
            ):
                send(proc, cmd)

            # Drain any initial stdout chatter (feature lines, etc.).
            start = time.time()
            while time.time() - start < 0.2:
                _ = read_line(proc, 0.01)

            plies_done = 0
            deadline = time.time() + max(1.0, float(args.timeout_s))
            turns = max(0, int(args.plies)) // 2

            for _ in range(turns):
                if time.time() > deadline:
                    raise TimeoutError("overall timeout exceeded")
                if proc.poll() is not None:
                    raise RuntimeError(f"engine exited early with code {proc.returncode}")

                moves_line = query_prefix(proc, "david_moves", "moves", timeout_s=1.0)
                if moves_line is None:
                    raise RuntimeError("engine did not respond to david_moves")
                moves = moves_line.split()[1:]
                if not moves:
                    break

                send(proc, "usermove " + moves[0])
                plies_done += 1

                reply = read_until_prefix(proc, "move ", timeout_s=1.0)
                if reply is None:
                    raise RuntimeError("engine did not reply with a move")
                plies_done += 1

            send(proc, "quit")
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=1.0)

            stderr_text = ""
            if proc.stderr is not None:
                stderr_text = proc.stderr.read() or ""
            if stderr_text.strip():
                print("unexpected stderr output in default build:", file=sys.stderr)
                print(stderr_text[:4000], file=sys.stderr)
                return 1

            print(f"ok: {plies_done} plies")
            return 0
        except Exception as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
    finally:
        if proc.poll() is None:
            proc.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
