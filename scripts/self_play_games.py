#!/usr/bin/env python3
"""Generate NDJSON self-play games for the forced-capture variant.

Each line: {"moves": [...], "result": "1-0/0-1/1/2-1/2", "start_fen": "...",
            "ply_data": [{"fen": "...", "zobrist_key": "0x...", "move": "..."}]}
"""

import argparse
import json
import os
import select
import shlex
import subprocess
import sys
import time
from typing import List, Optional, Tuple

# --- Zobrist (reuse engine scheme) ---
import hashlib

DEFAULT_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
FEN_TO_PIECE = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6, "p": 9, "n": 10, "b": 11, "r": 12, "q": 13, "k": 14}
WKCA = 1
WQCA = 2
BKCA = 4
BQCA = 8

def zobrist_rand(state):
    x, y = state
    state[0] = y
    x ^= (x << 23) & 0xFFFFFFFFFFFFFFFF
    state[1] = (x ^ y ^ (x >> 17) ^ (y >> 26)) & 0xFFFFFFFFFFFFFFFF
    return (state[1] + y) & 0xFFFFFFFFFFFFFFFF

def build_engine_zobrist(seed=0x9E3779B97F4A7C15):
    st = [seed & 0xFFFFFFFFFFFFFFFF, (seed ^ 0xA0761D6478BD642F) & 0xFFFFFFFFFFFFFFFF]
    piece_sq = [[0]*128 for _ in range(16)]
    for p in range(16):
        for sq in range(128):
            piece_sq[p][sq] = zobrist_rand(st)
    side = zobrist_rand(st)
    castling = [zobrist_rand(st) for _ in range(16)]
    ep = [zobrist_rand(st) for _ in range(8)]
    return piece_sq, side, castling, ep

Z_PIECE_SQ, Z_SIDE, Z_CASTLING, Z_EP_FILE = build_engine_zobrist()

def parse_fen_basic(fen: str):
    parts = fen.strip().split()
    if len(parts) < 4:
        return None
    placement, stm, castling, ep = parts[:4]
    board = [0]*128
    rank, file = 7, 0
    for ch in placement:
        if ch == "/":
            if file != 8:
                return None
            rank -= 1
            file = 0
            continue
        if ch.isdigit():
            file += int(ch)
            if file > 8:
                return None
            continue
        if ch not in FEN_TO_PIECE or file >= 8 or rank < 0:
            return None
        sq = (rank << 4) | file
        board[sq] = FEN_TO_PIECE[ch]
        file += 1
    if rank != 0 or file != 8:
        return None
    side = stm.lower()
    if side not in ("w", "b"):
        return None
    castling_mask = 0
    for c in castling if castling != "-" else "":
        if c == "K": castling_mask |= WKCA
        elif c == "Q": castling_mask |= WQCA
        elif c == "k": castling_mask |= BKCA
        elif c == "q": castling_mask |= BQCA
        else: return None
    ep_sq = None
    if ep != "-":
        if len(ep) != 2 or ep[0] < "a" or ep[0] > "h" or ep[1] < "1" or ep[1] > "8":
            return None
        ep_sq = ((int(ep[1])-1) << 4) | (ord(ep[0])-ord("a"))
    return board, side, castling_mask, ep_sq

def maybe_include_ep(side: str, ep_sq: int, board: List[int]) -> Optional[int]:
    ep_file = ep_sq & 7
    ep_rank = ep_sq >> 4
    if side == "w" and ep_rank == 5:
        for df in (-1,1):
            f = ep_file + df
            if 0 <= f < 8 and board[(4<<4)|f] == FEN_TO_PIECE["P"]:
                return ep_file
    if side == "b" and ep_rank == 2:
        for df in (-1,1):
            f = ep_file + df
            if 0 <= f < 8 and board[(3<<4)|f] == FEN_TO_PIECE["p"]:
                return ep_file
    return None

def zobrist_from_fen(fen: str) -> Optional[int]:
    parsed = parse_fen_basic(fen)
    if parsed is None:
        return None
    board, side, castling_mask, ep_sq = parsed
    key = 0
    for sq, piece in enumerate(board):
        if piece:
            key ^= Z_PIECE_SQ[piece][sq]
    if side == "b":
        key ^= Z_SIDE
    key ^= Z_CASTLING[castling_mask & 0xF]
    if ep_sq is not None:
        ep_file = maybe_include_ep(side, ep_sq, board)
        if ep_file is not None:
            key ^= Z_EP_FILE[ep_file]
    return key

# --- Engine helpers ---

def send(proc: subprocess.Popen, cmd: str) -> bool:
    try:
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()
        return True
    except BrokenPipeError:
        return False

def read_until(proc: subprocess.Popen, timeout: float, predicate) -> Optional[str]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        rlist, _, _ = select.select([proc.stdout], [], [], max(0, deadline - time.time()))
        if not rlist:
            continue
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        if predicate(line):
            return line
    return None

def query_prefix(proc: subprocess.Popen, cmd: str, prefix: str, timeout: float = 1.0) -> Optional[str]:
    if not send(proc, cmd):
        return None
    return read_until(proc, timeout, lambda l: l.startswith(prefix))

def list_moves(proc: subprocess.Popen) -> List[str]:
    line = query_prefix(proc, "david_moves", "moves", timeout=1.0)
    if line is None:
        return []
    return line.split()[1:]

def start_engine(cmd: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        shlex.split(cmd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # capture for debugging and to avoid pipe fill
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,  # line-buffered to cooperate with readline/select
    )
    return proc

def _drain_stderr_nonblock(proc: subprocess.Popen) -> None:
    if not proc.stderr:
        return
    try:
        r, _, _ = select.select([proc.stderr], [], [], 0)
        while r:
            line = proc.stderr.readline()
            if not line:
                break
            sys.stderr.write(line)
            sys.stderr.flush()
            r, _, _ = select.select([proc.stderr], [], [], 0)
    except Exception:
        pass

def handshake(proc: subprocess.Popen, move_time_ms: int, depth: int, timeout: float = 1.0) -> bool:
    if not send(proc, "xboard"):
        return False
    send(proc, "protover 2")
    send(proc, "new")
    send(proc, "force")
    if depth > 0:
        send(proc, f"sd {depth}")
    send(proc, f"stms {move_time_ms}")
    send(proc, f"time {move_time_ms*10}")
    send(proc, f"otim {move_time_ms*10}")
    _drain_stderr_nonblock(proc)
    probe = query_prefix(proc, "david_fen", "fen ", timeout=timeout)
    _drain_stderr_nonblock(proc)
    return probe is not None

def request_fen(proc: subprocess.Popen) -> Optional[str]:
    line = query_prefix(proc, "david_fen", "fen ", timeout=1.0)
    if line is None:
        return None
    fen = line[len("fen "):].strip()
    return fen if fen else None

def engine_best_move(proc: subprocess.Popen, side: str, move_timeout: float) -> Optional[str]:
    if proc.poll() is not None:
        return None
    go_cmd = "white" if side == "w" else "black"
    if not send(proc, go_cmd):
        return None
    line = read_until(proc, move_timeout, lambda l: l.startswith("move ") or l in ("1-0", "0-1", "1/2-1/2"))
    if line is None:
        return None
    if line in ("1-0", "0-1", "1/2-1/2"):
        return line  # terminal result surfaced by engine
    parts = line.split()
    if len(parts) < 2:
        return None
    mv = parts[1].strip()
    return mv if mv else None

# --- Main loop ---

def play_game(proc: subprocess.Popen, game_id: int, max_plies: int, move_timeout: float) -> Optional[dict]:
    moves: List[str] = []
    ply_data = []
    start_fen = request_fen(proc) or DEFAULT_START_FEN
    for ply in range(max_plies):
        fen = request_fen(proc)
        if fen is None:
            return None
        key = zobrist_from_fen(fen)
        if key is None:
            return None
        legal = list_moves(proc)
        if not legal:
            # terminal: capture final position
            ply_data.append({"fen": fen, "zobrist_key": f"{key:016x}", "move": ""})
            break
        side = fen.split()[1] if len(fen.split()) > 1 else "w"
        mv = engine_best_move(proc, side, move_timeout)
        if mv is None:
            return None
        if mv in ("1-0", "0-1", "1/2-1/2"):
            ply_data.append({"fen": fen, "zobrist_key": f"{key:016x}", "move": ""})
            result = mv
            return {"moves": moves, "result": result, "start_fen": start_fen, "ply_data": ply_data, "game_id": game_id}
        moves.append(mv)
        ply_data.append({"fen": fen, "zobrist_key": f"{key:016x}", "move": mv})
        send(proc, f"usermove {mv}")
    # determine result: prefer explicit result line if emitted
    res_line = read_until(proc, 0.5, lambda l: l in ("1-0", "0-1", "1/2-1/2"))
    if res_line in ("1-0", "0-1", "1/2-1/2"):
        result = res_line
    else:
        legal = list_moves(proc)
        stm = request_fen(proc)
        side = stm.split()[1] if stm else "w"
        in_check = False
        chk = query_prefix(proc, "david_check", "check ")
        if chk is not None:
            try:
                in_check = bool(int(chk.split()[1]))
            except Exception:
                in_check = False
        if legal == []:
            if in_check:
                result = "0-1" if side == "w" else "1-0"
            else:
                result = "1/2-1/2"
        else:
            result = "1/2-1/2"
    return {"moves": moves, "result": result, "start_fen": start_fen, "ply_data": ply_data, "game_id": game_id}

def main():
    ap = argparse.ArgumentParser(description="Forced-capture self-play game generator (NDJSON).")
    ap.add_argument("--engine-cmd", default="./program", help="Engine binary (CECP).")
    ap.add_argument("--games", type=int, default=100, help="Number of games to generate.")
    ap.add_argument("--max-plies", type=int, default=80, help="Maximum plies per game.")
    ap.add_argument("--move-time-ms", type=int, default=150, help="Per-move time sent to engine.")
    ap.add_argument("--depth", type=int, default=6, help="Optional fixed depth (0 to ignore).")
    ap.add_argument("--move-timeout", type=float, default=5.0, help="Seconds to wait for engine move.")
    ap.add_argument("--output", default="build/selfplay_games.jsonl", help="Output NDJSON path (.gz not supported).")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out_fh:
        for gid in range(1, args.games + 1):
            proc = start_engine(args.engine_cmd)
            try:
                if not handshake(proc, args.move_time_ms, args.depth):
                    print(f"game {gid}: handshake failed", file=sys.stderr)
                    continue
                send(proc, "new")
                send(proc, "force")
                game = play_game(proc, gid, args.max_plies, args.move_timeout)
                if game is None or not game["moves"]:
                    print(f"game {gid}: aborted", file=sys.stderr)
                    continue
                out_fh.write(json.dumps(game) + "\n")
                out_fh.flush()
                sys.stdout.write(f"game {gid}: {len(game['moves'])} plies written\n")
                sys.stdout.flush()
            finally:
                send(proc, "quit")
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()

    print(f"Done. Wrote games to {args.output}")

if __name__ == "__main__":
    main()
