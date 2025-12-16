#!/usr/bin/env python3
"""Mine forced-capture self-play for opening WDL stats and a small book.

Expected input: NDJSON (one JSON object per line) with at least:
  - "moves": list of UCI strings
  - "result": game result ("1-0", "0-1", "1/2-1/2", 1, 0, 0.5, etc.)
Optional:
  - "start_fen": custom starting FEN (defaults to normal startpos)
  - "fens": list of FENs before each ply (length >= len(moves)); if present,
            replaying via engine/python-chess is skipped.

Outputs:
  - Compact book map JSON: {hex_key: "best_move"} for positions with >= min_samples.
  - Detailed JSONL with per-move WDL stats for inspection.

Hashing: uses a deterministic Zobrist scheme (polyglot-like) derived from FEN
state (pieces, side, castling, EP). Collisions are guarded by FEN equality.

Replaying: defaults to the built engine (./program) in force mode for variant
correctness; python-chess replay is opt-in via --prefer-python.
"""

import argparse
import gzip
import json
import os
import select
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import chess  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    chess = None


DEFAULT_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Mirror the engine’s piece codes (see board.h).
FEN_TO_PIECE = {
    "P": 1,  # WPAWN
    "N": 2,
    "B": 3,
    "R": 4,
    "Q": 5,
    "K": 6,
    "p": 9,  # BPAWN
    "n": 10,
    "b": 11,
    "r": 12,
    "q": 13,
    "k": 14,
}

WKCA = 1
WQCA = 2
BKCA = 4
BQCA = 8


def zobrist_rand(state: List[int]) -> int:
    # Replicates zobrist_rand() from zobrist.cpp.
    x = state[0]
    y = state[1]
    state[0] = y
    x ^= (x << 23) & 0xFFFFFFFFFFFFFFFF
    state[1] = (x ^ y ^ (x >> 17) ^ (y >> 26)) & 0xFFFFFFFFFFFFFFFF
    return (state[1] + y) & 0xFFFFFFFFFFFFFFFF


def build_engine_zobrist(seed: int = 0x9E3779B97F4A7C15) -> Tuple[List[List[int]], int, List[int], List[int]]:
    state = [seed & 0xFFFFFFFFFFFFFFFF, (seed ^ 0xA0761D6478BD642F) & 0xFFFFFFFFFFFFFFFF]
    piece_sq = [[0] * 128 for _ in range(16)]
    for p in range(16):
        for sq in range(128):
            piece_sq[p][sq] = zobrist_rand(state)
    side = zobrist_rand(state)
    castling = [zobrist_rand(state) for _ in range(16)]
    ep = [zobrist_rand(state) for _ in range(8)]
    return piece_sq, side, castling, ep


Z_PIECE_SQ, Z_SIDE, Z_CASTLING, Z_EP_FILE = build_engine_zobrist()


def parse_fen_basic(fen: str):
    parts = fen.strip().split()
    if len(parts) < 4:
        return None
    placement, side_token, castling, ep = parts[:4]

    board_0x88: List[int] = [0] * 128
    rank = 7
    file = 0
    for ch in placement:
        if ch == "/":
            if file != 8:
                return None
            rank -= 1
            file = 0
            continue
        if ch.isdigit():
            span = int(ch)
            if span < 1 or span > 8:
                return None
            file += span
            if file > 8:
                return None
            continue
        if ch not in FEN_TO_PIECE or file >= 8 or rank < 0:
            return None
        sq0x88 = (rank << 4) | file
        board_0x88[sq0x88] = FEN_TO_PIECE[ch]
        file += 1
    if rank != 0 or file != 8:
        return None

    side = side_token.lower()
    if side not in ("w", "b"):
        return None

    castling_mask = 0
    if castling != "-":
        for c in castling:
            if c == "K":
                castling_mask |= WKCA
            elif c == "Q":
                castling_mask |= WQCA
            elif c == "k":
                castling_mask |= BKCA
            elif c == "q":
                castling_mask |= BQCA
            else:
                return None

    ep_square = None
    if ep != "-":
        if len(ep) != 2 or ep[0] < "a" or ep[0] > "h" or ep[1] < "1" or ep[1] > "8":
            return None
        file_idx = ord(ep[0]) - ord("a")
        rank_idx = int(ep[1]) - 1
        ep_square = (rank_idx << 4) | file_idx

    return {"board": board_0x88, "side": side, "castling_mask": castling_mask, "ep": ep_square}


def maybe_include_ep(side: str, ep_sq: int, board_0x88: List[int]) -> Optional[int]:
    ep_file = ep_sq & 7
    ep_rank = ep_sq >> 4
    if side == "w" and ep_rank == 5:
        if ep_file > 0:
            sqL = (4 << 4) | (ep_file - 1)
            if board_0x88[sqL] == FEN_TO_PIECE["P"]:
                return ep_file
        if ep_file < 7:
            sqR = (4 << 4) | (ep_file + 1)
            if board_0x88[sqR] == FEN_TO_PIECE["P"]:
                return ep_file
    elif side == "b" and ep_rank == 2:
        if ep_file > 0:
            sqL = (3 << 4) | (ep_file - 1)
            if board_0x88[sqL] == FEN_TO_PIECE["p"]:
                return ep_file
        if ep_file < 7:
            sqR = (3 << 4) | (ep_file + 1)
            if board_0x88[sqR] == FEN_TO_PIECE["p"]:
                return ep_file
    return None


def zobrist_from_fen(fen: str) -> Tuple[int, str]:
    parsed = parse_fen_basic(fen)
    if parsed is None:
        raise ValueError(f"invalid FEN: {fen}")
    board = parsed["board"]
    side = parsed["side"]
    castling_mask = parsed["castling_mask"]
    ep_sq = parsed["ep"]

    key = 0
    for sq in range(128):
        piece = board[sq]
        if piece:
            key ^= Z_PIECE_SQ[piece][sq]
    if side == "b":
        key ^= Z_SIDE
    key ^= Z_CASTLING[castling_mask & 0xF]
    if ep_sq is not None:
        ep_file = maybe_include_ep(side, ep_sq, board)
        if ep_file is not None:
            key ^= Z_EP_FILE[ep_file]
    return key, side


def parse_result_to_white_score(result) -> Optional[float]:
    if isinstance(result, (int, float)):
        if result >= 0.75:
            return 1.0
        if result <= 0.25:
            return 0.0
        return 0.5
    if not isinstance(result, str):
        return None
    token = result.strip().lower()
    if token in ("1-0", "w", "white", "white_win", "white-wins"):
        return 1.0
    if token in ("0-1", "b", "black", "black_win", "black-wins"):
        return 0.0
    if token in ("1/2-1/2", "1/2", "0.5", "draw", "d"):
        return 0.5
    return None


def open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


class ReplayError(Exception):
    pass


class EngineReplayer:
    def __init__(self, engine_cmd: str, timeout: float = 1.0):
        self.timeout = timeout
        self.proc = subprocess.Popen(
            shlex.split(engine_cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=0,
        )
        self._handshake()

    def _send(self, cmd: str) -> bool:
        try:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
            return True
        except BrokenPipeError:
            return False

    def _read_until(self, prefix: str) -> Optional[str]:
        deadline = time.time() + self.timeout
        while True:
            timeout_left = deadline - time.time()
            if timeout_left <= 0:
                break
            rlist, _, _ = select.select([self.proc.stdout], [], [], timeout_left)
            if not rlist:
                break
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith(prefix):
                return line
        return None

    def _handshake(self):
        if not self._send("xboard"):
            raise ReplayError("failed to start engine (xboard send)")
        self._send("protover 2")
        self._send("new")
        self._send("force")
        self._send("david_forced 1")
        probe = self._query_prefix("david_fen", "fen ")
        if probe is None:
            raise ReplayError("engine did not respond to david_fen")

    def _query_prefix(self, cmd: str, prefix: str) -> Optional[str]:
        if not self._send(cmd):
            return None
        return self._read_until(prefix)

    def reset(self, start_fen: str):
        self._send("new")
        self._send("force")
        self._send("david_forced 1")
        self._send(f"setboard {start_fen}")

    def get_fen(self) -> Optional[str]:
        line = self._query_prefix("david_fen", "fen ")
        if line is None:
            return None
        return line[len("fen ") :].strip()

    def list_moves(self) -> Optional[List[str]]:
        line = self._query_prefix("david_moves", "moves")
        if line is None:
            return None
        parts = line.split()
        return parts[1:]

    def replay(self, moves: List[str], start_fen: str, max_plies: int) -> List[str]:
        self.reset(start_fen)
        fens: List[str] = []
        for idx, mv in enumerate(moves):
            if idx >= max_plies:
                break
            fen = self.get_fen()
            if fen is None:
                raise ReplayError(f"missing FEN before ply {idx}")
            fens.append(fen)
            legal = self.list_moves()
            if legal is None:
                raise ReplayError(f"engine timeout listing moves at ply {idx}")
            if not legal:
                break  # terminal position reached
            if mv not in legal:
                raise ReplayError(f"illegal move {mv} at ply {idx}")
            if not self._send(f"usermove {mv}"):
                raise ReplayError("engine refused usermove")
        return fens

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=0.5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class PythonChessReplayer:
    def __init__(self):
        if chess is None:
            raise ReplayError("python-chess not available")

    def replay(self, moves: List[str], start_fen: str, max_plies: int) -> List[str]:
        board = chess.Board(start_fen)
        fens: List[str] = []
        for idx, mv in enumerate(moves):
            if idx >= max_plies:
                break
            legal = list(board.legal_moves)
            move_obj = chess.Move.from_uci(mv)
            if move_obj not in legal:
                raise ReplayError(f"illegal move {mv} at ply {idx}")
            has_capture = any(board.is_capture(m) for m in legal)
            if has_capture and not board.is_capture(move_obj):
                raise ReplayError(f"non-capture played with capture available at ply {idx}")
            fens.append(board.fen())
            board.push(move_obj)
        return fens


def select_replayer(args) -> object:
    # Default to engine replay for variant correctness; python-chess is opt-in.
    if args.prefer_python and chess is not None:
        return PythonChessReplayer()
    return EngineReplayer(args.engine_cmd, timeout=args.engine_timeout)


@dataclass
class MoveStats:
    wins: int = 0
    draws: int = 0
    losses: int = 0

    def add(self, outcome: float) -> None:
        if outcome >= 0.99:
            self.wins += 1
        elif outcome <= 0.01:
            self.losses += 1
        else:
            self.draws += 1

    @property
    def total(self) -> int:
        return self.wins + self.draws + self.losses


@dataclass
class PositionStats:
    key: int
    fen: Optional[str]
    moves: Dict[str, MoveStats] = field(default_factory=dict)

    def add(self, move: str, outcome: float) -> None:
        self.moves.setdefault(move, MoveStats()).add(outcome)

    @property
    def total(self) -> int:
        return sum(m.total for m in self.moves.values())


@dataclass
class PlyEntry:
    move: str
    key: int
    side: str
    fen: Optional[str]


def parse_side_from_fen(fen: str) -> Optional[str]:
    parts = fen.strip().split()
    if len(parts) >= 2:
        side = parts[1].lower()
        if side in ("w", "b"):
            return side
    return None


def parse_key_value(raw) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        try:
            return int(raw, 0)
        except ValueError:
            return None
    return None


def entries_from_ply_data(
    ply_data: List[dict], max_plies: int
) -> Optional[List[PlyEntry]]:
    entries: List[PlyEntry] = []
    for idx, ply in enumerate(ply_data):
        if idx >= max_plies:
            break
        if not isinstance(ply, dict):
            return None
        move = ply.get("move") or ply.get("uci")
        if not isinstance(move, str):
            return None
        fen = ply.get("fen")
        side = ply.get("side")
        if isinstance(side, str):
            side = side.lower()
            if side not in ("w", "b"):
                side = None
        if side is None and isinstance(fen, str):
            side = parse_side_from_fen(fen)
        if side not in ("w", "b"):
            return None
        key = parse_key_value(ply.get("zobrist_key") or ply.get("key"))
        if key is None:
            if not isinstance(fen, str):
                return None
            try:
                key, _ = zobrist_from_fen(fen)
            except ValueError:
                return None
        entries.append(PlyEntry(move=move, key=key, side=side, fen=fen))
    if not entries:
        return None
    return entries


def entries_from_fens_list(
    fens: List[str], moves: List[str], max_plies: int
) -> Optional[List[PlyEntry]]:
    entries: List[PlyEntry] = []
    for idx, fen in enumerate(fens):
        if idx >= len(moves) or idx >= max_plies:
            break
        if not isinstance(fen, str):
            return None
        try:
            key, side = zobrist_from_fen(fen)
        except ValueError:
            return None
        entries.append(PlyEntry(move=moves[idx], key=key, side=side, fen=fen))
    if not entries:
        return None
    return entries


def build_entries_from_replay(fens: List[str], moves: List[str], max_plies: int) -> List[PlyEntry]:
    entries: List[PlyEntry] = []
    for idx, fen in enumerate(fens):
        if idx >= len(moves) or idx >= max_plies:
            break
        move = moves[idx]
        key, side = zobrist_from_fen(fen)
        entries.append(PlyEntry(move=move, key=key, side=side, fen=fen))
    return entries


def score_move(ms: MoveStats, smoothing: float) -> float:
    prior = 0.5 * smoothing
    denom = ms.total + smoothing
    return (ms.wins + 0.5 * ms.draws + prior) / denom if denom > 0 else 0.0


def process_game(
    game_obj,
    replayer,
    args,
    fail_writer,
) -> Tuple[List[PlyEntry], Optional[float]]:
    moves = game_obj.get("moves") or game_obj.get("uci") or []
    if not isinstance(moves, list) or not moves:
        fail_writer("missing moves")
        return [], None
    result = parse_result_to_white_score(game_obj.get("result"))
    if result is None:
        fail_writer("invalid result")
        return [], None
    start_fen = game_obj.get("start_fen") or game_obj.get("initial_fen") or DEFAULT_START_FEN

    entries: Optional[List[PlyEntry]] = None
    ply_data = game_obj.get("ply_data")
    if isinstance(ply_data, list) and ply_data:
        parsed = entries_from_ply_data(ply_data, args.max_ply)
        if parsed is not None:
            entries = parsed
        else:
            fail_writer("invalid ply_data; falling back to replay")
    if entries is None:
        provided_fens = game_obj.get("fens")
        if isinstance(provided_fens, list) and provided_fens:
            parsed_fens = entries_from_fens_list(provided_fens, moves, args.max_ply)
            if parsed_fens is not None:
                entries = parsed_fens
            else:
                fail_writer("invalid fens list; falling back to replay")
    if entries is None:
        try:
            fens = replayer.replay(moves, start_fen=start_fen, max_plies=args.max_ply)
        except ReplayError as exc:
            fail_writer(str(exc))
            return [], None
        entries = build_entries_from_replay(fens, moves, args.max_ply)
    return entries, result


def write_json(obj, path: str):
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True)
    except OSError as exc:
        sys.stderr.write(f"Failed to write JSON to {path}: {exc}\n")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Mine forced-capture self-play for opening book moves.")
    parser.add_argument("--input", required=True, help="NDJSON self-play games (optionally gz).")
    parser.add_argument("--output-map", default="build/opening_book_map.json", help="Path for compact map hex_key->best UCI.")
    parser.add_argument("--output-details", default="build/opening_book_stats.jsonl", help="Path for detailed JSONL with per-move WDL.")
    parser.add_argument("--max-ply", type=int, default=8, help="Plies to consider from game start.")
    parser.add_argument("--min-samples", type=int, default=6, help="Minimum total samples to keep a position.")
    parser.add_argument("--top-n", type=int, default=2, help="Top-N moves to emit per position in details (map always picks best).")
    parser.add_argument("--smoothing", type=float, default=1.0, help="Laplace-style smoothing for score selection.")
    parser.add_argument("--engine-cmd", default="./program", help="Engine binary when python-chess is unavailable.")
    parser.add_argument("--engine-timeout", type=float, default=1.0, help="Seconds to wait for engine responses.")
    parser.add_argument("--prefer-python", action="store_true", help="Prefer python-chess replay (defaults to engine replay for variant correctness).")
    parser.add_argument("--fail-log", default="build/opening_miner_fail.log", help="Log path for bad games/plies.")
    parser.add_argument("--limit-games", type=int, default=None, help="Stop after processing this many games.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.fail_log)), exist_ok=True)
    fail_fh = open(args.fail_log, "w", encoding="utf-8")

    def log_fail(msg: str):
        fail_fh.write(msg + "\n")

    try:
        replayer = select_replayer(args)
    except ReplayError as exc:
        fail_fh.close()
        sys.stderr.write(f"Failed to prepare replayer: {exc}\n")
        sys.exit(1)

    positions: Dict[int, PositionStats] = {}
    collisions = 0
    games_total = 0
    games_ok = 0
    ply_records = 0

    with open_maybe_gzip(args.input) as fh:
        for line_no, line in enumerate(fh, 1):
            if args.limit_games is not None and games_total >= args.limit_games:
                break
            if not line.strip():
                continue
            games_total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                log_fail(f"line {line_no}: json decode error")
                continue
            entries, white_score = process_game(
                obj,
                replayer,
                args,
                fail_writer=lambda msg, ln=line_no: log_fail(f"line {ln}: {msg}"),
            )
            if not entries or white_score is None:
                continue
            games_ok += 1
            for idx, entry in enumerate(entries):
                key = entry.key
                move = entry.move
                side = entry.side
                fen = entry.fen
                outcome = 0.5
                if white_score == 1.0:
                    outcome = 1.0 if side == "w" else 0.0
                elif white_score == 0.0:
                    outcome = 0.0 if side == "w" else 1.0
                existing = positions.get(key)
                if existing is not None and existing.fen and fen and existing.fen != fen:
                    collisions += 1
                    log_fail(f"line {line_no}: zobrist collision, skipping ply {idx}")
                    continue
                if existing is None:
                    positions[key] = PositionStats(key=key, fen=fen)
                    existing = positions[key]
                elif existing.fen is None and fen:
                    existing.fen = fen
                existing.add(move, outcome)
                ply_records += 1

    if hasattr(replayer, "close"):
        try:
            replayer.close()
        except Exception:
            pass
    fail_fh.close()

    kept = 0
    book_map: Dict[str, str] = {}
    with open(args.output_details, "w", encoding="utf-8") as detail_fh:
        for key, pos in positions.items():
            if pos.total < args.min_samples:
                continue
            scored = []
            for mv, ms in pos.moves.items():
                scored.append(
                    (
                        score_move(ms, args.smoothing),
                        ms.total,
                        mv,
                        ms,
                    )
                )
            scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
            if not scored:
                continue
            top_moves = scored[: max(1, args.top_n)]
            book_map[f"{key:016x}"] = top_moves[0][2]
            entry = {
                "key_hex": f"{key:016x}",
                "fen": pos.fen or "",
                "total": pos.total,
                "moves": [
                    {
                        "uci": mv,
                        "wins": ms.wins,
                        "draws": ms.draws,
                        "losses": ms.losses,
                        "win_rate": ms.wins / ms.total if ms.total else 0.0,
                        "draw_rate": ms.draws / ms.total if ms.total else 0.0,
                        "loss_rate": ms.losses / ms.total if ms.total else 0.0,
                        "score": score,
                        "samples": ms.total,
                    }
                    for score, _, mv, ms in top_moves
                ],
            }
            detail_fh.write(json.dumps(entry) + "\n")
            kept += 1

    write_json(book_map, args.output_map)

    sys.stdout.write(
        f"Processed {games_ok}/{games_total} games, recorded {ply_records} plies. "
        f"Kept {kept} positions (>= {args.min_samples} samples). "
        f"Collisions: {collisions}. Map: {args.output_map}\n"
    )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
