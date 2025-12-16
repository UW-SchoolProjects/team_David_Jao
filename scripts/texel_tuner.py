#!/usr/bin/env python3
"""
Texel tuning (logistic regression) for PST and small eval terms.

Takes the deep-labeled CSV from Task 5.2.2 and fits:
- Middlegame / endgame PST values (per piece, white-oriented; black uses mirror).
- Bishop pair bonus.
- Tempo bonus.
- Optional endgame king-proximity term (aggression).
- Simple pawn/rook/king-structure scalars (passed/isolated/doubled pawns, rook open files, king pawn shield, pawn advance).

Outputs a C++ header with constexpr tables/constants ready to drop into eval.cpp.

Notes:
- Supports two objectives: `sign` (Texel logistic) and `cp` (centipawn regression).
- For `sign`, zero/blank labels are skipped (ambiguous target).
- Handles positions even if there are no legal moves (purely parses FEN, no move gen).
- Clamps PST weights to +/-200, small terms to +/-50 before emission.
"""

import argparse
import csv
import gzip
import math
import os
import random
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants (mirrors C++ eval.cpp conventions)
# ---------------------------------------------------------------------------

PHASE_WEIGHT = {
    "P": 0,
    "N": 1,
    "B": 1,
    "R": 2,
    "Q": 4,
    "K": 0,
}
PHASE_TOTAL = 24

MG_PIECE_VALUE = {
    "P": 100,
    "N": 330,
    "B": 500,
    "R": 550,
    "Q": 1500,
    "K": 0,
}

EG_PIECE_VALUE = {
    "P": 100,
    "N": 300,
    "B": 330,
    "R": 500,
    "Q": 950,
    "K": 0,
}

PIECE_ORDER = ["P", "N", "B", "R", "Q", "K"]  # maps to indices 0..5
PIECE_INDEX = {p: i for i, p in enumerate(PIECE_ORDER)}

# Feature indices
NUM_PST = len(PIECE_ORDER) * 64
IDX_MG_START = 0
IDX_EG_START = NUM_PST
IDX_BISHOP_PAIR = NUM_PST * 2
IDX_TEMPO = NUM_PST * 2 + 1
IDX_KING_PROX = NUM_PST * 2 + 2
IDX_BIAS = NUM_PST * 2 + 3
IDX_PAWN_ADVANCE = NUM_PST * 2 + 4
IDX_ISOLATED_PAWN = NUM_PST * 2 + 5
IDX_DOUBLED_PAWN = NUM_PST * 2 + 6
IDX_PASSED_PAWN = NUM_PST * 2 + 7
IDX_ROOK_OPEN_FILE = NUM_PST * 2 + 8
IDX_ROOK_SEMI_OPEN_FILE = NUM_PST * 2 + 9
IDX_KING_SHIELD = NUM_PST * 2 + 10
NUM_FEATURES = NUM_PST * 2 + 11


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def open_maybe_gzip(path: str, mode: str):
    is_text = "b" not in mode
    def normalized(m: str) -> str:
        if is_text:
            return m if "t" in m else m + "t"
        return m if "b" in m else m + "b"

    normalized_mode = normalized(mode)
    if path.endswith(".gz"):
        if is_text:
            return gzip.open(path, normalized_mode, encoding="utf-8", newline="", errors="replace")
        return gzip.open(path, normalized_mode)
    if is_text:
        return open(path, normalized_mode, encoding="utf-8", newline="", errors="replace")
    return open(path, normalized_mode)


def mirror_square(sq64: int) -> int:
    """Vertical mirror: a1=0 -> a8=56, h1=7 -> h8=63."""
    rank = sq64 // 8
    file = sq64 % 8
    return (7 - rank) * 8 + file


def manhattan(a: int, b: int) -> int:
    af, ar = a % 8, a // 8
    bf, br = b % 8, b // 8
    return abs(af - bf) + abs(ar - br)


def parse_fen(fen: str):
    """Minimal FEN parser returning (squares[64], side_to_move) or (None, None) on failure."""
    parts = fen.strip().split()
    if len(parts) < 2:
        return None, None
    placement, side = parts[0], parts[1]
    if side not in ("w", "b"):
        return None, None
    squares = [None] * 64
    rank = 7
    file = 0
    ranks_seen = 0
    legal_pieces = set("PNBRQKpnbrqk")
    for ch in placement:
        if ch == "/":
            if file != 8 or ranks_seen >= 7:
                return None, None
            ranks_seen += 1
            rank -= 1
            file = 0
            continue
        if ch.isdigit():
            file += int(ch)
            if file > 8:
                return None, None
            continue
        if ch not in legal_pieces or file >= 8 or rank < 0:
            return None, None
        squares[rank * 8 + file] = ch
        file += 1
        if file > 8:
            return None, None
    if ranks_seen != 7 or file != 8:
        return None, None
    return squares, side


def normalize_fen_key(fen: str) -> Optional[str]:
    """Normalize a FEN for dedup/splitting: keep only placement, stm, castling, ep."""
    parts = fen.strip().split()
    if len(parts) < 4:
        return None
    return " ".join(parts[:4])


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    base: float
    features: List[Tuple[int, float]]
    label: float  # objective-dependent (e.g., cp or +/-1)
    game_id: Optional[str] = None
    fen_key: Optional[str] = None


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def build_sample(fen: str, label_cp: int, objective: str, label_clamp: int) -> Optional[Sample]:
    squares, side = parse_fen(fen)
    if squares is None or side not in ("w", "b"):
        return None
    if objective == "sign" and label_cp == 0:
        return None
    if label_clamp > 0:
        label_cp = max(-label_clamp, min(label_clamp, label_cp))

    stm_sign = 1 if side == "w" else -1

    phase = 0
    mg_base = 0.0
    eg_base = 0.0
    white_bish = 0
    black_bish = 0
    king_sq = {"w": None, "b": None}
    pawn_sq = {"w": [], "b": []}
    rook_sq = {"w": [], "b": []}

    features: List[Tuple[int, float]] = []

    for idx, piece in enumerate(squares):
        if piece is None:
            continue
        color = "w" if piece.isupper() else "b"
        p = piece.upper()
        if p not in PIECE_INDEX:
            continue
        phase += PHASE_WEIGHT.get(p, 0)
        pt_idx = PIECE_INDEX[p]
        sign_color = 1 if color == "w" else -1
        # PST uses white-oriented squares; black mirrored.
        pst_sq = idx if color == "w" else mirror_square(idx)

        mg_base += sign_color * MG_PIECE_VALUE[p]
        eg_base += sign_color * EG_PIECE_VALUE[p]

        mg_feature_idx = IDX_MG_START + pt_idx * 64 + pst_sq
        eg_feature_idx = IDX_EG_START + pt_idx * 64 + pst_sq

        # Phase weights apply later; incorporate them into the feature value.
        # STM orientation is encoded via stm_sign.
        features.append((mg_feature_idx, sign_color * stm_sign))
        features.append((eg_feature_idx, sign_color * stm_sign))

        if p == "B":
            if color == "w":
                white_bish += 1
            else:
                black_bish += 1
        if p == "K":
            king_sq[color] = idx
        if p == "P":
            pawn_sq[color].append(idx)
        if p == "R":
            rook_sq[color].append(idx)

    phase = max(0, min(PHASE_TOTAL, phase))
    mg_wt = phase / PHASE_TOTAL
    eg_wt = (PHASE_TOTAL - phase) / PHASE_TOTAL

    # Scale PST features by phase weights.
    scaled_features: List[Tuple[int, float]] = []
    for idx, val in features:
        if idx < IDX_EG_START:
            scaled_features.append((idx, val * mg_wt))
        else:
            scaled_features.append((idx, val * eg_wt))

    # Base material blended and oriented to STM.
    base = stm_sign * (mg_wt * mg_base + eg_wt * eg_base)

    # Bishop pair (phase-independent).
    bp_feat = (1 if white_bish >= 2 else 0) - (1 if black_bish >= 2 else 0)
    if bp_feat:
        scaled_features.append((IDX_BISHOP_PAIR, bp_feat * stm_sign))

    # Tempo always helps side to move.
    scaled_features.append((IDX_TEMPO, 1.0))

    # Endgame king proximity (encourage STM king approaching opponent in EG).
    if king_sq["w"] is not None and king_sq["b"] is not None:
        dist = manhattan(king_sq["w"], king_sq["b"])
        prox = max(0, 10 - dist)
        if prox > 0:
            # This term is applied in White POV then converted to side-to-move POV,
            # so it must be oriented by stm_sign here.
            scaled_features.append((IDX_KING_PROX, prox * eg_wt * stm_sign))

    # Pawn/rook/king structure features (FEN-only; keep them as simple scalars).
    # In eval.cpp these are added in White POV and then converted to STM POV,
    # so we orient by stm_sign here.
    if pawn_sq["w"] or pawn_sq["b"]:
        wp_files = [0] * 8
        bp_files = [0] * 8
        wp_rank_mask = [0] * 8  # 8-bit mask of ranks containing a pawn on each file
        bp_rank_mask = [0] * 8
        for sq in pawn_sq["w"]:
            f, r = sq % 8, sq // 8
            wp_files[f] += 1
            wp_rank_mask[f] |= 1 << r
        for sq in pawn_sq["b"]:
            f, r = sq % 8, sq // 8
            bp_files[f] += 1
            bp_rank_mask[f] |= 1 << r

        # Pawn advancement: sum(rank) for white minus sum(7-rank) for black.
        pawn_adv = sum((sq // 8) for sq in pawn_sq["w"]) - sum((7 - (sq // 8)) for sq in pawn_sq["b"])
        if pawn_adv:
            scaled_features.append((IDX_PAWN_ADVANCE, pawn_adv * stm_sign))

        # Isolated pawns: no friendly pawn on adjacent files.
        isolated_w = 0
        for sq in pawn_sq["w"]:
            f = sq % 8
            left = (f > 0) and wp_files[f - 1] > 0
            right = (f < 7) and wp_files[f + 1] > 0
            if not left and not right:
                isolated_w += 1
        isolated_b = 0
        for sq in pawn_sq["b"]:
            f = sq % 8
            left = (f > 0) and bp_files[f - 1] > 0
            right = (f < 7) and bp_files[f + 1] > 0
            if not left and not right:
                isolated_b += 1
        isolated_diff = isolated_b - isolated_w
        if isolated_diff:
            scaled_features.append((IDX_ISOLATED_PAWN, isolated_diff * stm_sign))

        # Doubled pawns: extra pawns per file.
        doubled_w = sum(max(0, c - 1) for c in wp_files)
        doubled_b = sum(max(0, c - 1) for c in bp_files)
        doubled_diff = doubled_b - doubled_w
        if doubled_diff:
            scaled_features.append((IDX_DOUBLED_PAWN, doubled_diff * stm_sign))

        # Passed pawns: no enemy pawn on same/adjacent files ahead. EG-weighted.
        passed_w_sum = 0
        for sq in pawn_sq["w"]:
            f, r = sq % 8, sq // 8
            above = (~((1 << (r + 1)) - 1)) & 0xFF  # ranks > r
            blocked = False
            for ff in (f - 1, f, f + 1):
                if 0 <= ff < 8 and (bp_rank_mask[ff] & above):
                    blocked = True
                    break
            if not blocked:
                passed_w_sum += r
        passed_b_sum = 0
        for sq in pawn_sq["b"]:
            f, r = sq % 8, sq // 8
            below = (1 << r) - 1  # ranks < r
            blocked = False
            for ff in (f - 1, f, f + 1):
                if 0 <= ff < 8 and (wp_rank_mask[ff] & below):
                    blocked = True
                    break
            if not blocked:
                passed_b_sum += (7 - r)
        passed_diff = passed_w_sum - passed_b_sum
        if passed_diff:
            scaled_features.append((IDX_PASSED_PAWN, passed_diff * eg_wt * stm_sign))

        # Rook open/semi-open files (MG-weighted): requires pawn file counts.
        if rook_sq["w"] or rook_sq["b"]:
            open_w = semi_w = 0
            for sq in rook_sq["w"]:
                f = sq % 8
                if wp_files[f] != 0:
                    continue
                if bp_files[f] == 0:
                    open_w += 1
                else:
                    semi_w += 1
            open_b = semi_b = 0
            for sq in rook_sq["b"]:
                f = sq % 8
                if bp_files[f] != 0:
                    continue
                if wp_files[f] == 0:
                    open_b += 1
                else:
                    semi_b += 1
            open_diff = open_w - open_b
            semi_diff = semi_w - semi_b
            if open_diff:
                scaled_features.append((IDX_ROOK_OPEN_FILE, open_diff * mg_wt * stm_sign))
            if semi_diff:
                scaled_features.append((IDX_ROOK_SEMI_OPEN_FILE, semi_diff * mg_wt * stm_sign))

        # King pawn shield (MG-weighted): count friendly pawns on 1-2 ranks in front of king.
        def king_shield(color_key: str) -> int:
            k = king_sq.get(color_key)
            if k is None:
                return 0
            f0, r0 = k % 8, k // 8
            shield = 0
            if color_key == "w":
                for df in (-1, 0, 1):
                    f = f0 + df
                    if not (0 <= f < 8):
                        continue
                    for step in (1, 2):
                        r = r0 + step
                        if r >= 8:
                            continue
                        sq = r * 8 + f
                        if squares[sq] == "P":
                            shield += 1
            else:
                for df in (-1, 0, 1):
                    f = f0 + df
                    if not (0 <= f < 8):
                        continue
                    for step in (1, 2):
                        r = r0 - step
                        if r < 0:
                            continue
                        sq = r * 8 + f
                        if squares[sq] == "p":
                            shield += 1
            return shield

        shield_diff = king_shield("w") - king_shield("b")
        if shield_diff:
            scaled_features.append((IDX_KING_SHIELD, shield_diff * mg_wt * stm_sign))

    # Bias term
    scaled_features.append((IDX_BIAS, 1.0))

    if objective == "sign":
        label: float = 1.0 if label_cp > 0 else -1.0
    elif objective == "cp":
        # Side-to-move oriented centipawns (already oriented in the dataset); keep as a regression target.
        label = float(label_cp)
    else:
        raise ValueError(f"Unknown objective: {objective}")
    return Sample(base=base, features=scaled_features, label=label)


# ---------------------------------------------------------------------------
# Training (simple SGD with L2 regularization)
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def train(samples: List[Sample], args):
    objective = args.objective
    optimizer = getattr(args, "optimizer", "sgd")
    if objective == "cp" and optimizer == "ridge":
        return train_ridge(samples, args)

    w = [0.0] * NUM_FEATURES
    lr = args.lr
    batch_size = args.batch_size
    l2 = args.l2
    scale = args.logistic_scale
    if objective == "cp":
        scale = float(args.mse_scale)
        if scale <= 0:
            raise ValueError("--mse-scale must be > 0 for objective=cp")

    if optimizer not in ("sgd", "adam"):
        raise ValueError(f"Unknown optimizer: {optimizer}")

    m = None
    v = None
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    step = 0
    if optimizer == "adam":
        m = [0.0] * NUM_FEATURES
        v = [0.0] * NUM_FEATURES

    for epoch in range(args.epochs):
        random.shuffle(samples)
        total_loss = 0.0
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            grad = [0.0] * NUM_FEATURES
            for s in batch:
                pred = s.base
                for idx, val in s.features:
                    pred += w[idx] * val
                if objective == "sign":
                    y = s.label
                    z = -y * pred / scale
                    # loss = log(1 + exp(z))
                    if z > 50:
                        loss = z  # avoid overflow
                    else:
                        loss = math.log1p(math.exp(z))
                    total_loss += loss
                    sig = sigmoid(z)
                    g_factor = -y * sig / scale
                    for idx, val in s.features:
                        grad[idx] += g_factor * val
                elif objective == "cp":
                    # Squared error on centipawns, normalized by mse_scale to keep gradients stable.
                    # loss = 0.5 * ((pred - y)/mse_scale)^2
                    y = s.label
                    err = (pred - y) / scale
                    total_loss += 0.5 * err * err
                    g_factor = err / scale  # d(loss)/d(pred)
                    for idx, val in s.features:
                        grad[idx] += g_factor * val
                else:
                    raise ValueError(f"Unknown objective: {objective}")
            inv_bs = 1.0 / max(1, len(batch))
            if optimizer == "sgd":
                for i in range(NUM_FEATURES):
                    g = grad[i] * inv_bs + l2 * w[i]
                    w[i] -= lr * g
            else:
                step += 1
                assert m is not None and v is not None
                bias_correction1 = 1.0 - (beta1**step)
                bias_correction2 = 1.0 - (beta2**step)
                for i in range(NUM_FEATURES):
                    g = grad[i] * inv_bs + l2 * w[i]
                    mi = m[i] = beta1 * m[i] + (1.0 - beta1) * g
                    vi = v[i] = beta2 * v[i] + (1.0 - beta2) * (g * g)
                    m_hat = mi / max(1e-12, bias_correction1)
                    v_hat = vi / max(1e-12, bias_correction2)
                    w[i] -= lr * m_hat / (math.sqrt(v_hat) + eps)

        avg_loss = total_loss / max(1, len(samples))
        if objective == "cp":
            rmse_est = math.sqrt(max(0.0, 2.0 * avg_loss)) * scale
            max_w = max(abs(x) for x in w) if w else 0.0
            print(f"Epoch {epoch+1}/{args.epochs}: avg_loss={avg_loss:.4f} (rmse≈{rmse_est:.1f} cp, max|w|={max_w:.3f})")
        else:
            max_w = max(abs(x) for x in w) if w else 0.0
            print(f"Epoch {epoch+1}/{args.epochs}: avg_loss={avg_loss:.4f} (max|w|={max_w:.3f})")
    return w


def train_ridge(samples: List[Sample], args):
    if args.objective != "cp":
        raise ValueError("optimizer=ridge is only supported for objective=cp")
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise RuntimeError("optimizer=ridge requires numpy; use --optimizer adam/sgd instead") from e

    n = len(samples)
    if n == 0:
        return [0.0] * NUM_FEATURES

    scale = float(args.mse_scale)
    if scale <= 0:
        raise ValueError("--mse-scale must be > 0 for objective=cp")

    X = np.zeros((n, NUM_FEATURES), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    for i, s in enumerate(samples):
        y[i] = s.label - s.base
        for idx, val in s.features:
            X[i, idx] = val
    # Objective in SGD form:
    #   (1/(2n)) * ||(Xw - y)/scale||^2 + (l2/2) * ||w||^2
    # => normal eq:
    #   (XᵀX + n*scale^2*l2 * I) w = Xᵀy
    lam = float(n) * (scale * scale) * float(args.l2)
    if lam <= 0:
        # With l2=0, XᵀX can be singular (unused/collinear features). Use SVD least-squares.
        w, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        max_w = float(np.max(np.abs(w))) if w.size else 0.0
        print(f"OLS solve: rank={rank}/{NUM_FEATURES} max|w|={max_w:.3f} (lambda={lam:.3f})")
        return w.tolist()

    xtx = X.T @ X
    xty = X.T @ y
    xtx.flat[:: NUM_FEATURES + 1] += lam
    try:
        w = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        # Extremely ill-conditioned; fall back to least-squares on the normal equations.
        w, *_ = np.linalg.lstsq(xtx, xty, rcond=None)
    max_w = float(np.max(np.abs(w))) if w.size else 0.0
    print(f"Ridge solve: max|w|={max_w:.3f} (lambda={lam:.3f})")
    return w.tolist()


def predict_sign(sample: Sample, w) -> int:
    pred = sample.base
    for idx, val in sample.features:
        pred += w[idx] * val
    return 1 if pred >= 0 else -1


def accuracy(samples: List[Sample], w) -> float:
    if not samples:
        return 0.0
    correct = 0
    for s in samples:
        true_sign = 1 if s.label >= 0 else -1
        if predict_sign(s, w) == true_sign:
            correct += 1
    return correct / len(samples)


def rmse(samples: List[Sample], w) -> float:
    if not samples:
        return 0.0
    se = 0.0
    for s in samples:
        pred = s.base
        for idx, val in s.features:
            pred += w[idx] * val
        err = pred - s.label
        se += err * err
    return math.sqrt(se / len(samples))


# ---------------------------------------------------------------------------
# Splitting / loading helpers
# ---------------------------------------------------------------------------

def split_train_val(samples: List[Sample], holdout: float, seed: int, split_by: str):
    if not samples:
        return [], []

    h = max(0.0, min(0.95, float(holdout)))
    rng = random.Random(seed)

    if split_by == "row":
        shuffled = list(samples)
        rng.shuffle(shuffled)
        split = int(len(shuffled) * (1.0 - h))
        return shuffled[:split], shuffled[split:]

    if split_by != "game":
        raise ValueError(f"Unknown split_by: {split_by}")

    # Group by game_id to prevent leakage between train/val from the same self-play game.
    groups: Dict[str, List[Sample]] = {}
    for i, s in enumerate(samples):
        gid = (s.game_id or "").strip()
        if not gid:
            gid = f"__row_{i}"
        groups.setdefault(gid, []).append(s)

    game_ids = list(groups.keys())
    rng.shuffle(game_ids)

    target_val = int(round(len(samples) * h))
    val: List[Sample] = []
    train: List[Sample] = []
    val_count = 0
    for gid in game_ids:
        bucket = groups[gid]
        if val_count < target_val:
            val.extend(bucket)
            val_count += len(bucket)
        else:
            train.extend(bucket)

    # Guard against pathological cases (e.g., 1 game).
    if not train and val:
        train, val = val, train
    if not val and train:
        # Keep at least one sample in val if possible.
        val.append(train.pop())
    return train, val


def parse_l2_grid(spec: str) -> List[float]:
    vals: List[float] = []
    for tok in (spec or "").split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            v = float(t)
        except ValueError as e:
            raise ValueError(f"Invalid --l2-grid entry: {t!r}") from e
        if v < 0:
            raise ValueError(f"--l2-grid entries must be >= 0, got {v}")
        vals.append(v)
    if not vals:
        raise ValueError("--l2-grid parsed to an empty list")
    return vals


# ---------------------------------------------------------------------------
# Header emission
# ---------------------------------------------------------------------------

def clamp_and_round(value: float, clamp: float) -> int:
    # Sanitize non-finite values to zero to avoid emitting invalid headers.
    if not math.isfinite(value):
        value = 0.0
    value = max(-clamp, min(clamp, value))
    return int(round(value))


def clamp_for_emission(w, pst_clamp: float, small_clamp: float, tempo_clamp: float, bias_clamp: float):
    """Return the weights as they will be emitted to the C++ header (clamped + rounded)."""
    w_out = list(w)
    for pt_idx, _ in enumerate(PIECE_ORDER):
        for sq in range(64):
            mg_idx = IDX_MG_START + pt_idx * 64 + sq
            eg_idx = IDX_EG_START + pt_idx * 64 + sq
            w_out[mg_idx] = float(clamp_and_round(w_out[mg_idx], pst_clamp))
            w_out[eg_idx] = float(clamp_and_round(w_out[eg_idx], pst_clamp))

    w_out[IDX_BISHOP_PAIR] = float(clamp_and_round(w_out[IDX_BISHOP_PAIR], small_clamp))
    w_out[IDX_KING_PROX] = float(clamp_and_round(w_out[IDX_KING_PROX], small_clamp))
    w_out[IDX_TEMPO] = float(clamp_and_round(w_out[IDX_TEMPO], tempo_clamp))
    w_out[IDX_BIAS] = float(clamp_and_round(w_out[IDX_BIAS], bias_clamp))
    for idx in (
        IDX_PAWN_ADVANCE,
        IDX_ISOLATED_PAWN,
        IDX_DOUBLED_PAWN,
        IDX_PASSED_PAWN,
        IDX_ROOK_OPEN_FILE,
        IDX_ROOK_SEMI_OPEN_FILE,
        IDX_KING_SHIELD,
    ):
        w_out[idx] = float(clamp_and_round(w_out[idx], small_clamp))
    return w_out


def emit_header(w, path: str, pst_clamp: float, small_clamp: float, tempo_clamp: float, bias_clamp: float):
    if len(w) < NUM_FEATURES:
        raise ValueError(f"Weight vector too small: got {len(w)}, expected >= {NUM_FEATURES}")
    dirpath = os.path.dirname(os.path.abspath(path))
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    mg = [[0] * 64 for _ in PIECE_ORDER]
    eg = [[0] * 64 for _ in PIECE_ORDER]

    for pt_idx, _ in enumerate(PIECE_ORDER):
        for sq in range(64):
            mg_idx = IDX_MG_START + pt_idx * 64 + sq
            eg_idx = IDX_EG_START + pt_idx * 64 + sq
            if mg_idx >= len(w) or eg_idx >= len(w):
                raise IndexError(f"PST index out of range (mg:{mg_idx}, eg:{eg_idx}, len(w)={len(w)})")
            mg[pt_idx][sq] = clamp_and_round(w[mg_idx], pst_clamp)
            eg[pt_idx][sq] = clamp_and_round(w[eg_idx], pst_clamp)

    for idx_name, idx in (
        ("BISHOP_PAIR", IDX_BISHOP_PAIR),
        ("TEMPO", IDX_TEMPO),
        ("KING_PROX", IDX_KING_PROX),
        ("BIAS", IDX_BIAS),
    ):
        if idx >= len(w):
            raise IndexError(f"{idx_name} index {idx} out of range for len(w)={len(w)}")

    bp = clamp_and_round(w[IDX_BISHOP_PAIR], small_clamp)
    tempo = clamp_and_round(w[IDX_TEMPO], tempo_clamp)
    king_prox = clamp_and_round(w[IDX_KING_PROX], small_clamp)
    bias = clamp_and_round(w[IDX_BIAS], bias_clamp)
    pawn_adv = clamp_and_round(w[IDX_PAWN_ADVANCE], small_clamp)
    iso = clamp_and_round(w[IDX_ISOLATED_PAWN], small_clamp)
    doubled = clamp_and_round(w[IDX_DOUBLED_PAWN], small_clamp)
    passed = clamp_and_round(w[IDX_PASSED_PAWN], small_clamp)
    rook_open = clamp_and_round(w[IDX_ROOK_OPEN_FILE], small_clamp)
    rook_semi = clamp_and_round(w[IDX_ROOK_SEMI_OPEN_FILE], small_clamp)
    shield = clamp_and_round(w[IDX_KING_SHIELD], small_clamp)

    def fmt_table(table):
        lines = []
        for row in table:
            rows = []
            for r in range(0, 64, 8):
                rows.append(", ".join(f"{row[r + c]:4d}" for c in range(8)))
            lines.append("    {" + ",\n     ".join(rows) + "}")
        return ",\n".join(lines)

    guard = "WEIGHTS_EVALUATED_H"
    header = f"""// Auto-generated by scripts/texel_tuner.py
// Do not edit by hand. Values are from Texel logistic regression.

#ifndef {guard}
#define {guard}
#pragma once

constexpr int TUNED_PST_MG[7][64] = {{
    {{0}},
{fmt_table(mg)}
}};

constexpr int TUNED_PST_EG[7][64] = {{
    {{0}},
{fmt_table(eg)}
}};

constexpr int TUNED_BISHOP_PAIR = {bp};
constexpr int TUNED_TEMPO = {tempo};
constexpr int TUNED_KING_PROX = {king_prox};
constexpr int TUNED_EVAL_BIAS = {bias};
constexpr int TUNED_PAWN_ADVANCE = {pawn_adv};
constexpr int TUNED_ISOLATED_PAWN = {iso};
constexpr int TUNED_DOUBLED_PAWN = {doubled};
constexpr int TUNED_PASSED_PAWN = {passed};
constexpr int TUNED_ROOK_OPEN_FILE = {rook_open};
constexpr int TUNED_ROOK_SEMI_OPEN_FILE = {rook_semi};
constexpr int TUNED_KING_SHIELD = {shield};

#endif // {guard}
"""
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write(header.rstrip() + "\n")
    print(f"Wrote tuned weights to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Texel tuner for PST and small eval scalars.")
    p.add_argument("--input", default="build/deep_labeled_positions.csv.gz", help="Input CSV (.gz ok) from deep_labeler.")
    p.add_argument("--output-header", default="src/core/gameTreeSearch/weights_evaluated.h", help="Header path to write tuned weights.")
    p.add_argument("--holdout", type=float, default=0.2, help="Fraction of data for validation.")
    p.add_argument(
        "--split-by",
        choices=("row", "game"),
        default="row",
        help="Train/val split strategy: row (random rows) or game (group by game_id).",
    )
    p.add_argument("--max-samples", type=int, default=0, help="Limit number of samples (0 = all).")
    p.add_argument("--seed", type=int, default=13, help="RNG seed.")
    p.add_argument("--epochs", type=int, default=8, help="Training epochs.")
    p.add_argument(
        "--optimizer",
        choices=("adam", "sgd", "ridge"),
        default=None,
        help="Optimizer/solver: ridge (closed-form for objective=cp), or adam/sgd (iterative).",
    )
    p.add_argument("--batch-size", type=int, default=128, help="Batch size for training.")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    p.add_argument("--l2", type=float, default=1e-4, help="L2 weight decay.")
    p.add_argument(
        "--l2-grid",
        default="",
        help="Comma-separated l2 values to try (ridge + objective=cp only); best val RMSE is selected.",
    )
    p.add_argument("--logistic-scale", type=float, default=400.0, help="Scale for Texel logistic loss.")
    p.add_argument("--mse-scale", type=float, default=50.0, help="Normalization for cp MSE (objective=cp).")
    p.add_argument("--pst-clamp", type=float, default=200.0, help="Clamp for PST weights.")
    p.add_argument(
        "--small-clamp",
        type=float,
        default=50.0,
        help="Clamp for small scalar terms (bishop pair/king prox/pawn structure/rook files/king shield).",
    )
    p.add_argument("--tempo-clamp", type=float, default=50.0, help="Clamp for tempo bonus.")
    p.add_argument("--bias-clamp", type=float, default=50.0, help="Clamp for eval bias.")
    p.add_argument("--objective", choices=("sign", "cp"), default="cp", help="Training objective: sign-classification or cp-regression.")
    p.add_argument("--require-deep", action="store_true", help="Only use eval_deep_cp; skip rows without deep labels.")
    p.add_argument("--label-clamp", type=int, default=2000, help="Clamp labels to +/-N centipawns (0 disables).")
    p.add_argument(
        "--dedup",
        choices=("none", "first", "mean"),
        default="none",
        help="Deduplicate by normalized FEN (first 4 fields). 'mean' averages labels per unique FEN.",
    )
    return p.parse_args()


def load_samples(path: str, max_samples: int, objective: str, require_deep: bool, label_clamp: int, dedup: str) -> List[Sample]:
    samples: List[Sample] = []
    bad_rows = 0
    dup_rows = 0

    if dedup not in ("none", "first", "mean"):
        raise ValueError(f"Unknown dedup mode: {dedup}")

    agg: Dict[str, Tuple[str, int, int, str]] = {}
    seen = set()
    with open_maybe_gzip(path, "rt") as fh:
        reader = csv.DictReader(fh)
        for row_idx, row in enumerate(reader):
            fen = (row.get("fen") or "").strip()
            if not fen:
                bad_rows += 1
                continue
            fen_key = normalize_fen_key(fen)
            if fen_key is None:
                bad_rows += 1
                continue

            score_str = row.get("eval_deep_cp") if require_deep else (row.get("eval_deep_cp") or row.get("eval_cp"))
            if score_str is None:
                bad_rows += 1
                continue
            try:
                score = int(score_str)
            except (TypeError, ValueError):
                bad_rows += 1
                continue

            gid = (row.get("game_id") or "").strip()

            if dedup == "none":
                s = build_sample(fen, score, objective=objective, label_clamp=label_clamp)
                if s is None:
                    continue
                s.game_id = gid or None
                s.fen_key = fen_key
                samples.append(s)
                if max_samples and len(samples) >= max_samples:
                    break
                continue

            if dedup == "first":
                if fen_key in seen:
                    dup_rows += 1
                    continue
                seen.add(fen_key)
                s = build_sample(fen, score, objective=objective, label_clamp=label_clamp)
                if s is None:
                    continue
                s.game_id = gid or None
                s.fen_key = fen_key
                samples.append(s)
                if max_samples and len(samples) >= max_samples:
                    break
                continue

            # dedup == "mean": keep up to max_samples unique keys but aggregate across the whole file
            if fen_key in agg:
                fen0, sum_score, count, gid0 = agg[fen_key]
                agg[fen_key] = (fen0, sum_score + score, count + 1, gid0)
                continue
            if max_samples and len(agg) >= max_samples:
                dup_rows += 1
                continue
            agg[fen_key] = (fen, score, 1, gid)

    if dedup == "mean":
        for fen_key, (fen, sum_score, count, gid) in agg.items():
            mean_score = int(round(sum_score / max(1, count)))
            s = build_sample(fen, mean_score, objective=objective, label_clamp=label_clamp)
            if s is None:
                continue
            s.game_id = gid.strip() or None
            s.fen_key = fen_key
            samples.append(s)

    if bad_rows:
        print(f"Skipped {bad_rows} rows due to missing/invalid fen or score.")
    if dedup != "none":
        if dup_rows:
            print(f"Dedup ({dedup}): skipped {dup_rows} duplicate/extra rows.")
        print(f"Dedup ({dedup}): kept {len(samples)} unique positions.")
    return samples


def main():
    args = parse_args()
    if args.optimizer is None:
        args.optimizer = "ridge" if args.objective == "cp" else "adam"
    random.seed(args.seed)
    samples = load_samples(args.input, args.max_samples, args.objective, args.require_deep, args.label_clamp, args.dedup)
    if not samples:
        print("No samples loaded; aborting.")
        return

    train_samples, val_samples = split_train_val(samples, args.holdout, args.seed, args.split_by)
    if not train_samples or not val_samples:
        print("Train/val split failed (need at least 1 sample in each); aborting.")
        return

    unique_games = {s.game_id for s in samples if (s.game_id or "").strip()}
    unique_fens = {s.fen_key for s in samples if s.fen_key}
    if unique_games:
        print(f"Loaded {len(samples)} samples (games {len(unique_games)}, unique_fen {len(unique_fens)}).")
    else:
        print(f"Loaded {len(samples)} samples (unique_fen {len(unique_fens)}).")
    print(f"Split ({args.split_by}): train {len(train_samples)}, val {len(val_samples)}.")

    w0 = [0.0] * NUM_FEATURES
    base_train_acc = accuracy(train_samples, w0)
    base_val_acc = accuracy(val_samples, w0)
    if args.objective == "cp":
        base_train_rmse = rmse(train_samples, w0)
        base_val_rmse = rmse(val_samples, w0)
        print(
            f"Baseline RMSE(cp): train={base_train_rmse:.1f} val={base_val_rmse:.1f} | "
            f"sign_acc: train={base_train_acc*100:.2f}% val={base_val_acc*100:.2f}%"
        )
    else:
        print(f"Baseline sign accuracy: train={base_train_acc*100:.2f}% val={base_val_acc*100:.2f}%")

    w = None
    if args.objective == "cp" and args.optimizer == "ridge" and args.l2_grid.strip():
        grid = parse_l2_grid(args.l2_grid)
        best = None
        best_l2 = None
        best_val_rmse = None
        print(f"Ridge sweep over {len(grid)} l2 values...")
        for l2 in grid:
            args2 = copy.copy(args)
            args2.l2 = l2
            w_try = train_ridge(train_samples, args2)
            val_rmse = rmse(val_samples, w_try)
            train_rmse = rmse(train_samples, w_try)
            val_acc = accuracy(val_samples, w_try)
            print(f"  l2={l2:g}: rmse train={train_rmse:.1f} val={val_rmse:.1f} | sign_acc val={val_acc*100:.2f}%")
            if best_val_rmse is None or val_rmse < best_val_rmse:
                best = w_try
                best_l2 = l2
                best_val_rmse = val_rmse
        assert best is not None and best_l2 is not None
        args.l2 = float(best_l2)
        w = best
        print(f"Selected l2={args.l2:g} (val RMSE {best_val_rmse:.1f}).")
    else:
        w = train(train_samples, args)

    train_acc = accuracy(train_samples, w)
    val_acc = accuracy(val_samples, w)
    if args.objective == "cp":
        train_rmse = rmse(train_samples, w)
        val_rmse = rmse(val_samples, w)
        print(f"RMSE(cp): train={train_rmse:.1f} val={val_rmse:.1f} | sign_acc: train={train_acc*100:.2f}% val={val_acc*100:.2f}%")
    else:
        print(f"Texel sign accuracy: train={train_acc*100:.2f}% val={val_acc*100:.2f}%")

    w_emitted = clamp_for_emission(w, args.pst_clamp, args.small_clamp, args.tempo_clamp, args.bias_clamp)
    if any(a != b for a, b in zip(w, w_emitted)):
        train_acc_e = accuracy(train_samples, w_emitted)
        val_acc_e = accuracy(val_samples, w_emitted)
        if args.objective == "cp":
            train_rmse_e = rmse(train_samples, w_emitted)
            val_rmse_e = rmse(val_samples, w_emitted)
            print(
                f"RMSE(cp, emitted): train={train_rmse_e:.1f} val={val_rmse_e:.1f} | "
                f"sign_acc: train={train_acc_e*100:.2f}% val={val_acc_e*100:.2f}%"
            )
        else:
            print(f"Texel sign accuracy (emitted): train={train_acc_e*100:.2f}% val={val_acc_e*100:.2f}%")

    emit_header(w_emitted, args.output_header, args.pst_clamp, args.small_clamp, args.tempo_clamp, args.bias_clamp)


if __name__ == "__main__":
    main()
