#!/usr/bin/env python3
"""
Texel tuning (logistic regression) for PST and small eval terms.

Takes the deep-labeled CSV from Task 5.2.2 and fits:
- Middlegame / endgame PST values (per piece, white-oriented; black uses mirror).
- Bishop pair bonus.
- Tempo bonus.
- Optional endgame king-proximity term (aggression).

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
import json
import math
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
NUM_FEATURES = NUM_PST * 2 + 4


# ---------------------------------------------------------------------------
# Extra eval tunables (emitted into weights_evaluated.h)
# ---------------------------------------------------------------------------

def default_extra_params():
    # These are not optimized by this script yet; they are emitted so the C++
    # eval can be tuned from a single generated header.
    return {
        "TUNED_MG_PIECE_VALUE": [0, MG_PIECE_VALUE["P"], MG_PIECE_VALUE["N"], MG_PIECE_VALUE["B"], MG_PIECE_VALUE["R"], MG_PIECE_VALUE["Q"], MG_PIECE_VALUE["K"]],
        "TUNED_EG_PIECE_VALUE": [0, EG_PIECE_VALUE["P"], EG_PIECE_VALUE["N"], EG_PIECE_VALUE["B"], EG_PIECE_VALUE["R"], EG_PIECE_VALUE["Q"], EG_PIECE_VALUE["K"]],
        "TUNED_CENTER_MG_WEIGHT": [0, 6, -2, -3, -3, -5, 0],
        "TUNED_CENTER_EG_WEIGHT": [0, 2, 2, 3, 3, 4, 6],
        "TUNED_KING_PROX_DIST_BASE": 10,
        "TUNED_TRADE_RISK_SCALE_PCT": 70,
        "TUNED_MK_BONUS_MAX": 100,
        "TUNED_MK_TARGET": 8,
        "TUNED_MK_MISSING_SQUARE_WEIGHT": 12,
        "TUNED_MK_IN_CHECK_SCALE_PCT": 50,
        "TUNED_CONSTRAINT_BASE": 220,
        "TUNED_CONSTRAINT_CAP": 70,
        "TUNED_CONSTRAINT_TRIGGER_MK_MAX": 1,
        "TUNED_RICHNESS_BASE_MAX": 120,
        "TUNED_RICHNESS_MIN": 20,
        "TUNED_RICHNESS_PIECES_CAP": 30,
        "TUNED_RICHNESS_PIECE_PENALTY": 3,
        "TUNED_TRADE_ATTACKER_NEG_SEE_DIV": 5,
        "TUNED_TRADE_QUEEN_PAWN_PENALTY": 20,
        "TUNED_TRADE_SMALL_VICTIM_DIV": 2,
        "TUNED_TRADE_ATTACKER_SMALL_VICTIM_DIV": 6,
        "TUNED_MOBILITY_RISKY_CAP_WEIGHT": 3,
        "TUNED_MOBILITY_SAFE_CAP_WEIGHT": 8,
        "TUNED_KING_ATTACK_OPP_NONKING_MAX": 1,
        "TUNED_KING_ATTACK_ACTIVITY_DIST_BASE": 10,
        "TUNED_KING_ATTACK_ACTIVITY_WEIGHT": 8,
        "TUNED_KING_ATTACK_ATTACKER_PULL_DIST_BASE": 14,
        "TUNED_KING_ATTACK_ATTACKER_PULL_WEIGHT": 2,
        "TUNED_KING_ATTACK_BOOSTED_MK_DIV": 2,
        "TUNED_KING_AGGRESSION_MAT_DIFF_THRESHOLD": 400,
        "TUNED_KING_AGGRESSION_PHASE_MARGIN": 2,
        "TUNED_KING_AGGRESSION_DIST_BASE": 12,
        "TUNED_KING_AGGRESSION_WEIGHT": 6,
        "TUNED_PAWN_ADVANCE_RANK_WEIGHT": 2,
        "TUNED_FIFTY_MOVE_THRESHOLD": 60,
        "TUNED_FIFTY_MOVE_PENALTY_PER_PLY": 3,
    }


def load_extra_params_json(path: str):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("--extra-params-json must contain a JSON object/dict at the top level.")
    return data


def merge_extra_params(defaults: dict, overrides: dict):
    merged = dict(defaults)
    for raw_key, raw_val in overrides.items():
        key = raw_key if raw_key.startswith("TUNED_") else f"TUNED_{raw_key}"
        if key not in defaults:
            print(f"Warning: ignoring unknown extra param: {raw_key}")
            continue
        default_val = defaults[key]
        if isinstance(default_val, list):
            if not isinstance(raw_val, list) or len(raw_val) != len(default_val):
                raise ValueError(f"{raw_key} must be a list of length {len(default_val)}.")
            merged[key] = [int(x) for x in raw_val]
        else:
            merged[key] = int(raw_val)
    return merged


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


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    base: float
    features: List[Tuple[int, float]]
    label: float  # objective-dependent (e.g., cp or +/-1)


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
    return w_out


def emit_header(w, path: str, pst_clamp: float, small_clamp: float, tempo_clamp: float, bias_clamp: float, extra_params: dict):
    if len(w) < NUM_FEATURES:
        raise ValueError(f"Weight vector too small: got {len(w)}, expected >= {NUM_FEATURES}")
    required_extras = default_extra_params()
    missing = [k for k in required_extras.keys() if k not in extra_params]
    if missing:
        raise ValueError(f"Missing extra params for header emission: {missing}")
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

    def fmt_table(table):
        lines = []
        for row in table:
            rows = []
            for r in range(0, 64, 8):
                rows.append(", ".join(f"{row[r + c]:4d}" for c in range(8)))
            lines.append("    {" + ",\n     ".join(rows) + "}")
        return ",\n".join(lines)

    def fmt_arr(values):
        return ", ".join(str(int(v)) for v in values)

    extra = extra_params
    extra_section = f"""// ---------------------------------------------------------------------------
// Extra tunables (defaults match eval.cpp)
// ---------------------------------------------------------------------------

constexpr int TUNED_MG_PIECE_VALUE[7] = {{{fmt_arr(extra["TUNED_MG_PIECE_VALUE"])}}};
constexpr int TUNED_EG_PIECE_VALUE[7] = {{{fmt_arr(extra["TUNED_EG_PIECE_VALUE"])}}};

constexpr int TUNED_CENTER_MG_WEIGHT[7] = {{{fmt_arr(extra["TUNED_CENTER_MG_WEIGHT"])}}};
constexpr int TUNED_CENTER_EG_WEIGHT[7] = {{{fmt_arr(extra["TUNED_CENTER_EG_WEIGHT"])}}};

constexpr int TUNED_KING_PROX_DIST_BASE = {int(extra["TUNED_KING_PROX_DIST_BASE"])};

constexpr int TUNED_TRADE_RISK_SCALE_PCT = {int(extra["TUNED_TRADE_RISK_SCALE_PCT"])};

constexpr int TUNED_MK_BONUS_MAX = {int(extra["TUNED_MK_BONUS_MAX"])};
constexpr int TUNED_MK_TARGET = {int(extra["TUNED_MK_TARGET"])};
constexpr int TUNED_MK_MISSING_SQUARE_WEIGHT = {int(extra["TUNED_MK_MISSING_SQUARE_WEIGHT"])};
constexpr int TUNED_MK_IN_CHECK_SCALE_PCT = {int(extra["TUNED_MK_IN_CHECK_SCALE_PCT"])};

constexpr int TUNED_CONSTRAINT_BASE = {int(extra["TUNED_CONSTRAINT_BASE"])};
constexpr int TUNED_CONSTRAINT_CAP = {int(extra["TUNED_CONSTRAINT_CAP"])};
constexpr int TUNED_CONSTRAINT_TRIGGER_MK_MAX = {int(extra["TUNED_CONSTRAINT_TRIGGER_MK_MAX"])};
constexpr int TUNED_RICHNESS_BASE_MAX = {int(extra["TUNED_RICHNESS_BASE_MAX"])};
constexpr int TUNED_RICHNESS_MIN = {int(extra["TUNED_RICHNESS_MIN"])};
constexpr int TUNED_RICHNESS_PIECES_CAP = {int(extra["TUNED_RICHNESS_PIECES_CAP"])};
constexpr int TUNED_RICHNESS_PIECE_PENALTY = {int(extra["TUNED_RICHNESS_PIECE_PENALTY"])};

constexpr int TUNED_TRADE_ATTACKER_NEG_SEE_DIV = {int(extra["TUNED_TRADE_ATTACKER_NEG_SEE_DIV"])};
constexpr int TUNED_TRADE_QUEEN_PAWN_PENALTY = {int(extra["TUNED_TRADE_QUEEN_PAWN_PENALTY"])};
constexpr int TUNED_TRADE_SMALL_VICTIM_DIV = {int(extra["TUNED_TRADE_SMALL_VICTIM_DIV"])};
constexpr int TUNED_TRADE_ATTACKER_SMALL_VICTIM_DIV = {int(extra["TUNED_TRADE_ATTACKER_SMALL_VICTIM_DIV"])};

constexpr int TUNED_MOBILITY_RISKY_CAP_WEIGHT = {int(extra["TUNED_MOBILITY_RISKY_CAP_WEIGHT"])};
constexpr int TUNED_MOBILITY_SAFE_CAP_WEIGHT = {int(extra["TUNED_MOBILITY_SAFE_CAP_WEIGHT"])};

constexpr int TUNED_KING_ATTACK_OPP_NONKING_MAX = {int(extra["TUNED_KING_ATTACK_OPP_NONKING_MAX"])};
constexpr int TUNED_KING_ATTACK_ACTIVITY_DIST_BASE = {int(extra["TUNED_KING_ATTACK_ACTIVITY_DIST_BASE"])};
constexpr int TUNED_KING_ATTACK_ACTIVITY_WEIGHT = {int(extra["TUNED_KING_ATTACK_ACTIVITY_WEIGHT"])};
constexpr int TUNED_KING_ATTACK_ATTACKER_PULL_DIST_BASE = {int(extra["TUNED_KING_ATTACK_ATTACKER_PULL_DIST_BASE"])};
constexpr int TUNED_KING_ATTACK_ATTACKER_PULL_WEIGHT = {int(extra["TUNED_KING_ATTACK_ATTACKER_PULL_WEIGHT"])};
constexpr int TUNED_KING_ATTACK_BOOSTED_MK_DIV = {int(extra["TUNED_KING_ATTACK_BOOSTED_MK_DIV"])};

constexpr int TUNED_KING_AGGRESSION_MAT_DIFF_THRESHOLD = {int(extra["TUNED_KING_AGGRESSION_MAT_DIFF_THRESHOLD"])};
constexpr int TUNED_KING_AGGRESSION_PHASE_MARGIN = {int(extra["TUNED_KING_AGGRESSION_PHASE_MARGIN"])};
constexpr int TUNED_KING_AGGRESSION_DIST_BASE = {int(extra["TUNED_KING_AGGRESSION_DIST_BASE"])};
constexpr int TUNED_KING_AGGRESSION_WEIGHT = {int(extra["TUNED_KING_AGGRESSION_WEIGHT"])};

constexpr int TUNED_PAWN_ADVANCE_RANK_WEIGHT = {int(extra["TUNED_PAWN_ADVANCE_RANK_WEIGHT"])};

constexpr int TUNED_FIFTY_MOVE_THRESHOLD = {int(extra["TUNED_FIFTY_MOVE_THRESHOLD"])};
constexpr int TUNED_FIFTY_MOVE_PENALTY_PER_PLY = {int(extra["TUNED_FIFTY_MOVE_PENALTY_PER_PLY"])};

"""

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

{extra_section}

#endif // {guard}
"""
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write(header.rstrip() + "\n")
    print(f"Wrote tuned weights to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Texel logistic tuner for PST/bishop pair/tempo.")
    p.add_argument("--input", default="build/deep_labeled_positions.csv.gz", help="Input CSV (.gz ok) from deep_labeler.")
    p.add_argument("--output-header", default="src/core/gameTreeSearch/weights_evaluated.h", help="Header path to write tuned weights.")
    p.add_argument(
        "--extra-params-json",
        default="",
        help="Optional JSON file of extra eval constants to embed into the header (keys may be with/without TUNED_ prefix).",
    )
    p.add_argument("--holdout", type=float, default=0.2, help="Fraction of data for validation.")
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
    p.add_argument("--logistic-scale", type=float, default=400.0, help="Scale for Texel logistic loss.")
    p.add_argument("--mse-scale", type=float, default=50.0, help="Normalization for cp MSE (objective=cp).")
    p.add_argument("--pst-clamp", type=float, default=200.0, help="Clamp for PST weights.")
    p.add_argument("--small-clamp", type=float, default=50.0, help="Clamp for bishop pair / king proximity.")
    p.add_argument("--tempo-clamp", type=float, default=50.0, help="Clamp for tempo bonus.")
    p.add_argument("--bias-clamp", type=float, default=50.0, help="Clamp for eval bias.")
    p.add_argument("--objective", choices=("sign", "cp"), default="cp", help="Training objective: sign-classification or cp-regression.")
    p.add_argument("--require-deep", action="store_true", help="Only use eval_deep_cp; skip rows without deep labels.")
    p.add_argument("--label-clamp", type=int, default=2000, help="Clamp labels to +/-N centipawns (0 disables).")
    return p.parse_args()


def load_samples(path: str, max_samples: int, objective: str, require_deep: bool, label_clamp: int) -> List[Sample]:
    samples: List[Sample] = []
    bad_rows = 0
    with open_maybe_gzip(path, "rt") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fen = (row.get("fen") or "").strip()
            if not fen:
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
            s = build_sample(fen, score, objective=objective, label_clamp=label_clamp)
            if s is None:
                continue
            samples.append(s)
            if max_samples and len(samples) >= max_samples:
                break
    if bad_rows:
        print(f"Skipped {bad_rows} rows due to missing/invalid fen or score.")
    return samples


def main():
    args = parse_args()
    if args.optimizer is None:
        args.optimizer = "ridge" if args.objective == "cp" else "adam"
    random.seed(args.seed)
    samples = load_samples(args.input, args.max_samples, args.objective, args.require_deep, args.label_clamp)
    if not samples:
        print("No samples loaded; aborting.")
        return

    random.shuffle(samples)
    split = int(len(samples) * (1.0 - args.holdout))
    train_samples = samples[:split]
    val_samples = samples[split:]

    print(f"Loaded {len(samples)} samples (train {len(train_samples)}, val {len(val_samples)}).")
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

    extra_params = merge_extra_params(default_extra_params(), load_extra_params_json(args.extra_params_json))
    emit_header(w_emitted, args.output_header, args.pst_clamp, args.small_clamp, args.tempo_clamp, args.bias_clamp, extra_params)


if __name__ == "__main__":
    main()
