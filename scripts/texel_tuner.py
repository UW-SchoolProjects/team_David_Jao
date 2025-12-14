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
- Uses sign(labels) as targets (cp from side-to-move POV). Zero/blank labels are skipped.
- Handles positions even if there are no legal moves (purely parses FEN, no move gen).
- Clamps PST weights to +/-200, small terms to +/-50 before emission.
"""

import argparse
import csv
import gzip
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
# Utility helpers
# ---------------------------------------------------------------------------

def open_maybe_gzip(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode, newline="", encoding="utf-8", errors="replace")
    return open(path, mode, newline="", encoding="utf-8", errors="replace")


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
    squares = [None] * 64
    rank = 7
    file = 0
    for ch in placement:
        if ch == "/":
            rank -= 1
            file = 0
            continue
        if ch.isdigit():
            file += int(ch)
            continue
        if file >= 8 or rank < 0:
            return None, None
        squares[rank * 8 + file] = ch
        file += 1
    return squares, side


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    base: float
    features: List[Tuple[int, float]]
    label: int  # +1 or -1


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def build_sample(fen: str, label_cp: int) -> Optional[Sample]:
    squares, side = parse_fen(fen)
    if squares is None or side not in ("w", "b"):
        return None
    if label_cp == 0:
        return None

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
            scaled_features.append((IDX_KING_PROX, prox * eg_wt))

    # Bias term
    scaled_features.append((IDX_BIAS, 1.0))

    label = 1 if label_cp > 0 else -1
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
    w = [0.0] * NUM_FEATURES
    lr = args.lr
    batch_size = args.batch_size
    l2 = args.l2
    scale = args.logistic_scale

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
            inv_bs = 1.0 / max(1, len(batch))
            for i in range(NUM_FEATURES):
                grad[i] = grad[i] * inv_bs + l2 * w[i]
                w[i] -= lr * grad[i]

        avg_loss = total_loss / max(1, len(samples))
        print(f"Epoch {epoch+1}/{args.epochs}: avg_loss={avg_loss:.4f}")
    return w


def predict_sign(sample: Sample, w) -> int:
    pred = sample.base
    for idx, val in sample.features:
        pred += w[idx] * val
    return 1 if pred >= 0 else -1


def accuracy(samples: List[Sample], w) -> float:
    if not samples:
        return 0.0
    correct = sum(1 for s in samples if predict_sign(s, w) == s.label)
    return correct / len(samples)


# ---------------------------------------------------------------------------
# Header emission
# ---------------------------------------------------------------------------

def clamp_and_round(value: float, clamp: float) -> int:
    return int(round(max(-clamp, min(clamp, value))))


def emit_header(w, path: str, pst_clamp: float, small_clamp: float):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    mg = [[0] * 64 for _ in PIECE_ORDER]
    eg = [[0] * 64 for _ in PIECE_ORDER]

    for pt_idx, _ in enumerate(PIECE_ORDER):
        for sq in range(64):
            mg_idx = IDX_MG_START + pt_idx * 64 + sq
            eg_idx = IDX_EG_START + pt_idx * 64 + sq
            mg[pt_idx][sq] = clamp_and_round(w[mg_idx], pst_clamp)
            eg[pt_idx][sq] = clamp_and_round(w[eg_idx], pst_clamp)

    bp = clamp_and_round(w[IDX_BISHOP_PAIR], small_clamp)
    tempo = clamp_and_round(w[IDX_TEMPO], small_clamp)
    king_prox = clamp_and_round(w[IDX_KING_PROX], small_clamp)
    bias = clamp_and_round(w[IDX_BIAS], small_clamp)

    def fmt_table(table):
        lines = []
        for row in table:
            rows = []
            for r in range(0, 64, 8):
                rows.append(", ".join(f"{row[r + c]:4d}" for c in range(8)))
            lines.append("    {" + ",\n     ".join(rows) + "}")
        return ",\n".join(lines)

    header = f"""// Auto-generated by scripts/texel_tuner.py
// Do not edit by hand. Values are from Texel logistic regression.

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
"""
    with open(path, "w", encoding="ascii") as fh:
        fh.write(header)
    print(f"Wrote tuned weights to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Texel logistic tuner for PST/bishop pair/tempo.")
    p.add_argument("--input", default="build/deep_labeled_positions.csv.gz", help="Input CSV (.gz ok) from deep_labeler.")
    p.add_argument("--output-header", default="src/core/gameTreeSearch/weights_evaluated.h", help="Header path to write tuned weights.")
    p.add_argument("--holdout", type=float, default=0.2, help="Fraction of data for validation.")
    p.add_argument("--max-samples", type=int, default=0, help="Limit number of samples (0 = all).")
    p.add_argument("--seed", type=int, default=13, help="RNG seed.")
    p.add_argument("--epochs", type=int, default=8, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=512, help="Batch size for SGD.")
    p.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    p.add_argument("--l2", type=float, default=1e-4, help="L2 weight decay.")
    p.add_argument("--logistic-scale", type=float, default=400.0, help="Scale for Texel logistic loss.")
    p.add_argument("--pst-clamp", type=float, default=200.0, help="Clamp for PST weights.")
    p.add_argument("--small-clamp", type=float, default=50.0, help="Clamp for bishop pair/tempo/etc.")
    return p.parse_args()


def load_samples(path: str, max_samples: int) -> List[Sample]:
    samples: List[Sample] = []
    with open_maybe_gzip(path, "rt") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fen = (row.get("fen") or "").strip()
            if not fen:
                continue
            try:
                score = int(row.get("eval_deep_cp") or row.get("eval_cp") or 0)
            except ValueError:
                continue
            s = build_sample(fen, score)
            if s is None:
                continue
            samples.append(s)
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def main():
    args = parse_args()
    random.seed(args.seed)
    samples = load_samples(args.input, args.max_samples)
    if not samples:
        print("No samples loaded; aborting.")
        return

    random.shuffle(samples)
    split = int(len(samples) * (1.0 - args.holdout))
    train_samples = samples[:split]
    val_samples = samples[split:]

    print(f"Loaded {len(samples)} samples (train {len(train_samples)}, val {len(val_samples)}).")
    w = train(train_samples, args)
    train_acc = accuracy(train_samples, w)
    val_acc = accuracy(val_samples, w)
    print(f"Texel sign accuracy: train={train_acc*100:.2f}% val={val_acc*100:.2f}%")

    emit_header(w, args.output_header, args.pst_clamp, args.small_clamp)


if __name__ == "__main__":
    main()
