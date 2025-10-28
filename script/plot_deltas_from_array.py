#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute and plot δ_A/B/C/D and δ_rel vs epoch from existing history.json
and per-epoch NPZ arrays (scores, labels, weights, masses) — no retraining.

Usage:
  setup
  python script/plot_deltas_from_arrays.py --run_root runs/<run_label>

Outputs:
  runs/<run_label>/plots/history/from_arrays/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot δ_i and δ_rel vs epoch from saved arrays + history")
    p.add_argument("--run_root", required=True, help="Path to runs/<run_label>")
    p.add_argument("--splits", default="train,val", help="Comma-separated splits to plot (default: train,val)")
    p.add_argument("--wps", default="10,30,50", help="Comma-separated working points in % (default: 10,30,50)")
    p.add_argument("--m_top", type=float, default=172.5, help="Top mass used in m_hat (default: 172.5)")
    return p.parse_args()


def load_history(run_root: Path) -> List[Dict]:
    hist_path = run_root / "history" / "history.json"
    if not hist_path.exists():
        raise FileNotFoundError(f"history.json not found at {hist_path}")
    with open(hist_path, "r") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError(f"history.json at {hist_path} is empty or malformed")
    return rows


def list_npz_for_split(run_root: Path, split: str) -> List[Tuple[int, Path]]:
    """Return sorted list of (epoch, path) for given split."""
    arr_dir = run_root / "history" / "arrays"
    if not arr_dir.exists():
        return []
    files = []
    for p in sorted(arr_dir.glob(f"{split}_epoch_*.npz")):
        # Expect name like split_epoch_123.npz
        try:
            epoch = int(p.stem.split("_")[-1])
        except Exception:
            continue
        files.append((epoch, p))
    return sorted(files, key=lambda x: x[0])


def compute_mhat(masses: np.ndarray, m_top: float) -> np.ndarray:
    masses = masses.astype(np.float64)
    return 1.0 - np.abs(masses - m_top) / m_top


def compute_deltas(scores: np.ndarray,
                   labels: np.ndarray,
                   weights: np.ndarray,
                   mhat: np.ndarray,
                   cut: float,
                   mhat_thr: float) -> Dict[str, float]:
    """Compute δ_A/B/C/D and δ_rel with weighted counts."""
    eps = 1e-12
    pass_score = scores >= cut
    in_sr = mhat >= mhat_thr

    A = pass_score & in_sr
    B = pass_score & (~in_sr)
    C = (~pass_score) & in_sr
    D = (~pass_score) & (~in_sr)

    s_mask = labels > 0.5
    b_mask = ~s_mask
    w = weights.astype(float)

    def sums(mask):
        return float(np.sum(w[mask & s_mask])), float(np.sum(w[mask & b_mask]))

    A_s, A_b = sums(A)
    B_s, B_b = sums(B)
    C_s, C_b = sums(C)
    D_s, D_b = sums(D)

    delta_A = A_s / (A_b + eps)
    delta_B = B_s / (B_b + eps)
    delta_C = C_s / (C_b + eps)
    delta_D = D_s / (D_b + eps)
    delta_rel = (delta_B + delta_C - delta_D) / (delta_A + eps)
    return {"deltaA": delta_A, "deltaB": delta_B, "deltaC": delta_C, "deltaD": delta_D, "delta_rel": delta_rel}


def collect_series(run_root: Path,
                   history_rows: List[Dict],
                   split: str,
                   wps: List[str],
                   m_top: float) -> Dict[str, List[Tuple[int, float]]]:
    """Return time series per metric key for this split and WPs."""
    # Index history by epoch for quick lookup
    by_epoch: Dict[int, Dict] = {int(r["epoch"]): r for r in history_rows if "epoch" in r}

    files = list_npz_for_split(run_root, split)
    if not files:
        raise FileNotFoundError(f"No NPZ arrays found for split='{split}' under {run_root}/history/arrays")

    series: Dict[str, List[Tuple[int, float]]] = {}
    for epoch, npz_path in files:
        r = by_epoch.get(epoch)
        if r is None:
            continue
        data = np.load(npz_path)
        scores = data["scores"].astype(np.float64)
        labels = data["labels"].astype(np.float64)
        weights = data["weights"].astype(np.float64)
        masses = data["masses"].astype(np.float64)
        mhat = compute_mhat(masses, m_top)

        # m_hat threshold from history; fallback to constant (if missing)
        mhat_thr = r.get(f"{split}_mhat_threshold", None)
        if mhat_thr is None or not np.isfinite(mhat_thr):
            # Fallback to the code's fixed choice (delta_m=35 GeV)
            mhat_thr = 1.0 - (35.0 / m_top)

        for wp in wps:
            cut = r.get(f"{split}_score_cut_wp{wp}", None)
            if cut is None or not np.isfinite(cut):
                # If cut is missing, skip this WP at this epoch
                continue
            d = compute_deltas(scores, labels, weights, mhat, float(cut), float(mhat_thr))
            for k, v in d.items():
                key = f"{split}_{k}_{wp}"
                series.setdefault(key, []).append((epoch, v))
    # Sort by epoch
    for k in series:
        series[k] = sorted(series[k], key=lambda x: x[0])
    return series


def plot_series(series: Dict[str, List[Tuple[int, float]]], out_dir: Path, title_prefix: str) -> None:
    ensure_dir(out_dir)
    # δ_rel vs epoch
    plt.figure(figsize=(8, 5))
    for wp, style in [("10", "--"), ("30", "-"), ("50", ":")]:
        for split, color in [("train", "C0"), ("val", "C1")]:
            key = f"{split}_delta_rel_{wp}"
            if key not in series:
                continue
            xs, ys = zip(*series[key])
            plt.plot(xs, ys, linestyle=style, color=color, label=f"{split} δ_rel ({wp}%)")
    plt.xlabel("Epoch"); plt.ylabel("δ_rel"); plt.title(f"{title_prefix}: δ_rel vs epoch")
    plt.legend(ncol=2); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / "delta_rel_vs_epoch.png"); plt.yscale("log")
    plt.savefig(out_dir / "delta_rel_vs_epoch_log.png"); plt.close()

    # δ_A/B/C/D vs epoch (one canvas per region)
    for region, base in [("A", "deltaA"), ("B", "deltaB"), ("C", "deltaC"), ("D", "deltaD")]:
        plt.figure(figsize=(8, 5))
        for wp, style in [("10", "--"), ("30", "-"), ("50", ":")]:
            for split, color in [("train", "C0"), ("val", "C1")]:
                key = f"{split}_{base}_{wp}"
                if key not in series:
                    continue
                xs, ys = zip(*series[key])
                plt.plot(xs, ys, linestyle=style, color=color, label=f"{split} δ_{region} ({wp}%)")
        plt.xlabel("Epoch"); plt.ylabel(f"δ_{region}"); plt.title(f"{title_prefix}: δ_{region} vs epoch")
        plt.legend(ncol=2); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(out_dir / f"delta_{region}_vs_epoch.png"); plt.yscale("log")
        plt.savefig(out_dir / f"delta_{region}_vs_epoch_log.png"); plt.close()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    history_rows = load_history(run_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    wps = [w.strip() for w in args.wps.split(",") if w.strip()]

    series_all: Dict[str, List[Tuple[int, float]]] = {}
    for split in splits:
        series_split = collect_series(run_root, history_rows, split, wps, m_top=float(args.m_top))
        for k, v in series_split.items():
            series_all[k] = v

    out_dir = run_root / "plots" / "history" / "from_arrays"
    plot_series(series_all, out_dir, title_prefix=f"{run_root.name}")
    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()
    