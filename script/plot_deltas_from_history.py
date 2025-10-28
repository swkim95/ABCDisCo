#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot δ_i (i=A,B,C,D) and normalized signal contamination δ_rel vs epoch
from an existing training history.json, without retraining.

Usage (HPC hint): run `setup` in your shell first to load the ABCDisCo env.

Example:
  python script/plot_deltas_from_history.py \
      --history_json runs/smear25/history/history.json

Outputs will be written under: runs/<label>/plots/history/from_history/
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot δ_i and δ_rel vs epoch from saved history.json")
    p.add_argument("--history_json", required=True, help="Path to history.json produced by training")
    p.add_argument("--splits", default="train,val", help="Comma-separated splits to plot (default: train,val)")
    p.add_argument("--wps", default="10,30,50", help="Comma-separated working points in % (default: 10,30,50)")
    p.add_argument("--regions", default="A,B,C,D", help="Comma-separated regions (default: A,B,C,D)")
    p.add_argument("--out_root", default=None, help="Optional output root dir; defaults to <run>/plots/history/from_history")
    return p.parse_args()


def load_history(history_json: Path) -> pd.DataFrame:
    with open(history_json, "r") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or len(rows) == 0:
        raise ValueError(f"history.json at {history_json} is empty or malformed")
    return pd.DataFrame(rows)


def default_out_root(history_json: Path) -> Path:
    # history_json is expected at runs/<label>/history/history.json
    # Place outputs under runs/<label>/plots/history/from_history
    run_root = history_json.parent.parent  # .../runs/<label>
    out_root = run_root / "plots" / "history" / "from_history"
    ensure_dir(out_root)
    return out_root


def plot_delta_rel_vs_epoch(df: pd.DataFrame, splits: List[str], wps: List[str], out_dir: Path) -> None:
    ensure_dir(out_dir)
    x = df.get("epoch", None)
    if x is None:
        raise KeyError("'epoch' not found in history DataFrame")
    x = df["epoch"].values

    # Linear scale
    plt.figure(figsize=(8, 5))
    for wp, style in [("10", "--"), ("30", "-"), ("50", ":")]:
        if wp not in wps:
            continue
        for split, color in [("train", "C0"), ("val", "C1")]:
            if split not in splits:
                continue
            key = f"{split}_delta_rel_{wp}"
            y = df.get(key, None)
            if y is None:
                continue
            plt.plot(x, df[key].values, color=color, linestyle=style, label=f"{split} δ_rel ({wp}%)")
    plt.xlabel("Epoch")
    plt.ylabel("δ_rel")
    plt.title("Normalized signal contamination δ_rel vs epoch")
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "delta_rel_vs_epoch.png")
    plt.yscale("log")
    plt.savefig(out_dir / "delta_rel_vs_epoch_log.png")
    plt.close()


def plot_delta_region_vs_epoch(df: pd.DataFrame, splits: List[str], wps: List[str], region: str, out_dir: Path) -> None:
    ensure_dir(out_dir)
    x = df.get("epoch", None)
    if x is None:
        raise KeyError("'epoch' not found in history DataFrame")
    x = df["epoch"].values
    key_base = {
        "A": "deltaA",
        "B": "deltaB",
        "C": "deltaC",
        "D": "deltaD",
    }.get(region)
    if key_base is None:
        raise ValueError(f"Unknown region '{region}' (expected A,B,C,D)")

    plt.figure(figsize=(8, 5))
    for wp, style in [("10", "--"), ("30", "-"), ("50", ":")]:
        if wp not in wps:
            continue
        for split, color in [("train", "C0"), ("val", "C1")]:
            if split not in splits:
                continue
            key = f"{split}_{key_base}_{wp}"
            y = df.get(key, None)
            if y is None:
                continue
            plt.plot(x, df[key].values, color=color, linestyle=style, label=f"{split} δ_{region} ({wp}%)")
    plt.xlabel("Epoch")
    plt.ylabel(f"δ_{region}")
    plt.title(f"Signal contamination δ_{region} vs epoch")
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"delta_{region}_vs_epoch.png")
    plt.yscale("log")
    plt.savefig(out_dir / f"delta_{region}_vs_epoch_log.png")
    plt.close()


def main() -> None:
    args = parse_args()
    history_path = Path(args.history_json).resolve()
    if not history_path.exists():
        raise FileNotFoundError(f"history.json not found at: {history_path}")
    df = load_history(history_path)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    wps = [w.strip() for w in args.wps.split(",") if w.strip()]
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]

    if args.out_root is not None:
        out_root = Path(args.out_root)
    else:
        out_root = default_out_root(history_path)
    ensure_dir(out_root)

    # δ_rel plots
    plot_delta_rel_vs_epoch(df, splits, wps, out_root)

    # Per-region δ plots
    for region in regions:
        plot_delta_region_vs_epoch(df, splits, wps, region, out_root)

    print(f"Wrote plots to: {out_root}")


if __name__ == "__main__":
    main()


