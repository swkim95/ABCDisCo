#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Double-DisCo QNN training script (two HybridQNN heads) with full evaluation.

This follows the logic and settings of:
- DNN double-DisCo: doubleDisCo_script/train_abcd_double.py (ABCD pair scanning, f/g axes, JSDvsR using outputs)
- QNN single-DisCo: QNN_script/train_QNN_abcd_single.py (HybridQNN head, QNN CLI knobs and dataloader settings)

Key features:
- Two separate HybridQNN classifiers trained jointly on the same 13 HLF inputs
  (f: x-axis, g: y-axis for ABCD). Each produces its own logits and score.
- Joint objective: BCE(f) + BCE(g) + alpha * dCorr_unbiased(f_bg, g_bg)
  computed on background only with weights normalized to sample size.
- 2D ABCD evaluation at εS = 10/30/50% by scanning (cut_f, cut_g) pairs and
  selecting the pair with closure closest to 1 (tie-break by higher 1/ε_B).
- JSDvsR computed using the two outputs (threshold on f as score axis; g as
  mass-like variable); no use of m_hat in double-DisCo evaluation.
- Per-model (f and g) metrics and plots; scatter f vs g with WP lines; histogram
  overlays with estimated background; history logging and plots for both heads.
- Per-epoch NPZ exports with logits/scores for both heads for reproducibility.

Reference: T. Aarrestad et al., "ABCDisCo: Automating the ABCD Method with
Machine Learning", Eur. Phys. J. C 81, 256 (2021), arXiv:2007.14400.
"""

##### Imports and CLI arguments ###############################################################

import os
import sys
import time
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless plotting for batch environments
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

# Local modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from data_loader import TopTaggingDataset
from QNN_networks import HybridQNN
from evaluation import JSDvsR, weighted_quantile
from disco import distance_corr_unbiased


# Human-readable names for the 13 NN input features (scaled)
# Original 13 observables in file columns 1: (after label at col 0):
# [mass, pt, tau1_half, tau2_half, tau3_half, tau1, tau2, tau3, tau4,
#  tau1_sq, tau2_sq, tau3_sq, tau4_sq]
FEATURE_NAMES_13: List[str] = [
    "mass",
    "pt",
    "tau1_half",
    "tau2_half",
    "tau3_half",
    "tau1",
    "tau2",
    "tau3",
    "tau4",
    "tau1_sq",
    "tau2_sq",
    "tau3_sq",
    "tau4_sq",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Double-DisCo QNN training (two HybridQNN heads on 13 features) with full diagnostics"
    )
    parser.add_argument("--gpunum", default="0", help="GPU index passed to CUDA_VISIBLE_DEVICES (default: 0)")
    parser.add_argument("--logfile", default="log.csv", help="CSV log file for legacy state rows (as in original)")
    parser.add_argument("--smear", default="25", help="Gaussian mass smearing sigma [GeV] (default: 25)")
    parser.add_argument("--run_label", default=None, help="Optional label to name run outputs (e.g. smear25)")
    parser.add_argument("--resume", default=None, help="Path to checkpoint .pth to resume from (model+optimizer+history)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default: 200)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size (default: 10000)")
    parser.add_argument("--alpha", type=float, default=100.0, help="Double-DisCo penalty coefficient (default: 100)")
    parser.add_argument(
        "--max_points_scatter", type=int, default=100000,
        help="Max points used in scatter plots to avoid memory blowup (default: 100k)"
    )
    parser.add_argument(
        "--output_root", default="runs", help="Root directory for outputs (checkpoints/plots/history)"
    )
    parser.add_argument(
        "--save_epoch_arrays", action="store_true", default=True,
        help="Save per-epoch arrays (scores_f/g, logits_f/g, labels, weights, masses, features) to NPZ"
    )
    parser.add_argument("--test", action="store_true", default=False, help="Test mode: train using small dataset")
    # QNN-specific knobs (adopted from single-QNN script)
    parser.add_argument("--qnn-n-qubits", type=int, default=6, help="Number of qubits / angles for VQC")
    parser.add_argument("--qnn-hidden-dim", type=int, default=64, help="Hidden dimension of front FCN")
    parser.add_argument("--qnn-vqc-depth", type=int, default=4, help="Number of StronglyEntanglingLayers")
    parser.add_argument("--qnn-n-outputs", type=int, default=2, help="Number of output logits")
    parser.add_argument("--qnn-device", type=str, default="default.qubit", help="PennyLane device (e.g., default.qubit, lightning.qubit, lightning.gpu)")
    # Dataloader workers for HPC
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (0=main thread)")
    return parser


##### Utility helpers #########################################################################

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_device(gpu_index_str: str) -> torch.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index_str
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def save_json(obj: Any, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def to_numpy_detached(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def softmax_class1(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1)[:, 1]


def weighted_confusion_stats(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (scores >= threshold).astype(float)
    w = weights.astype(float)
    tp = float(np.sum(w * (preds == 1) * (labels == 1)))
    tn = float(np.sum(w * (preds == 0) * (labels == 0)))
    fp = float(np.sum(w * (preds == 1) * (labels == 0)))
    fn = float(np.sum(w * (preds == 0) * (labels == 1)))
    total = tp + tn + fp + fn + 1e-12
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def compute_auc(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    s_mask = labels > 0.5
    b_mask = ~s_mask
    s_scores = scores[s_mask]
    b_scores = scores[b_mask]
    s_w = weights[s_mask]
    b_w = weights[b_mask]
    if s_scores.size == 0 or b_scores.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([s_scores, b_scores]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, order.size + 1)
    s_w_sum = float(np.sum(s_w))
    b_w_sum = float(np.sum(b_w))
    if s_w_sum <= 0 or b_w_sum <= 0:
        return float("nan")
    s_ranks = ranks[: s_scores.size]
    auc = (np.sum(s_w * s_ranks) - s_w_sum * (s_w_sum + 1) / 2.0) / (s_w_sum * b_w_sum)
    return float(auc)


def weighted_threshold_at_signal_eff(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray, target_eff: float) -> float:
    s_mask = labels > 0.5
    if np.sum(weights[s_mask]) <= 0:
        return float(np.nan)
    return float(weighted_quantile(scores[s_mask], 1.0 - target_eff, sample_weight=weights[s_mask]))


def compute_bg_eff_rej_contamination(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray, cut: float) -> Dict[str, float]:
    mask_pass = scores >= cut
    s_mask = labels > 0.5
    b_mask = ~s_mask
    w = weights.astype(float)
    s_pass = float(np.sum(w[mask_pass & s_mask]))
    b_pass = float(np.sum(w[mask_pass & b_mask]))
    s_tot = float(np.sum(w[s_mask])) + 1e-12
    b_tot = float(np.sum(w[b_mask])) + 1e-12
    b_eff = b_pass / b_tot
    b_rej = 1.0 / max(b_eff, 1e-12)
    contamination = b_pass / max(b_pass + s_pass, 1e-12)
    return {"b_eff": b_eff, "b_rej": b_rej, "sig_contamination": contamination}


##### Output directory layout #################################################################

def prepare_output_dirs(output_root: Path, run_label: str) -> Dict[str, Path]:
    base = output_root / run_label
    dirs = {
        "base": base,
        "checkpoints": base / "checkpoints",
        "plots_classification": base / "plots" / "classification",
        "plots_abcd": base / "plots" / "abcd",
        "plots_features": base / "plots" / "features",
        "plots_history": base / "plots" / "history",
        "history": base / "history",
        "history_arrays": base / "history" / "arrays",
    }
    for p in dirs.values():
        ensure_dir(p)
    return dirs


##### Data loading & preprocessing ############################################################

def load_and_preprocess(smear_sigma: float, test: bool = False) -> Dict[str, Any]:
    """Load train/val/test from .dat files, smear mass, global min–max scale, split.

    - Files: topsample_train_tau.dat, topsample_val_tau.dat, topsample_test_tau.dat
    - Skip header rows (15) matching original script; delimiter ','
    - Smear mass (column index 1) by N(0, smear_sigma)
    - Global min–max scaling applied to all 13 observables (columns 1:)
    - Features used for the classifier are all 13 scaled observables (no removal)
    - Dataset items also carry labels, unit weights, dummy binnums, and unscaled smeared mass
    """
    data_train = PROJECT_ROOT / "topsample_train_tau.dat"
    data_val = PROJECT_ROOT / "topsample_val_tau.dat"
    data_test = PROJECT_ROOT / "topsample_test_tau.dat"
    if not data_train.exists() or not data_val.exists() or not data_test.exists():
        raise FileNotFoundError(
            f"Could not find dataset files at project root: {data_train}, {data_val}, {data_test}. "
            f"Please ensure you run from the repo and that the .dat files are present."
        )
    train_raw = np.loadtxt(str(data_train), delimiter=",", skiprows=15)
    val_raw = np.loadtxt(str(data_val), delimiter=",", skiprows=15)
    test_raw = np.loadtxt(str(data_test), delimiter=",", skiprows=15)

    all_raw = np.concatenate((train_raw, val_raw, test_raw), axis=0)

    # Apply Gaussian smearing to the mass (column index 1)
    rng = np.random.default_rng()
    all_raw[:, 1] = all_raw[:, 1] + smear_sigma * rng.standard_normal(size=all_raw.shape[0])

    # Global min–max scaling across 13 observables (cols 1:)
    obs = all_raw[:, 1:]
    obs_min = np.min(obs, axis=0)
    obs_max = np.max(obs, axis=0)
    obs_scaled = (obs - obs_min) / (obs_max - obs_min + 1e-12)

    labels = all_raw[:, 0].reshape((-1, 1))
    weights = np.ones((all_raw.shape[0], 1), dtype=np.float64)
    binnums = np.ones((all_raw.shape[0], 1), dtype=np.float64)
    masses_column = all_raw[:, 1].reshape((-1, 1)).astype(np.float64)

    stacked = np.hstack(
        (
            obs_scaled.astype(np.float32),
            labels.astype(np.float32),
            weights.astype(np.float32),
            binnums.astype(np.float32),
            masses_column.astype(np.float32),
        )
    )
    stacked_t = torch.from_numpy(stacked.astype("float32"))

    # Split sizes
    Ntrain = 20000 if test else 200000
    Nval = 20000 if test else 900000
    Ntest = 20000 if test else 900000

    traindata = stacked_t[:Ntrain]
    valdata = stacked_t[Ntrain : (Ntrain + Nval)]
    testdata = stacked_t[(Ntrain + Nval) : (Ntrain + Nval + Ntest)]

    # Use ALL 13 features (:-4)
    trainset = TopTaggingDataset(traindata[:, :-4], traindata[:, -4], traindata[:, -3], traindata[:, -2], traindata[:, -1])
    valset = TopTaggingDataset(valdata[:, :-4], valdata[:, -4], valdata[:, -3], valdata[:, -2], valdata[:, -1])
    testset = TopTaggingDataset(testdata[:, :-4], testdata[:, -4], testdata[:, -3], testdata[:, -2], testdata[:, -1])

    # Numpy copies for summaries/plots
    train_features_np = to_numpy_detached(traindata[:, :-4])
    val_features_np = to_numpy_detached(valdata[:, :-4])
    test_features_np = to_numpy_detached(testdata[:, :-4])
    train_labels_np = to_numpy_detached(traindata[:, -4])
    val_labels_np = to_numpy_detached(valdata[:, -4])
    test_labels_np = to_numpy_detached(testdata[:, -4])

    return {
        "trainset": trainset,
        "valset": valset,
        "testset": testset,
        "train_features_np": train_features_np,
        "val_features_np": val_features_np,
        "test_features_np": test_features_np,
        "train_labels_np": train_labels_np,
        "val_labels_np": val_labels_np,
        "test_labels_np": test_labels_np,
    }


def print_dataset_summary_split(name: str, x: np.ndarray, y: np.ndarray, out_dir: Path, num_rows_preview: int = 10) -> None:
    total = x.shape[0]
    cls0 = int(np.sum(y < 0.5))
    cls1 = int(np.sum(y > 0.5))
    print(f"[{name}] events={total} | signal={cls1} | background={cls0}")
    lines: List[str] = []
    lines.append(f"Split: {name}\n")
    lines.append(f"Total events: {total}\n")
    lines.append(f"Signal: {cls1} | Background: {cls0}\n")
    k = min(5, x.shape[1])
    mins = np.min(x, axis=0)
    maxs = np.max(x, axis=0)
    lines.append("Feature ranges (first 5):\n")
    for i in range(k):
        lines.append(f"  f{i}: min={mins[i]:.4f}, max={maxs[i]:.4f}\n")
    s_idx = np.where(y > 0.5)[0][:num_rows_preview]
    b_idx = np.where(y < 0.5)[0][:num_rows_preview]
    lines.append(f"First {len(s_idx)} signal feature rows:\n")
    for idx in s_idx:
        lines.append("  " + ", ".join(f"{v:.6f}" for v in x[idx, :k]) + "\n")
    lines.append(f"\nFirst {len(b_idx)} background feature rows:\n")
    for idx in b_idx:
        lines.append("  " + ", ".join(f"{v:.6f}" for v in x[idx, :k]) + "\n")
    out_path = out_dir / f"summary_{name}.txt"
    with open(out_path, "w") as f:
        f.writelines(lines)


def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def plot_feature_distributions_once(train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray,
                                    train_y: np.ndarray, val_y: np.ndarray, test_y: np.ndarray,
                                    feature_names: List[str], out_dir: Path) -> None:
    ensure_dir(out_dir)
    X = np.concatenate([train_x, val_x, test_x], axis=0)
    Y = np.concatenate([train_y, val_y, test_y], axis=0)
    nfeat = X.shape[1]
    for i in range(nfeat):
        fname = sanitize_name(feature_names[i]) if i < len(feature_names) else f"feature{i}"
        plt.figure(figsize=(8, 5))
        plt.hist(X[Y < 0.5, i], bins=80, alpha=0.5, label="background", density=False)
        plt.hist(X[Y > 0.5, i], bins=80, alpha=0.5, label="signal", density=False)
        xlabel = feature_names[i] if i < len(feature_names) else f"feature[{i}]"
        plt.xlabel(f"{xlabel} (scaled)")
        plt.ylabel("events")
        plt.title(f"Input feature distribution: {xlabel} (full dataset)")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / f"feature_{fname}_hist_full.png")
        plt.close()


##### Models, loss, and training/evaluation routines #########################################

class DoubleQNNModel(nn.Module):
    """Two HybridQNN heads (f and g) sharing the same 13 inputs."""
    def __init__(self,
                 n_features: int = 13,
                 n_classes: int = 2,
                 qnn_n_qubits: int = 6,
                 qnn_hidden_dim: int = 64,
                 qnn_vqc_depth: int = 4,
                 qnn_n_outputs: int = 2,
                 qnn_device: str = "default.qubit") -> None:
        super().__init__()
        self.f_head = HybridQNN(
            n_features=int(n_features), hidden_dim=int(qnn_hidden_dim),
            n_qubits=int(qnn_n_qubits), vqc_depth=int(qnn_vqc_depth),
            n_outputs=int(qnn_n_outputs), qdevice=str(qnn_device)
        )
        self.g_head = HybridQNN(
            n_features=int(n_features), hidden_dim=int(qnn_hidden_dim),
            n_qubits=int(qnn_n_qubits), vqc_depth=int(qnn_vqc_depth),
            n_outputs=int(qnn_n_outputs), qdevice=str(qnn_device)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_f = self.f_head(x)
        logits_g = self.g_head(x)
        score_f = softmax_class1(logits_f)
        score_g = softmax_class1(logits_g)
        return logits_f, score_f, logits_g, score_g


def compute_losses_for_batch_double(model: nn.Module, batch: Tuple[torch.Tensor, ...], alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """BCE losses for f and g + unbiased distance-correlation between f and g on background.

    Returns: (loss_total, loss_bce_f, loss_bce_g, loss_disco_fg, score_f, score_g)
    """
    features, labels, weights, _binnums, _masses = batch
    logits_f, score_f, logits_g, score_g = model(features)
    loss_bce_f = F.binary_cross_entropy(score_f, labels, weight=weights)
    loss_bce_g = F.binary_cross_entropy(score_g, labels, weight=weights)

    loss_disco = torch.tensor(0.0, device=features.device, dtype=features.dtype)
    b_mask = labels < 0.5
    if torch.any(b_mask):
        s_f_b = score_f[b_mask]
        s_g_b = score_g[b_mask]
        w_b = weights[b_mask]
        if s_f_b.numel() > 2 and s_g_b.numel() > 2:
            w_norm = w_b / (torch.sum(w_b) + 1e-12) * float(len(w_b))
            loss_disco = distance_corr_unbiased(s_f_b, s_g_b, w_norm, power=1)

    loss_total = loss_bce_f + loss_bce_g + alpha * loss_disco
    return loss_total, loss_bce_f.detach(), loss_bce_g.detach(), loss_disco.detach(), score_f.detach(), score_g.detach()


@torch.no_grad()
def collect_epoch_arrays_double(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    logits_f_all: List[np.ndarray] = []
    scores_f_all: List[np.ndarray] = []
    logits_g_all: List[np.ndarray] = []
    scores_g_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    weights_all: List[np.ndarray] = []
    masses_all: List[np.ndarray] = []
    features_all: List[np.ndarray] = []
    for batch in tqdm(loader, leave=False, desc="collect"):
        x, y, w, _b, m = batch
        x_cpu = x
        x = x.to(device)
        logits_f, score_f, logits_g, score_g = model(x)
        logits_f_all.append(to_numpy_detached(logits_f))
        scores_f_all.append(to_numpy_detached(score_f))
        logits_g_all.append(to_numpy_detached(logits_g))
        scores_g_all.append(to_numpy_detached(score_g))
        labels_all.append(to_numpy_detached(y))
        weights_all.append(to_numpy_detached(w))
        masses_all.append(to_numpy_detached(m))
        features_all.append(to_numpy_detached(x_cpu))
    return {
        "logits_f": np.concatenate(logits_f_all, axis=0),
        "scores_f": np.concatenate(scores_f_all, axis=0),
        "logits_g": np.concatenate(logits_g_all, axis=0),
        "scores_g": np.concatenate(scores_g_all, axis=0),
        "labels": np.concatenate(labels_all, axis=0),
        "weights": np.concatenate(weights_all, axis=0),
        "masses": np.concatenate(masses_all, axis=0),
        "features": np.concatenate(features_all, axis=0),
    }


def run_training_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, alpha: float) -> Dict[str, float]:
    model.train()
    start = time.perf_counter()
    running_bce_f: List[float] = []
    running_bce_g: List[float] = []
    running_disco: List[float] = []
    for batch in tqdm(loader, leave=False, desc="train"):
        features, labels, weights, binnums, masses = batch
        features = features.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        binnums = binnums.to(device)
        masses = masses.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss_total, loss_bce_f, loss_bce_g, loss_disco, _sf, _sg = compute_losses_for_batch_double(
            model, (features, labels, weights, binnums, masses), alpha
        )
        loss_total.backward()
        optimizer.step()
        running_bce_f.append(float(loss_bce_f))
        running_bce_g.append(float(loss_bce_g))
        running_disco.append(float(loss_disco))
    duration = time.perf_counter() - start
    train_loss_bce_f = float(np.mean(running_bce_f)) if running_bce_f else float("nan")
    train_loss_bce_g = float(np.mean(running_bce_g)) if running_bce_g else float("nan")
    train_loss_disco = float(np.mean(running_disco)) if running_disco else float("nan")
    return {
        "train_loss_bce_f": train_loss_bce_f,
        "train_loss_bce_g": train_loss_bce_g,
        "train_loss_disco": train_loss_disco,
        "train_loss_total": train_loss_bce_f + train_loss_bce_g + train_loss_disco,
        "train_epoch_seconds": duration,
        "train_iterations_per_second": len(loader) / duration if duration > 0 else float("nan"),
    }


##### ABCD utilities (Double-DisCo, f and g axes) ############################################

def compute_abcd_yields(scores_f: np.ndarray, scores_g: np.ndarray, labels: np.ndarray, weights: np.ndarray, cut_f: float, cut_g: float) -> Dict[str, float]:
    s_mask = labels > 0.5
    b_mask = ~s_mask
    pass_f = scores_f >= cut_f
    pass_g = scores_g >= cut_g
    A = pass_f & pass_g
    B = pass_f & (~pass_g)
    C = (~pass_f) & pass_g
    D = (~pass_f) & (~pass_g)
    w = weights.astype(float)
    A_bg = float(np.sum(w[A & b_mask]))
    B_bg = float(np.sum(w[B & b_mask]))
    C_bg = float(np.sum(w[C & b_mask]))
    D_bg = float(np.sum(w[D & b_mask]))
    return {"A_bg": A_bg, "B_bg": B_bg, "C_bg": C_bg, "D_bg": D_bg, "A": A, "B": B, "C": C, "D": D}


def generate_candidate_cut_pairs(scores_f: np.ndarray, scores_g: np.ndarray, labels: np.ndarray, weights: np.ndarray, target_eff: float, num_grid: int = 40) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    s_mask = labels > 0.5
    s_w = weights[s_mask]
    if np.sum(s_w) <= 0:
        return pairs
    sig_f = scores_f[s_mask]
    sig_g = scores_g[s_mask]
    for eS1 in np.linspace(target_eff, 1.0, num_grid):
        if eS1 <= 0:
            continue
        eS2 = target_eff / eS1
        if eS2 <= 0 or eS2 > 1:
            continue
        cut_f = float(weighted_quantile(sig_f, 1.0 - eS1, sample_weight=s_w))
        mask_f_pass = sig_f >= cut_f
        if not np.any(mask_f_pass):
            continue
        cut_g = float(weighted_quantile(sig_g[mask_f_pass], 1.0 - eS2, sample_weight=s_w[mask_f_pass]))
        pairs.append((cut_f, cut_g))
    for eS2 in np.linspace(target_eff, 1.0, num_grid):
        if eS2 <= 0:
            continue
        eS1 = target_eff / eS2
        if eS1 <= 0 or eS1 > 1:
            continue
        cut_g = float(weighted_quantile(sig_g, 1.0 - eS2, sample_weight=s_w))
        mask_g_pass = sig_g >= cut_g
        if not np.any(mask_g_pass):
            continue
        cut_f = float(weighted_quantile(sig_f[mask_g_pass], 1.0 - eS1, sample_weight=s_w[mask_g_pass]))
        pairs.append((cut_f, cut_g))
    pairs_unique = list({(round(a, 6), round(b, 6)) for (a, b) in pairs})
    return [(a, b) for (a, b) in pairs_unique]


def compute_abcd_metrics_scan(scores_f: np.ndarray, scores_g: np.ndarray, labels: np.ndarray, weights: np.ndarray, target_eff: float) -> Dict[str, Any]:
    candidates = generate_candidate_cut_pairs(scores_f, scores_g, labels, weights, target_eff)
    if len(candidates) == 0:
        return {"cut_f": float("nan"), "cut_g": float("nan")}
    best_key = (float("inf"), -float("inf"))
    best_payload: Dict[str, Any] = {}
    for (cut_f, cut_g) in candidates:
        y = compute_abcd_yields(scores_f, scores_g, labels, weights, cut_f, cut_g)
        A_bg, B_bg, C_bg, D_bg = y["A_bg"], y["B_bg"], y["C_bg"], y["D_bg"]
        eps = 1e-12
        pred_bg = B_bg * C_bg / (D_bg + eps)
        closure = pred_bg / (A_bg + eps)
        b_mask = labels < 0.5
        w = weights.astype(float)
        b_tot = float(np.sum(w[b_mask])) + eps
        b_pass = float(np.sum(w[(scores_f >= cut_f) & (scores_g >= cut_g) & b_mask]))
        b_eff = b_pass / b_tot
        b_rej = 1.0 / max(b_eff, eps)
        # Signal contamination per region δ_i = N_S(i)/N_B(i)
        s_mask = labels > 0.5
        A = y["A"]; B = y["B"]; C = y["C"]; D = y["D"]
        A_sig = float(np.sum(w[A & s_mask]))
        B_sig = float(np.sum(w[B & s_mask]))
        C_sig = float(np.sum(w[C & s_mask]))
        D_sig = float(np.sum(w[D & s_mask]))
        delta_A = A_sig / (A_bg + eps)
        delta_B = B_sig / (B_bg + eps)
        delta_C = C_sig / (C_bg + eps)
        delta_D = D_sig / (D_bg + eps)
        delta_rel = (delta_B + delta_C - delta_D) / (delta_A + eps)
        key = (abs(closure - 1.0), -b_rej)
        if key < best_key:
            best_key = key
            best_payload = {
                "cut_f": cut_f,
                "cut_g": cut_g,
                "A_bg": A_bg,
                "B_bg": B_bg,
                "C_bg": C_bg,
                "D_bg": D_bg,
                "predicted_bg": pred_bg,
                "closure": closure,
                "b_eff_2d": b_eff,
                "b_rej_2d": b_rej,
                "delta_A": delta_A,
                "delta_B": delta_B,
                "delta_C": delta_C,
                "delta_D": delta_D,
                "delta_rel": delta_rel,
                "region_masks": {"A": y["A"], "B": y["B"], "C": y["C"], "D": y["D"]},
            }
    return best_payload


def plot_scatter_f_vs_g(scores_f: np.ndarray, scores_g: np.ndarray, labels: np.ndarray, out: Path, title: str, max_points: int = 100000, cuts_by_wp: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
    n = len(scores_f)
    if n > max_points:
        rng = np.random.default_rng(1337)
        select = rng.choice(n, size=max_points, replace=False)
    else:
        select = np.arange(n)
    xf = scores_f[select]
    yg = scores_g[select]
    y = labels[select]
    plt.figure(figsize=(8, 6))
    plt.scatter(xf[y < 0.5], yg[y < 0.5], s=2, alpha=0.3, label="background")
    plt.scatter(xf[y > 0.5], yg[y > 0.5], s=2, alpha=0.3, label="signal")
    plt.xlabel("f score (x-axis)")
    plt.ylabel("g score (y-axis)")
    if cuts_by_wp is not None:
        for wp, (cf, cg) in cuts_by_wp.items():
            if not (np.isnan(cf) or np.isnan(cg)):
                plt.axvline(cf, linestyle=(0, (4, 2)), linewidth=1.2, label=f"f cut @ εS={wp}%")
                plt.axhline(cg, linestyle=(0, (2, 3)), linewidth=1.2, label=f"g cut @ εS={wp}%")
    plt.title(title + "\nGuides: verticals = f-cuts, horizontals = g-cuts at εS=10/30/50%")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def make_hist_with_estimated_background(x: np.ndarray, labels: np.ndarray, weights: np.ndarray, regions: Dict[str, np.ndarray], bins: int, out: Path, title: str, x_label: Optional[str] = None, x_range: Optional[Tuple[float, float]] = None, log_scale: bool = False) -> None:
    A = regions["A"]
    B = regions["B"]
    C = regions["C"]
    D = regions["D"]
    b_mask = labels < 0.5
    s_mask = labels > 0.5
    w = weights.astype(float)
    B_bg = float(np.sum(w[B & b_mask]))
    C_bg = float(np.sum(w[C & b_mask]))
    D_bg = float(np.sum(w[D & b_mask]))
    eps = 1e-12
    tf_B_over_D = B_bg / (D_bg + eps)
    if x_range is None:
        x_range = (float(np.min(x)), float(np.max(x)))
    nb = bins
    b_counts, b_edges = np.histogram(x[A & b_mask], bins=nb, range=x_range, weights=w[A & b_mask])
    s_counts, _ = np.histogram(x[A & s_mask], bins=nb, range=x_range, weights=w[A & s_mask])
    bC_counts, _ = np.histogram(x[C & b_mask], bins=nb, range=x_range, weights=w[C & b_mask])
    est_from_C = tf_B_over_D * bC_counts
    centers = 0.5 * (b_edges[:-1] + b_edges[1:])
    width = (b_edges[1] - b_edges[0])
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3.0, 1.0], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    ax_top.bar(centers, b_counts, width=0.9 * width, color="tab:blue", alpha=0.5, label="True background (SR)")
    ax_top.bar(centers, s_counts, bottom=b_counts, width=0.9 * width, color="tab:orange", alpha=0.5, label="Signal (SR)")
    ax_top.step(centers, est_from_C, where="mid", color="green", lw=2, label="Estimated background (from C×B/D)")
    if log_scale:
        ax_top.set_yscale("log")
    ax_top.set_ylabel("Weighted events")
    ax_top.set_title(title + "\nABCD regions: A=pass_f&pass_g, B=pass_f&fail_g, C=fail_f&pass_g, D=fail_f&fail_g; TF: C×B/D")
    ax_top.legend(loc="best")
    denom = b_counts.copy().astype(float)
    denom[denom == 0.0] = np.nan
    ratio = est_from_C / denom
    ax_bot.axhline(1.0, color="black", lw=1, linestyle=":")
    ax_bot.step(centers, ratio, where="mid", color="green")
    ax_bot.set_ylabel("est/true")
    ax_bot.set_xlabel(x_label if x_label is not None else "Variable")
    ax_bot.set_xlim(x_range[0], x_range[1])
    ax_top.set_xlim(x_range[0], x_range[1])
    ax_bot.set_ylim(0.0, np.nanmax([2.0, np.nanmax(ratio) * 1.2]))
    ensure_dir(out.parent)
    plt.savefig(out)
    plt.close(fig)


##### Plot helpers (classification) ###########################################################

def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float, out: Path, title: str) -> None:
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    plt.xlabel("Background efficiency ε_B")
    plt.ylabel("Signal efficiency ε_S")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_confusion(cm: np.ndarray, out: Path, title: str, normalize: bool = False) -> None:
    plt.figure(figsize=(6, 5))
    cm_plot = cm.astype(float)
    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True) + 1e-12
        cm_plot = cm_plot / row_sums
    im = plt.imshow(cm_plot, cmap="viridis", vmin=0.0 if normalize else None, vmax=1.0 if normalize else None)
    plt.xticks([0, 1], ["Pred. bkg", "Pred. sig"]) ; plt.yticks([0, 1], ["True bkg", "True sig"])
    plt.title(title)
    for (i, j), v in np.ndenumerate(cm_plot):
        disp = f"{v:.2f}" if normalize else f"{v:.0f}"
        thresh = cm_plot.max()/2 if cm_plot.size > 0 else 0.5
        plt.text(j, i, disp, ha="center", va="center", color="white" if v > thresh else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_logit_and_score_distributions(logits: np.ndarray, scores: np.ndarray, labels: np.ndarray, out_root_dir: Path, file_stub: str, title_prefix: str) -> None:
    for k in range(min(2, logits.shape[1])):
        subdir = out_root_dir / f"logit{k}"
        ensure_dir(subdir)
        plt.figure(figsize=(8, 5))
        target_label = "background" if k == 0 else "signal"
        plt.hist(logits[labels < 0.5, k], bins=60, alpha=0.5, label="true background", density=True)
        plt.hist(logits[labels > 0.5, k], bins=60, alpha=0.5, label="true signal", density=True)
        plt.xlabel(f"raw model logit for {target_label}")
        plt.ylabel("density")
        plt.title(f"{title_prefix}: raw model logit distribution for {target_label}")
        plt.legend(); plt.tight_layout()
        plt.savefig(subdir / f"{file_stub}_logit{k}.png")
        plt.close()
    subdir_s = out_root_dir / "score"
    ensure_dir(subdir_s)
    plt.figure(figsize=(8, 5))
    plt.hist(scores[labels < 0.5], bins=60, alpha=0.5, label="true background", density=True)
    plt.hist(scores[labels > 0.5], bins=60, alpha=0.5, label="true signal", density=True)
    plt.xlabel("model probability for signal (softmax class=1)")
    plt.ylabel("density")
    plt.title(f"{title_prefix}: score distribution")
    plt.legend(); plt.tight_layout()
    plt.savefig(subdir_s / f"{file_stub}_score.png")
    plt.close()


##### Training loop orchestration #############################################################

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(sys.argv[1:])
    print(vars(args))

    device = setup_device(args.gpunum)
    print(f"Using device: {device}")

    smear_sigma = float(args.smear)
    run_label = args.run_label or f"smear{int(smear_sigma)}"
    dirs = prepare_output_dirs(Path(args.output_root), run_label)

    payload = load_and_preprocess(smear_sigma=smear_sigma, test=args.test)
    trainset = payload["trainset"] ; valset = payload["valset"] ; testset = payload["testset"]

    # Dataloaders (QNN single script style: workers + persistent)
    nw = int(args.num_workers)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=nw, persistent_workers=(nw > 0))
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=nw, persistent_workers=(nw > 0))
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=nw, persistent_workers=(nw > 0))

    # Dataset summaries
    print_dataset_summary_split("train", payload["train_features_np"], payload["train_labels_np"], dirs["history"])
    print_dataset_summary_split("val", payload["val_features_np"], payload["val_labels_np"], dirs["history"])
    print_dataset_summary_split("test", payload["test_features_np"], payload["test_labels_np"], dirs["history"])

    # Model & optimizer (two HybridQNN heads on 13 features)
    model = DoubleQNNModel(
        n_features=13, n_classes=2,
        qnn_n_qubits=int(args.qnn_n_qubits), qnn_hidden_dim=int(args.qnn_hidden_dim),
        qnn_vqc_depth=int(args.qnn_vqc_depth), qnn_n_outputs=int(args.qnn_n_outputs),
        qnn_device=str(args.qnn_device)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    # Resume support
    history: List[Dict[str, Any]] = []
    start_epoch = 0
    if args.resume is not None and Path(args.resume).exists():
        payload_ckpt = torch.load(args.resume, map_location=device)
        if "model_state" in payload_ckpt:
            model.load_state_dict(payload_ckpt["model_state"]) 
        if "optimizer_state" in payload_ckpt:
            try:
                optimizer.load_state_dict(payload_ckpt["optimizer_state"]) 
            except Exception:
                pass
        if "history" in payload_ckpt and isinstance(payload_ckpt["history"], list):
            history = payload_ckpt["history"]
        if "epoch" in payload_ckpt:
            start_epoch = int(payload_ckpt["epoch"]) + 1
        print(f"Resuming from epoch {start_epoch}")

    legacy_log = open(dirs["history"] / args.logfile, "a")
    history_rows: List[Dict[str, Any]] = []

    for epoch in range(start_epoch, int(args.epochs)):
        print(f"Epoch {epoch:03d}")

        train_stats = run_training_epoch(model, train_loader, optimizer, device, alpha=float(args.alpha))

        train_arrays = collect_epoch_arrays_double(model, DataLoader(trainset, batch_size=args.batch_size, shuffle=False, pin_memory=True), device)
        val_arrays = collect_epoch_arrays_double(model, val_loader, device)

        # Validation loss breakdown
        model.eval()
        val_bce_f_losses: List[float] = []
        val_bce_g_losses: List[float] = []
        val_disco_losses: List[float] = []
        start_val = time.perf_counter()
        with torch.no_grad():
            for batch in tqdm(val_loader, leave=False, desc="val-loss"):
                features, labels, weights, binnums, masses = batch
                features = features.to(device)
                labels = labels.to(device)
                weights = weights.to(device)
                binnums = binnums.to(device)
                masses = masses.to(device)
                loss_total, loss_bce_f, loss_bce_g, loss_disco, _sf, _sg = compute_losses_for_batch_double(
                    model, (features, labels, weights, binnums, masses), alpha=float(args.alpha)
                )
                val_bce_f_losses.append(float(loss_bce_f))
                val_bce_g_losses.append(float(loss_bce_g))
                val_disco_losses.append(float(loss_disco))
        val_duration = time.perf_counter() - start_val
        val_stats = {
            "val_loss_bce_f": float(np.mean(val_bce_f_losses)) if val_bce_f_losses else float("nan"),
            "val_loss_bce_g": float(np.mean(val_bce_g_losses)) if val_bce_g_losses else float("nan"),
            "val_loss_disco": float(np.mean(val_disco_losses)) if val_disco_losses else float("nan"),
            "val_loss_total": (float(np.mean(val_bce_f_losses)) + float(np.mean(val_bce_g_losses)) + float(np.mean(val_disco_losses))) if val_bce_f_losses else float("nan"),
            "val_epoch_seconds": val_duration,
        }

        # Per-split metrics
        split_metrics: Dict[str, Dict[str, Any]] = {}
        for split_name, arrays in (("train", train_arrays), ("val", val_arrays)):
            scores_f = arrays["scores_f"].astype(np.float64)
            scores_g = arrays["scores_g"].astype(np.float64)
            labels = arrays["labels"].astype(np.float64)
            weights = arrays["weights"].astype(np.float64)
            logits_f = arrays["logits_f"].astype(np.float64)
            logits_g = arrays["logits_g"].astype(np.float64)

            auc_f = compute_auc(scores_f, labels, weights)
            auc_g = compute_auc(scores_g, labels, weights)

            cm_f_stats = weighted_confusion_stats(labels, scores_f, weights, threshold=0.5)
            cm_g_stats = weighted_confusion_stats(labels, scores_g, weights, threshold=0.5)
            cm_f = np.array([[cm_f_stats["tn"], cm_f_stats["fp"]], [cm_f_stats["fn"], cm_f_stats["tp"]]])
            cm_g = np.array([[cm_g_stats["tn"], cm_g_stats["fp"]], [cm_g_stats["fn"], cm_g_stats["tp"]]])

            cuts_f = {"10": weighted_threshold_at_signal_eff(scores_f, labels, weights, 0.10),
                      "30": weighted_threshold_at_signal_eff(scores_f, labels, weights, 0.30),
                      "50": weighted_threshold_at_signal_eff(scores_f, labels, weights, 0.50)}
            cuts_g = {"10": weighted_threshold_at_signal_eff(scores_g, labels, weights, 0.10),
                      "30": weighted_threshold_at_signal_eff(scores_g, labels, weights, 0.30),
                      "50": weighted_threshold_at_signal_eff(scores_g, labels, weights, 0.50)}

            bgeff_rej_cont_f: Dict[str, Dict[str, float]] = {}
            bgeff_rej_cont_g: Dict[str, Dict[str, float]] = {}
            metrics_wp_f: Dict[str, Dict[str, float]] = {}
            metrics_wp_g: Dict[str, Dict[str, float]] = {}
            for k, cut in cuts_f.items():
                if not math.isnan(cut):
                    bgeff_rej_cont_f[k] = compute_bg_eff_rej_contamination(scores_f, labels, weights, cut)
                    wp_stats = weighted_confusion_stats(labels, scores_f, weights, threshold=float(cut))
                    metrics_wp_f[k] = {"accuracy": wp_stats["accuracy"], "precision": wp_stats["precision"], "recall": wp_stats["recall"], "f1": wp_stats["f1"]}
                else:
                    bgeff_rej_cont_f[k] = {"b_eff": float("nan"), "b_rej": float("nan"), "sig_contamination": float("nan")}
                    metrics_wp_f[k] = {"accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
            for k, cut in cuts_g.items():
                if not math.isnan(cut):
                    bgeff_rej_cont_g[k] = compute_bg_eff_rej_contamination(scores_g, labels, weights, cut)
                    wp_stats = weighted_confusion_stats(labels, scores_g, weights, threshold=float(cut))
                    metrics_wp_g[k] = {"accuracy": wp_stats["accuracy"], "precision": wp_stats["precision"], "recall": wp_stats["recall"], "f1": wp_stats["f1"]}
                else:
                    bgeff_rej_cont_g[k] = {"b_eff": float("nan"), "b_rej": float("nan"), "sig_contamination": float("nan")}
                    metrics_wp_g[k] = {"accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

            # ABCD diagnostics with 2D scan
            abcd_points: Dict[str, Any] = {}
            cuts_by_wp: Dict[str, Tuple[float, float]] = {}
            for k, eff in ("10", 0.10), ("30", 0.30), ("50", 0.50):
                abcd_points[k] = compute_abcd_metrics_scan(scores_f, scores_g, labels, weights, eff)
                cuts_by_wp[k] = (abcd_points[k].get("cut_f", float("nan")), abcd_points[k].get("cut_g", float("nan")))

            # JSD vs R using outputs (threshold on f, mass-like variable = g)
            s_mask = labels > 0.5
            jsd_vs_r: Dict[str, Dict[str, float]] = {}
            for eff in (10, 30, 50):
                out = JSDvsR(sigscore=scores_f[s_mask], bgscore=scores_f[~s_mask], bgmass=scores_g[~s_mask],
                             sigweights=weights[s_mask], bgweights=weights[~s_mask], sigeff=eff,
                             minmass=0.0, maxmass=1.0)
                jsd_vs_r[str(eff)] = {"b_rej": out[0], "inv_jsd": out[1], "jsd": (1.0 / out[1] if out[1] != 0 else float("inf"))}

            split_metrics[split_name] = {
                "auc_f": auc_f,
                "auc_g": auc_g,
                "confusion_f": cm_f,
                "confusion_g": cm_g,
                "metrics_f": {"accuracy": cm_f_stats["accuracy"], "precision": cm_f_stats["precision"], "recall": cm_f_stats["recall"], "f1": cm_f_stats["f1"]},
                "metrics_g": {"accuracy": cm_g_stats["accuracy"], "precision": cm_g_stats["precision"], "recall": cm_g_stats["recall"], "f1": cm_g_stats["f1"]},
                "metrics_wp_f": metrics_wp_f,
                "metrics_wp_g": metrics_wp_g,
                "bgeff_rej_cont_f": bgeff_rej_cont_f,
                "bgeff_rej_cont_g": bgeff_rej_cont_g,
                "abcd": abcd_points,
                "cuts_by_wp": cuts_by_wp,
                "jsd_vs_r": jsd_vs_r,
                "logits_f": logits_f,
                "scores_f": scores_f,
                "logits_g": logits_g,
                "scores_g": scores_g,
                "labels": labels,
                "weights": weights,
            }

        # Produce and save plots
        # 1) ROC curves per model
        for split_name in ("train", "val"):
            sm = split_metrics[split_name]
            split_dir = "train" if split_name == "train" else "valid"
            for model_key in ("f", "g"):
                scores = sm[f"scores_{model_key}"]
                labels = sm["labels"]
                weights = sm["weights"]
                order = np.argsort(-scores)
                s_sorted, y_sorted, w_sorted = scores[order], labels[order], weights[order]
                s_mask = y_sorted > 0.5
                b_mask = ~s_mask
                S = np.sum(w_sorted[s_mask]) + 1e-12
                B = np.sum(w_sorted[b_mask]) + 1e-12
                tpr = np.cumsum(w_sorted * s_mask) / S
                fpr = np.cumsum(w_sorted * b_mask) / B
                out_dir_roc = dirs["plots_classification"] / split_dir / model_key / "roc"
                ensure_dir(out_dir_roc)
                out_path = out_dir_roc / f"roc_{split_name}_{model_key}_epoch_{epoch:03d}.png"
                plot_roc(fpr, tpr, sm[f"auc_{model_key}"], out_path, f"ROC ({split_name}, {model_key}) epoch {epoch}")

        # 2) Confusion matrices (0.5 and WPs) per model
        for split_name in ("train", "val"):
            sm = split_metrics[split_name]
            split_dir = "train" if split_name == "train" else "valid"
            for model_key in ("f", "g"):
                cm = sm[f"confusion_{model_key}"]
                out_dir_05_conf = dirs["plots_classification"] / split_dir / model_key / "Score0p5" / "confusion"
                out_dir_05_norm = dirs["plots_classification"] / split_dir / model_key / "Score0p5" / "confusion_normalized"
                ensure_dir(out_dir_05_conf)
                ensure_dir(out_dir_05_norm)
                out_path = out_dir_05_conf / f"confusion_{split_name}_{model_key}_epoch_{epoch:03d}.png"
                plot_confusion(cm, out_path, f"Confusion ({split_name}, {model_key}) @ thr=0.5 epoch {epoch}")
                out_path_norm = out_dir_05_norm / f"confusion_{split_name}_{model_key}_norm_epoch_{epoch:03d}.png"
                plot_confusion(cm, out_path_norm, f"Confusion normalized ({split_name}, {model_key}) @ thr=0.5 epoch {epoch}", normalize=True)
                for wp in ("10", "30", "50"):
                    scores = sm[f"scores_{model_key}"]
                    labels = sm["labels"]
                    weights = sm["weights"]
                    cut_thr = weighted_threshold_at_signal_eff(scores, labels, weights, float(wp)/100.0)
                    if math.isnan(cut_thr):
                        continue
                    cms = weighted_confusion_stats(labels, scores, weights, threshold=float(cut_thr))
                    cm_wp = np.array([[cms["tn"], cms["fp"]], [cms["fn"], cms["tp"]]])
                    out_dir_wp_conf = dirs["plots_classification"] / split_dir / model_key / f"WP{wp}" / "confusion"
                    out_dir_wp_norm = dirs["plots_classification"] / split_dir / model_key / f"WP{wp}" / "confusion_normalized"
                    ensure_dir(out_dir_wp_conf)
                    ensure_dir(out_dir_wp_norm)
                    out_path_wp = out_dir_wp_conf / f"confusion_{split_name}_{model_key}_wp{wp}_epoch_{epoch:03d}.png"
                    plot_confusion(cm_wp, out_path_wp, f"Confusion ({split_name}, {model_key}) @ εS={wp}% epoch {epoch}")
                    out_path_wp_norm = out_dir_wp_norm / f"confusion_{split_name}_{model_key}_wp{wp}_norm_epoch_{epoch:03d}.png"
                    plot_confusion(cm_wp, out_path_wp_norm, f"Confusion normalized ({split_name}, {model_key}) @ εS={wp}% epoch {epoch}", normalize=True)

        # 3) Logit and score distributions per model
        for split_name in ("train", "val"):
            sm = split_metrics[split_name]
            split_dir = "train" if split_name == "train" else "valid"
            for model_key in ("f", "g"):
                dists_root = dirs["plots_classification"] / split_dir / model_key / "dists"
                ensure_dir(dists_root)
                plot_logit_and_score_distributions(
                    sm[f"logits_{model_key}"], sm[f"scores_{model_key}"], sm["labels"], dists_root,
                    f"dists_{split_name}_{model_key}_epoch_{epoch:03d}", f"{split_name} {model_key}"
                )

        # 4) ABCD scatter plots (f vs g)
        for split_name in ("train", "val"):
            sm = split_metrics[split_name]
            split_dir = "train" if split_name == "train" else "valid"
            out_dir_scatter = dirs["plots_abcd"] / split_dir / "scatter_f_vs_g"
            ensure_dir(out_dir_scatter)
            out_path = out_dir_scatter / f"scatter_f_vs_g_{split_name}_epoch_{epoch:03d}.png"
            plot_scatter_f_vs_g(
                sm["scores_f"], sm["scores_g"], sm["labels"], out_path,
                f"f vs g ({split_name}) epoch {epoch}", max_points=int(args.max_points_scatter), cuts_by_wp=sm["cuts_by_wp"]
            )

        # 5) Histogram overlays (f and g) using 2D regions per WP
        for split_name in ("train", "val"):
            sm = split_metrics[split_name]
            split_dir = "train" if split_name == "train" else "valid"
            for wp in ("10", "30", "50"):
                ab = sm["abcd"][wp]
                regions = ab.get("region_masks", None)
                if regions is None:
                    continue
                for scale in ("linear", "log"):
                    out_dir_f = dirs["plots_abcd"] / split_dir / "hist_f_overlay" / scale / f"WP{wp}"
                    ensure_dir(out_dir_f)
                    out_path_f = out_dir_f / f"hist_f_overlay_wp{wp}_{split_name}_epoch_{epoch:03d}.png"
                    make_hist_with_estimated_background(
                        sm["scores_f"], sm["labels"], sm["weights"], regions, bins=60, out=out_path_f,
                        title=f"f overlays ({split_name}) @ εS={wp}% epoch {epoch}",
                        x_label="f score", x_range=(-0.1, 1.1), log_scale=(scale == "log")
                    )
                for scale in ("linear", "log"):
                    out_dir_g = dirs["plots_abcd"] / split_dir / "hist_g_overlay" / scale / f"WP{wp}"
                    ensure_dir(out_dir_g)
                    out_path_g = out_dir_g / f"hist_g_overlay_wp{wp}_{split_name}_epoch_{epoch:03d}.png"
                    make_hist_with_estimated_background(
                        sm["scores_g"], sm["labels"], sm["weights"], regions, bins=60, out=out_path_g,
                        title=f"g overlays ({split_name}) @ εS={wp}% epoch {epoch}",
                        x_label="g score", x_range=(-0.1, 1.1), log_scale=(scale == "log")
                    )

        # Per-epoch feature histograms (13 inputs)
        def plot_feature_hists(x: np.ndarray, y: np.ndarray, split: str) -> None:
            nfeat = x.shape[1]
            for i in range(nfeat):
                plt.figure(figsize=(8,5))
                plt.hist(x[y < 0.5, i], bins=60, alpha=0.5, label="background", density=False)
                plt.hist(x[y > 0.5, i], bins=60, alpha=0.5, label="signal", density=False)
                name = FEATURE_NAMES_13[i] if i < len(FEATURE_NAMES_13) else f"feature[{i}]"
                safe = sanitize_name(name)
                plt.xlabel(f"{name} (scaled)"); plt.ylabel("events"); plt.title(f"{name} distribution ({split}) epoch {epoch}"); plt.legend()
                split_dir = "train" if split == "train" else "valid"
                feat_dir = dirs["plots_features"] / split_dir / safe
                ensure_dir(feat_dir)
                plt.tight_layout(); plt.savefig(feat_dir / f"feature_{safe}_hist_{split}_epoch_{epoch:03d}.png"); plt.close()
        plot_feature_hists(train_arrays.get("features"), train_arrays.get("labels"), "train")
        plot_feature_hists(val_arrays.get("features"), val_arrays.get("labels"), "val")

        # One-time full-dataset feature histograms
        if epoch == start_epoch:
            plot_feature_distributions_once(
                payload["train_features_np"], payload["val_features_np"], payload["test_features_np"],
                payload["train_labels_np"], payload["val_labels_np"], payload["test_labels_np"],
                FEATURE_NAMES_13, dirs["plots_features"]
            )

        # Per-epoch summary row
        row: Dict[str, Any] = {
            "epoch": epoch,
            **train_stats,
            **val_stats,
            "val_auc_f": split_metrics["val"]["auc_f"],
            "val_auc_g": split_metrics["val"]["auc_g"],
            "train_auc_f": split_metrics["train"]["auc_f"],
            "train_auc_g": split_metrics["train"]["auc_g"],
        }
        for split_name in ("train", "val"):
            for model_key in ("f", "g"):
                bgr = split_metrics[split_name][f"bgeff_rej_cont_{model_key}"]
                for wp in ("10", "30", "50"):
                    row[f"{split_name}_b_eff_{model_key}_{wp}"] = bgr[wp]["b_eff"]
                    row[f"{split_name}_b_rej_{model_key}_{wp}"] = bgr[wp]["b_rej"]
                    row[f"{split_name}_sig_cont_{model_key}_{wp}"] = bgr[wp]["sig_contamination"]
                m = split_metrics[split_name][f"metrics_{model_key}"]
                row[f"{split_name}_accuracy_{model_key}"] = m["accuracy"]
                row[f"{split_name}_precision_{model_key}"] = m["precision"]
                row[f"{split_name}_recall_{model_key}"] = m["recall"]
                row[f"{split_name}_f1_{model_key}"] = m["f1"]
                for wp in ("10", "30", "50"):
                    mwp = split_metrics[split_name][f"metrics_wp_{model_key}"][wp]
                    row[f"{split_name}_accuracy_{model_key}_wp{wp}"] = mwp["accuracy"]
                    row[f"{split_name}_precision_{model_key}_wp{wp}"] = mwp["precision"]
                    row[f"{split_name}_recall_{model_key}_wp{wp}"] = mwp["recall"]
                    row[f"{split_name}_f1_{model_key}_wp{wp}"] = mwp["f1"]
        for split_name in ("train", "val"):
            for wp in ("10", "30", "50"):
                ab = split_metrics[split_name]["abcd"][wp]
                row[f"{split_name}_cut_f_wp{wp}"] = ab.get("cut_f", float("nan"))
                row[f"{split_name}_cut_g_wp{wp}"] = ab.get("cut_g", float("nan"))
                row[f"{split_name}_closure_{wp}"] = ab.get("closure", float("nan"))
                row[f"{split_name}_b_rej_2d_{wp}"] = ab.get("b_rej_2d", float("nan"))
                j = split_metrics[split_name]["jsd_vs_r"][wp]
                row[f"{split_name}_inv_jsd_fg_{wp}"] = j["inv_jsd"]
                row[f"{split_name}_jsd_fg_{wp}"] = j["jsd"]
                row[f"{split_name}_b_rej_jsd_fg_{wp}"] = j["b_rej"]
                # Per-region signal contamination and normalized (2D ABCD)
                row[f"{split_name}_deltaA_{wp}"] = ab.get("delta_A", float("nan"))
                row[f"{split_name}_deltaB_{wp}"] = ab.get("delta_B", float("nan"))
                row[f"{split_name}_deltaC_{wp}"] = ab.get("delta_C", float("nan"))
                row[f"{split_name}_deltaD_{wp}"] = ab.get("delta_D", float("nan"))
                row[f"{split_name}_delta_rel_{wp}"] = ab.get("delta_rel", float("nan"))

        history_rows.append(row)

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history_rows,
        }
        ckpt_path = dirs["checkpoints"] / f"epoch_{epoch:03d}.pth"
        torch.save(ckpt, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

        # NPZ exports
        if args.save_epoch_arrays:
            npz_path_train = dirs["history_arrays"] / f"train_epoch_{epoch:03d}.npz"
            npz_path_val = dirs["history_arrays"] / f"val_epoch_{epoch:03d}.npz"
            ensure_dir(dirs["history_arrays"])
            np.savez_compressed(
                npz_path_train,
                scores_f=train_arrays["scores_f"], logits_f=train_arrays["logits_f"],
                scores_g=train_arrays["scores_g"], logits_g=train_arrays["logits_g"],
                labels=train_arrays["labels"], weights=train_arrays["weights"], masses=train_arrays["masses"], features=train_arrays["features"]
            )
            np.savez_compressed(
                npz_path_val,
                scores_f=val_arrays["scores_f"], logits_f=val_arrays["logits_f"],
                scores_g=val_arrays["scores_g"], logits_g=val_arrays["logits_g"],
                labels=val_arrays["labels"], weights=val_arrays["weights"], masses=val_arrays["masses"], features=val_arrays["features"]
            )
            print(f"Saved epoch arrays to {npz_path_train} and {npz_path_val}")

        legacy_log.write(
            f"epoch,{epoch},val_auc_f,{row['val_auc_f']},val_auc_g,{row['val_auc_g']},train_loss_total,{row['train_loss_total']},"
            f"train_loss_bce_f,{row['train_loss_bce_f']},train_loss_bce_g,{row['train_loss_bce_g']},train_loss_disco,{row['train_loss_disco']}\n"
        )
        legacy_log.flush()

        print(
            f"AUC(val,f/g)={row['val_auc_f']:.3f}/{row['val_auc_g']:.3f} | "
            f"loss(train,total/bce_f/bce_g/disco)={row['train_loss_total']:.3f}/{row['train_loss_bce_f']:.3f}/{row['train_loss_bce_g']:.3f}/{row['train_loss_disco']:.3f} | "
            f"loss(val,total/bce_f/bce_g/disco)={row['val_loss_total']:.3f}/{row['val_loss_bce_f']:.3f}/{row['val_loss_bce_g']:.3f}/{row['val_loss_disco']:.3f} | "
            f"closure30(val)={row['val_closure_30']:.3f} | invJSD30_fg(val)={row['val_inv_jsd_fg_30']:.3f} | b_rej30_fg(val)={row['val_b_rej_jsd_fg_30']:.2f}"
        )

    legacy_log.close()
    history_json = dirs["history"] / "history.json"
    save_json(history_rows, history_json)
    print(f"Saved history to {history_json}")

    # History plots (per-model and global) — mirror DNN double script layout
    import pandas as pd
    try:
        hist_df = pd.DataFrame(history_rows)
        x = hist_df["epoch"].values
        hist_root_f_lin = dirs["plots_history"] / "f" / "linear"
        hist_root_f_log = dirs["plots_history"] / "f" / "log"
        hist_root_g_lin = dirs["plots_history"] / "g" / "linear"
        hist_root_g_log = dirs["plots_history"] / "g" / "log"
        for p in (hist_root_f_lin, hist_root_f_log, hist_root_g_lin, hist_root_g_log):
            ensure_dir(p)

        plt.figure(figsize=(8,5))
        plt.plot(x, hist_df.get("train_auc_f", np.nan), label="train AUC f")
        plt.plot(x, hist_df.get("val_auc_f", np.nan), label="val AUC f")
        plt.xlabel("Epoch"); plt.ylabel("AUC"); plt.title("AUC (f) vs epoch"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(hist_root_f_lin / "auc_vs_epoch.png"); plt.yscale("log"); plt.savefig(hist_root_f_log / "auc_vs_epoch_log.png"); plt.close()
        plt.figure(figsize=(8,5))
        plt.plot(x, hist_df.get("train_auc_g", np.nan), label="train AUC g")
        plt.plot(x, hist_df.get("val_auc_g", np.nan), label="val AUC g")
        plt.xlabel("Epoch"); plt.ylabel("AUC"); plt.title("AUC (g) vs epoch"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(hist_root_g_lin / "auc_vs_epoch.png"); plt.yscale("log"); plt.savefig(hist_root_g_log / "auc_vs_epoch_log.png"); plt.close()

        plt.figure(figsize=(10,6))
        plt.plot(x, hist_df.get("train_loss_total", np.nan), color="C0", linestyle="-", label="train total")
        plt.plot(x, hist_df.get("val_loss_total", np.nan), color="C1", linestyle="-", label="val total")
        plt.plot(x, hist_df.get("train_loss_bce_f", np.nan), color="C0", linestyle="--", label="train BCE f")
        plt.plot(x, hist_df.get("val_loss_bce_f", np.nan), color="C1", linestyle="--", label="val BCE f")
        plt.plot(x, hist_df.get("train_loss_bce_g", np.nan), color="C0", linestyle=":", label="train BCE g")
        plt.plot(x, hist_df.get("val_loss_bce_g", np.nan), color="C1", linestyle=":", label="val BCE g")
        plt.plot(x, hist_df.get("train_loss_disco", np.nan), color="C2", linestyle="-.", label="train DisCo(f,g)")
        plt.plot(x, hist_df.get("val_loss_disco", np.nan), color="C3", linestyle="-.", label="val DisCo(f,g)")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss components vs epoch"); plt.legend(ncol=2); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(dirs["plots_history"] / "losses_vs_epoch.png"); plt.yscale("log"); plt.savefig(dirs["plots_history"] / "losses_vs_epoch_log.png"); plt.close()

        for metric in ("accuracy", "precision", "recall", "f1"):
            plt.figure(figsize=(8,5))
            plt.plot(x, hist_df.get(f"train_{metric}_f", np.nan), label=f"train {metric} (f)")
            plt.plot(x, hist_df.get(f"val_{metric}_f", np.nan), label=f"val {metric} (f)")
            plt.xlabel("Epoch"); plt.ylabel(metric); plt.title(f"{metric} (f) vs epoch"); plt.legend(); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(hist_root_f_lin / f"{metric}_vs_epoch.png"); plt.yscale("log"); plt.savefig(hist_root_f_log / f"{metric}_vs_epoch_log.png"); plt.close()
            plt.figure(figsize=(8,5))
            plt.plot(x, hist_df.get(f"train_{metric}_g", np.nan), label=f"train {metric} (g)")
            plt.plot(x, hist_df.get(f"val_{metric}_g", np.nan), label=f"val {metric} (g)")
            plt.xlabel("Epoch"); plt.ylabel(metric); plt.title(f"{metric} (g) vs epoch"); plt.legend(); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(hist_root_g_lin / f"{metric}_vs_epoch.png"); plt.yscale("log"); plt.savefig(hist_root_g_log / f"{metric}_vs_epoch_log.png"); plt.close()

        for mkey, ylabel in (("b_eff", "background efficiency ε_B"), ("b_rej", "background rejection 1/ε_B"), ("sig_cont", "signal contamination")):
            for model_key, root_lin, root_log in (("f", hist_root_f_lin, hist_root_f_log), ("g", hist_root_g_lin, hist_root_g_log)):
                plt.figure(figsize=(8,5))
                for wp, style in [("10","--"),("30","-"),("50",":")]:
                    plt.plot(x, hist_df.get(f"train_{mkey}_{model_key}_{wp}", np.nan), color="C0", linestyle=style, label=f"train @ εS={wp}%")
                    plt.plot(x, hist_df.get(f"val_{mkey}_{model_key}_{wp}", np.nan), color="C1", linestyle=style, linewidth=2, label=f"val @ εS={wp}%")
                plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(f"{ylabel} ({model_key}) vs epoch")
                plt.legend(ncol=2); plt.grid(True, alpha=0.3)
                plt.tight_layout(); plt.savefig(root_lin / f"{mkey}_vs_epoch.png"); plt.yscale("log"); plt.savefig(root_log / f"{mkey}_vs_epoch_log.png"); plt.close()

        plt.figure(figsize=(8,5))
        for wp, style in [("10","--"),("30","-"),("50",":")]:
            plt.plot(x, hist_df.get(f"train_jsd_fg_{wp}", np.nan), color="C0", linestyle=style, label=f"train JSD (fg {wp}%)")
            plt.plot(x, hist_df.get(f"val_jsd_fg_{wp}", np.nan), color="C1", linestyle=style, linewidth=2, label=f"val JSD (fg {wp}%)")
        plt.xlabel("Epoch"); plt.ylabel("JSD"); plt.title("JSD(f→g) vs epoch (εS=10/30/50%)")
        plt.legend(ncol=2); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(dirs["plots_history"] / "jsd_vs_epoch.png"); plt.yscale("log"); plt.savefig(dirs["plots_history"] / "jsd_vs_epoch_log.png"); plt.close()

        plt.figure(figsize=(8,5))
        for wp, style in [("10","--"),("30","-"),("50",":")]:
            plt.plot(x, hist_df.get(f"train_inv_jsd_fg_{wp}", np.nan), color="C0", linestyle=style, label=f"train 1/JSD (fg {wp}%)")
            plt.plot(x, hist_df.get(f"val_inv_jsd_fg_{wp}", np.nan), color="C1", linestyle=style, linewidth=2, label=f"val 1/JSD (fg {wp}%)")
        plt.xlabel("Epoch"); plt.ylabel("1/JSD"); plt.title("Inverse JSD(f→g) vs epoch (εS=10/30/50%)")
        plt.legend(ncol=2); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(dirs["plots_history"] / "inv_jsd_vs_epoch.png"); plt.yscale("log"); plt.savefig(dirs["plots_history"] / "inv_jsd_vs_epoch_log.png"); plt.close()

        plt.figure(figsize=(8,5))
        plt.axhspan(0.9, 1.1, color="grey", alpha=0.15, label="±10% band")
        for wp, style in [("10","--"),("30","-"),("50",":")]:
            plt.plot(x, hist_df.get(f"train_closure_{wp}", np.nan), color="C0", linestyle=style, label=f"train closure ({wp}%)")
            plt.plot(x, hist_df.get(f"val_closure_{wp}", np.nan), color="C1", linestyle=style, linewidth=2, label=f"val closure ({wp}%)")
        plt.xlabel("Epoch"); plt.ylabel("Closure N_pred/N_true"); plt.title("ABCD closure (2D f,g) vs epoch (εS=10/30/50%)")
        plt.legend(ncol=2); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(dirs["plots_history"] / "closure_vs_epoch.png"); plt.yscale("log"); plt.savefig(dirs["plots_history"] / "closure_vs_epoch_log.png"); plt.close()
        
        # Normalized signal contamination δ_rel vs epoch (εS=10/30/50%) [2D ABCD]
        plt.figure(figsize=(8,5))
        for wp, style in [("10","--"),("30","-"),("50",":")]:
            plt.plot(x, hist_df.get(f"train_delta_rel_{wp}", np.nan), color="C0", linestyle=style, label=f"train δ_rel ({wp}%)")
            plt.plot(x, hist_df.get(f"val_delta_rel_{wp}", np.nan), color="C1", linestyle=style, linewidth=2, label=f"val δ_rel ({wp}%)")
        plt.xlabel("Epoch"); plt.ylabel("δ_rel"); plt.title("Normalized signal contamination δ_rel (2D) vs epoch (εS=10/30/50%)")
        plt.legend(ncol=2); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(dirs["plots_history"] / "delta_rel_vs_epoch.png"); plt.yscale("log"); plt.savefig(dirs["plots_history"] / "delta_rel_vs_epoch_log.png"); plt.close()

        # Per-region signal contamination δ_A/B/C/D vs epoch [2D ABCD]
        for region, key in [("A", "deltaA"), ("B", "deltaB"), ("C", "deltaC"), ("D", "deltaD")]:
            plt.figure(figsize=(8,5))
            for wp, style in [("10","--"),("30","-"),("50",":")]:
                plt.plot(x, hist_df.get(f"train_{key}_{wp}", np.nan), color="C0", linestyle=style, label=f"train δ_{region} ({wp}%)")
                plt.plot(x, hist_df.get(f"val_{key}_{wp}", np.nan), color="C1", linestyle=style, linewidth=2, label=f"val δ_{region} ({wp}%)")
            plt.xlabel("Epoch"); plt.ylabel(f"δ_{region}"); plt.title(f"Signal contamination δ_{region} (2D) vs epoch (εS=10/30/50%)")
            plt.legend(ncol=2); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(dirs["plots_history"] / f"delta_{region}_vs_epoch.png"); plt.yscale("log"); plt.savefig(dirs["plots_history"] / f"delta_{region}_vs_epoch_log.png"); plt.close()
    except Exception as e:
        print(f"Failed to render history plots: {e}")

    # Final test-set evaluation
    print("Running final evaluation on the test set ...")
    test_arrays = collect_epoch_arrays_double(model, test_loader, device)
    test_scores_f = test_arrays["scores_f"].astype(np.float64)
    test_scores_g = test_arrays["scores_g"].astype(np.float64)
    test_labels = test_arrays["labels"].astype(np.float64)
    test_weights = test_arrays["weights"].astype(np.float64)
    test_auc_f = compute_auc(test_scores_f, test_labels, test_weights)
    test_auc_g = compute_auc(test_scores_g, test_labels, test_weights)
    abcd_test = compute_abcd_metrics_scan(test_scores_f, test_scores_g, test_labels, test_weights, 0.30)
    jsd_test_out = JSDvsR(sigscore=test_scores_f[test_labels > 0.5], bgscore=test_scores_f[test_labels < 0.5], bgmass=test_scores_g[test_labels < 0.5], sigweights=test_weights[test_labels > 0.5], bgweights=test_weights[test_labels < 0.5], sigeff=30, minmass=0.0, maxmass=1.0)
    summary = {
        "test_auc_f": test_auc_f,
        "test_auc_g": test_auc_g,
        "test_abcd_30": {
            "closure": abcd_test.get("closure", float("nan")),
            "b_rej_2d": abcd_test.get("b_rej_2d", float("nan")),
            "cut_f": abcd_test.get("cut_f", float("nan")),
            "cut_g": abcd_test.get("cut_g", float("nan")),
        },
        "test_jsd_vs_r_30_fg": {"b_rej": jsd_test_out[0], "inv_jsd": jsd_test_out[1]},
    }
    save_json(summary, dirs["history"] / "test_evaluation.json")
    print("Done.")


if __name__ == "__main__":
    main()
