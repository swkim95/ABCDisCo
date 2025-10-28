#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Single-DisCo training script (Torch) with extended evaluation and checkpointing.

This script faithfully preserves the original data processing, model, and loss logic from
`ABCD_topjets_HLF_mD_smear.py` and `model.py`, while adding:
  - m_hat computation after mass smearing: m_hat = 1 - |m_jet - m_top| / m_top (m_top = 172.5 GeV)
  - Per-epoch checkpointing (model + optimizer) and history logging
  - Rich train/val diagnostics each epoch (ROC/AUC, confusion matrices, score/logit distributions,
    accuracy/precision/recall/F1, background efficiency/rejection/contamination at εS = 10/30/50%)
  - ABCD-specific diagnostics using m_hat as y-axis: JSD, inverse JSD, JSDvsR, closure, scatter
    of (score vs m_hat), and stacked histograms with estimated background overlays

Notes for HEP/QML users (physics to code mapping):
  - Inputs to the classifier remain the original 11 HLF observables (mass and pT removed),
    scaled with a global min–max computed over the concatenated train/val/test set, exactly as
    in the reference script. The DisCo decorrelation penalty still uses the smeared physical jet
    mass as in the original.
  - m_hat is computed only for evaluation/ABCD axes and visualizations; it does not alter
    inputs or the decorrelation target to preserve baseline behavior.
  - ABCD regions use the classifier score (x-axis) and m_hat (y-axis). The signal region (SR)
    is defined by the central 40% of the signal m_hat distribution (quantiles [0.3, 0.7]).
  - Estimated background shapes in SR are overlaid by scaling the B-shape with C/D and the
    C-shape with B/D (simple shape transfer approximation consistent with ABCD normalization).

Usage (env hint on this HPC):
  - Run `setup` first in your shell (loads conda env: ABCDisCo).
  - Example: python script/train_abcd_single.py --smear 25 --gpunum 0 --logfile log.csv \
              --run_label smear25

This script is organized into notebook-like sections separated by "#####" comment headers.
Each section includes explanatory comments describing the physics/ML operations performed.
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

# Local modules (preserve original logic)
# Ensure repository root is on sys.path so local modules import correctly when running from script/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from data_loader import TopTaggingDataset
from networks import DNNclassifier
from QNN_networks import HybridQNN
from evaluation import JSDvsR, weighted_quantile
from disco import distance_corr_unbiased

# Human-readable names for the 11 NN input features (scaled).
# Original 13 observables in file columns 1: (after label at col 0):
# [mass, pt, tau1_half, tau2_half, tau3_half, tau1, tau2, tau3, tau4, tau1_sq, tau2_sq, tau3_sq, tau4_sq]
# Inputs passed to the model slice out mass and pt → keep the following 11:
FEATURE_NAMES: List[str] = [
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
        description="Unified Single-DisCo training with extended diagnostics and checkpointing"
    )
    parser.add_argument("--gpunum", default="0", help="GPU index passed to CUDA_VISIBLE_DEVICES (default: 0)")
    parser.add_argument("--logfile", default="log.csv", help="CSV log file for legacy state rows (as in original)")
    parser.add_argument("--smear", default="25", help="Gaussian mass smearing sigma [GeV] (default: 25)")
    parser.add_argument("--run_label", default=None, help="Optional label to name run outputs (e.g. smear25)")
    parser.add_argument("--resume", default=None, help="Path to checkpoint .pth to resume from (model+optimizer+history)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default: 200)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size (default: 10000)")
    parser.add_argument("--alpha", type=float, default=100.0, help="DisCo penalty coefficient (default: 100)")
    parser.add_argument(
        "--max_points_scatter", type=int, default=100000,
        help="Max points used in scatter plots to avoid memory blowup (default: 100k)"
    )
    parser.add_argument(
        "--output_root", default="runs", help="Root directory for outputs (checkpoints/plots/history)"
    )
    parser.add_argument(
        "--save_epoch_arrays", action="store_true", default=True,
        help="Save per-epoch arrays (scores/labels/weights/masses/features) to NPZ for full reproducibility"
    )
    parser.add_argument("--test", action="store_true", default=False, help="Test mode: train using small dataset")
    # Hybrid QNN configuration (for QNN head)
    parser.add_argument("--qnn-n-qubits", type=int, default=6, help="Number of qubits / angles for VQC")
    parser.add_argument("--qnn-hidden-dim", type=int, default=64, help="Hidden dimension of front FCN")
    parser.add_argument("--qnn-vqc-depth", type=int, default=4, help="Number of StronglyEntanglingLayers")
    parser.add_argument("--qnn-n-outputs", type=int, default=2, help="Number of output logits")
    parser.add_argument("--qnn-device", type=str, default="default.qubit", help="PennyLane device (e.g., default.qubit, lightning.qubit, lightning.gpu)")
    # Performance knobs (do not change data preprocessing)
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers for parallel batch prep (0=main thread)")
    parser.add_argument("--eval_train_sample_size", type=int, default=100000, help="Max train examples used when collecting epoch arrays (0=use full train set)")
    parser.add_argument("--disco_bg_max", type=int, default=2048, help="Max background events per batch used in DisCo penalty (subsample to limit O(n^2) cost)")
    return parser


##### Utility helpers #########################################################################

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_device(gpu_index_str: str) -> torch.device:
    """Select CUDA device if available; allow user to set CUDA_VISIBLE_DEVICES.

    - Sets CUDA_VISIBLE_DEVICES to the provided index string (e.g., "0").
    - Returns torch.device("cuda:0") if CUDA is available, else CPU.
    """
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


def compute_mhat_from_mass(mass_array: np.ndarray, m_top: float = 172.5) -> np.ndarray:
    """Compute m_hat = 1 - |m_jet - m_top| / m_top using smeared mass.
    Returns an array in principle within [1 - Δ/m_top], not explicitly clipped.
    """
    return 1.0 - np.abs(mass_array - m_top) / m_top


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
    """Compute ROC AUC using numpy only to avoid heavy sklearn dependency on HPC.
    Uses the Mann–Whitney U statistic equivalence to AUC.
    """
    # Extract positive (signal) and negative (background) scores with weights
    s_mask = labels > 0.5
    b_mask = ~s_mask
    s_scores = scores[s_mask]
    b_scores = scores[b_mask]
    s_w = weights[s_mask]
    b_w = weights[b_mask]
    if s_scores.size == 0 or b_scores.size == 0:
        return float("nan")
    # Sort combined scores and compute rank-based AUC with weights
    order = np.argsort(np.concatenate([s_scores, b_scores]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, order.size + 1)
    w = np.concatenate([s_w, b_w]).astype(float)
    s_w_sum = float(np.sum(s_w))
    b_w_sum = float(np.sum(b_w))
    if s_w_sum <= 0 or b_w_sum <= 0:
        return float("nan")
    # Weighted rank sum for signal
    s_ranks = ranks[: s_scores.size]
    auc = (np.sum(s_w * s_ranks) - s_w_sum * (s_w_sum + 1) / 2.0) / (s_w_sum * b_w_sum)
    return float(auc)


def weighted_threshold_at_signal_eff(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray, target_eff: float) -> float:
    """Find score cut such that the weighted signal efficiency equals target_eff in [0,1]."""
    s_mask = labels > 0.5
    if np.sum(weights[s_mask]) <= 0:
        return float(np.nan)
    cut = float(weighted_quantile(scores[s_mask], 1.0 - target_eff, sample_weight=weights[s_mask]))
    return cut


def compute_bg_eff_rej_contamination(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray, cut: float) -> Dict[str, float]:
    """Compute background efficiency, rejection, and signal contamination above a given cut."""
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
    """Load train/val/test from uncompressed .dat files, smear mass, compute global min–max scaling, split.

    - Files: topsample_train_tau.dat, topsample_val_tau.dat, topsample_test_tau.dat
    - Skip header rows (15) matching original script; delimiter ','
    - Smear mass (column index 1) by N(0, smear_sigma)
    - Global min–max scaling applied to all 13 observables (columns 1:)
    - Features used for the classifier are columns 2:-4 after stacking (i.e., the 11 HLF inputs;
      mass and pT are excluded), exactly as in the original.
    - Dataset items also carry labels, unit weights, dummy binnums, and unscaled smeared mass.
    - We additionally compute m_hat arrays (using smeared mass) for diagnostics.
    """
    # Read raw text tables
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

    # Apply Gaussian smearing to the mass (column index 1) to degrade mass power
    rng = np.random.default_rng()
    all_raw[:, 1] = all_raw[:, 1] + smear_sigma * rng.standard_normal(size=all_raw.shape[0])

    # m_hat from smeared physical mass (keep for diagnostics/ABCD; not used as NN input)
    all_masses = all_raw[:, 1].astype(np.float64)
    all_mhat = compute_mhat_from_mass(all_masses, m_top=172.5).astype(np.float64)

    # Global min–max scaling across 13 observables (cols 1:)
    obs = all_raw[:, 1:]
    obs_min = np.min(obs, axis=0)
    obs_max = np.max(obs, axis=0)
    obs_scaled = (obs - obs_min) / (obs_max - obs_min + 1e-12)

    labels = all_raw[:, 0].reshape((-1, 1))
    weights = np.ones((all_raw.shape[0], 1), dtype=np.float64)
    binnums = np.ones((all_raw.shape[0], 1), dtype=np.float64)
    masses_column = all_masses.reshape((-1, 1)).astype(np.float64)

    stacked = np.hstack((obs_scaled.astype(np.float32), labels.astype(np.float32), weights.astype(np.float32), binnums.astype(np.float32), masses_column.astype(np.float32)))
    stacked_t = torch.from_numpy(stacked.astype("float32"))

    # Original split sizes
    Ntrain = 20000 if(test) else 200000
    Nval = 20000 if(test) else 900000
    Ntest = 20000 if(test) else 900000

    traindata = stacked_t[:Ntrain]
    valdata = stacked_t[Ntrain : (Ntrain + Nval)]
    testdata = stacked_t[(Ntrain + Nval) : (Ntrain + Nval + Ntest)]

    # Keep m_hat aligned with splits (numpy arrays for diagnostics)
    mhat_train = all_mhat[:Ntrain]
    mhat_val = all_mhat[Ntrain : (Ntrain + Nval)]
    mhat_test = all_mhat[(Ntrain + Nval) : (Ntrain + Nval + Ntest)]

    # Dataset selection for NN inputs and targets (preserve original slicing)
    # Inputs: [:,2:-4] removes scaled mass and pT, leaving 11 HLF inputs
    trainset = TopTaggingDataset(traindata[:, 2:-4], traindata[:, -4], traindata[:, -3], traindata[:, -2], traindata[:, -1])
    valset = TopTaggingDataset(valdata[:, 2:-4], valdata[:, -4], valdata[:, -3], valdata[:, -2], valdata[:, -1])
    testset = TopTaggingDataset(testdata[:, 2:-4], testdata[:, -4], testdata[:, -3], testdata[:, -2], testdata[:, -1])

    # Also extract numpy arrays for summaries and plotting (features for preview only)
    train_features_np = to_numpy_detached(traindata[:, 2:-4])
    val_features_np = to_numpy_detached(valdata[:, 2:-4])
    test_features_np = to_numpy_detached(testdata[:, 2:-4])
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
        "mhat_train": mhat_train,
        "mhat_val": mhat_val,
        "mhat_test": mhat_test,
    }


def print_dataset_summary_split(name: str, x: np.ndarray, y: np.ndarray, mhat: np.ndarray, out_dir: Path, num_rows_preview: int = 10) -> None:
    """Print and save a rich textual summary of the split:
    - total events, class counts
    - first N rows of features for signal and background separately
    - basic ranges for the first few features and m_hat
    """
    total = x.shape[0]
    cls0 = int(np.sum(y < 0.5))
    cls1 = int(np.sum(y > 0.5))
    print(f"[{name}] events={total} | signal={cls1} | background={cls0}")
    # Save textual preview
    lines: List[str] = []
    lines.append(f"Split: {name}\n")
    lines.append(f"Total events: {total}\n")
    lines.append(f"Signal: {cls1} | Background: {cls0}\n")
    # Feature ranges (first 5 features)
    k = min(5, x.shape[1])
    mins = np.min(x, axis=0)
    maxs = np.max(x, axis=0)
    lines.append("Feature ranges (first 5):\n")
    for i in range(k):
        lines.append(f"  f{i}: min={mins[i]:.4f}, max={maxs[i]:.4f}\n")
    lines.append(f"m_hat: min={np.min(mhat):.4f}, max={np.max(mhat):.4f}\n\n")
    # First rows
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
    """Draw once the 1D histograms of signal and background for m_hat and all 11 input features.
    We use the full input samples (train+val+test concatenated) since their distributions are static.
    """
    ensure_dir(out_dir)
    X = np.concatenate([train_x, val_x, test_x], axis=0)
    Y = np.concatenate([train_y, val_y, test_y], axis=0)
    nfeat = X.shape[1]
    for i in range(nfeat):
        fname = sanitize_name(feature_names[i]) if i < len(feature_names) else f"feature{i}"
        plt.figure(figsize=(8, 5))
        plt.hist(X[Y < 0.5, i], bins=80, alpha=0.5, label="background", density=False)
        plt.hist(X[Y > 0.5, i], bins=80, alpha=0.5, label="signal", density=False)
        plt.xlabel(f"{feature_names[i] if i < len(feature_names) else f'feature[{i}]'} (scaled)")
        plt.ylabel("events")
        plt.title(f"Input feature distribution: {feature_names[i] if i < len(feature_names) else f'feature[{i}]'} (full dataset)")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / f"feature_{fname}_hist_full.png")
        plt.close()


##### Model, loss, and training/evaluation routines ##########################################

class SingleDiscoModel(nn.Module):
    """Wrapper exposing logits and score; uses HybridQNN as head for QNN."""
    def __init__(self, n_features: int = 11, n_classes: int = 2,
                 qnn_n_qubits: int = 6, qnn_hidden_dim: int = 64,
                 qnn_vqc_depth: int = 4, qnn_n_outputs: int = 2,
                 qnn_device: str = "default.qubit") -> None:
        super().__init__()
        # Keep signature compatible; the head consumes 11 features, outputs logits
        self.head = HybridQNN(
            n_features=int(n_features),
            hidden_dim=int(qnn_hidden_dim),
            n_qubits=int(qnn_n_qubits),
            vqc_depth=int(qnn_vqc_depth),
            n_outputs=int(qnn_n_outputs),
            qdevice=str(qnn_device),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.head(x)
        score = softmax_class1(logits)
        return logits, score


def compute_losses_for_batch(model: nn.Module, batch: Tuple[torch.Tensor, ...], alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute BCE classification loss and unbiased distance-correlation penalty on background.

    Inputs:
      - model: returns (logits, score)
      - batch: (features, labels, weights, binnums, masses)
      - alpha: penalty scale

    Outputs:
      - loss_total, loss_bce, loss_disco, score (class-1 probability)

    The decorrelation loss matches the original 'dist_unbiased' mode in model.py: it uses the
    unbiased distance-correlation between the classifier score and the jet mass, restricted to
    the background subset in the batch, with weights normalised to sum to the sample size.
    """
    features, labels, weights, _binnums, masses = batch
    logits, score = model(features)
    loss_bce = F.binary_cross_entropy(score, labels, weight=weights)

    # Background-only decorrelation penalty (unbiased distance correlation)
    loss_disco = torch.tensor(0.0, device=features.device, dtype=features.dtype)
    b_mask = labels < 0.5
    if torch.any(b_mask):
        s_b = score[b_mask]
        m_b = masses[b_mask]
        w_b = weights[b_mask]
        if s_b.numel() > 2:
            w_norm = w_b / (torch.sum(w_b) + 1e-12) * float(len(w_b))
            loss_disco = distance_corr_unbiased(s_b, m_b, w_norm, power=1)

    loss_total = loss_bce + alpha * loss_disco
    return loss_total, loss_bce.detach(), loss_disco.detach(), score.detach()


@torch.no_grad()
def collect_epoch_arrays(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    """Run inference to collect logits, scores, labels, weights, and masses for a full split.
    This function is used for both training (evaluation pass) and validation splits.
    """
    model.eval()
    logits_all: List[np.ndarray] = []
    scores_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    weights_all: List[np.ndarray] = []
    masses_all: List[np.ndarray] = []
    features_all: List[np.ndarray] = []
    for batch in tqdm(loader, leave=False, desc="collect"):
        x, y, w, _b, m = batch
        x_cpu = x  # preserve CPU copy for features
        x = x.to(device)
        logits, score = model(x)
        logits_all.append(to_numpy_detached(logits))
        scores_all.append(to_numpy_detached(score))
        labels_all.append(to_numpy_detached(y))
        weights_all.append(to_numpy_detached(w))
        masses_all.append(to_numpy_detached(m))
        features_all.append(to_numpy_detached(x_cpu))
    logits_np = np.concatenate(logits_all, axis=0)
    scores_np = np.concatenate(scores_all, axis=0)
    labels_np = np.concatenate(labels_all, axis=0)
    weights_np = np.concatenate(weights_all, axis=0)
    masses_np = np.concatenate(masses_all, axis=0)
    features_np = np.concatenate(features_all, axis=0)
    return {"logits": logits_np, "scores": scores_np, "labels": labels_np, "weights": weights_np, "masses": masses_np, "features": features_np}


def run_training_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, alpha: float) -> Dict[str, float]:
    """One training epoch with tqdm and timing.

    - Iterates over mini-batches, computes losses, and performs SGD updates.
    - Tracks running means for BCE and DisCo losses.
    - Returns a dictionary with averaged losses and wall-clock stats.
    """
    model.train()
    start = time.perf_counter()
    running_bce: List[float] = []
    running_disco: List[float] = []
    for batch in tqdm(loader, leave=False, desc="train"):
        features, labels, weights, binnums, masses = batch
        features = features.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        binnums = binnums.to(device)
        masses = masses.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss_total, loss_bce, loss_disco, _ = compute_losses_for_batch(
            model, (features, labels, weights, binnums, masses), alpha
        )
        loss_total.backward()
        optimizer.step()
        running_bce.append(float(loss_bce))
        running_disco.append(float(loss_disco))
    duration = time.perf_counter() - start
    return {
        "train_loss_bce": float(np.mean(running_bce)) if running_bce else float("nan"),
        "train_loss_disco": float(np.mean(running_disco)) if running_disco else float("nan"),
        "train_loss_total": (float(np.mean(running_bce)) + float(np.mean(running_disco))) if running_bce else float("nan"),
        "train_epoch_seconds": duration,
        "train_iterations_per_second": len(loader) / duration if duration > 0 else float("nan"),
    }


##### ABCD utilities (m_hat-based) ############################################################

def compute_abcd_metrics_with_mhat(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray, mhat: np.ndarray, target_eff: float, m_top: float = 172.5, delta_m: float = 35.0) -> Dict[str, Any]:
    """Compute ABCD metrics for a given signal efficiency using (x=score, y=m_hat).

    - Compute score cut achieving the target weighted signal efficiency.
    - Define SR mass window as central [0.3, 0.7] quantiles of signal m_hat.
    - Partition A/B/C/D and compute background yields and the ABCD prediction.
    - Return closure metrics and transfer factors.
    """
    cut = weighted_threshold_at_signal_eff(scores, labels, weights, target_eff)
    s_mask = labels > 0.5
    b_mask = ~s_mask
    if not np.any(s_mask) or math.isnan(cut):
        return {"cut": float("nan")}
    # Fixed SR using delta_m around m_top: m_hat >= 1 - delta_m/m_top
    mhat_thr = 1.0 - float(delta_m) / float(m_top)
    in_sr = (mhat >= mhat_thr)
    pass_score = scores >= cut
    A = pass_score & in_sr
    B = pass_score & (~in_sr)
    C = (~pass_score) & in_sr
    D = (~pass_score) & (~in_sr)
    w = weights.astype(float)
    # Background-only sums
    A_bg = float(np.sum(w[A & b_mask]))
    B_bg = float(np.sum(w[B & b_mask]))
    C_bg = float(np.sum(w[C & b_mask]))
    D_bg = float(np.sum(w[D & b_mask]))
    eps = 1e-12
    pred_bg = B_bg * C_bg / (D_bg + eps)
    closure = pred_bg / (A_bg + eps)
    tf_BD = B_bg / (D_bg + eps)
    tf_CD = C_bg / (D_bg + eps)

    # Signal contamination per region δ_i = N_S(i) / N_B(i)
    A_sig = float(np.sum(w[A & s_mask]))
    B_sig = float(np.sum(w[B & s_mask]))
    C_sig = float(np.sum(w[C & s_mask]))
    D_sig = float(np.sum(w[D & s_mask]))
    delta_A = A_sig / (A_bg + eps)
    delta_B = B_sig / (B_bg + eps)
    delta_C = C_sig / (C_bg + eps)
    delta_D = D_sig / (D_bg + eps)
    # Normalized (relative) contamination
    delta_rel = (delta_B + delta_C - delta_D) / (delta_A + eps)
    # Classification side summaries at this cut
    cls = compute_bg_eff_rej_contamination(scores, labels, weights, cut)
    return {
        "cut": cut,
        "mhat_threshold": mhat_thr,
        "A_bg": A_bg,
        "B_bg": B_bg,
        "C_bg": C_bg,
        "D_bg": D_bg,
        "predicted_bg": pred_bg,
        "closure": closure,
        "tf_B_over_D": tf_BD,
        "tf_C_over_D": tf_CD,
        "cls_at_cut": cls,
        "delta_A": delta_A,
        "delta_B": delta_B,
        "delta_C": delta_C,
        "delta_D": delta_D,
        "delta_rel": delta_rel,
        "region_masks": {"A": A, "B": B, "C": C, "D": D},
    }


def make_scatter_score_vs_mhat(x_scores: np.ndarray, y_mhat: np.ndarray, labels: np.ndarray, out: Path, title: str, max_points: int = 100000, score_cuts: Optional[Dict[str, float]] = None, mhat_threshold: Optional[float] = None) -> None:
    """Plot a downsampled scatter of model score vs m_hat, colored by class."""
    n = len(x_scores)
    if n > max_points:
        rng = np.random.default_rng(1337)
        select = rng.choice(n, size=max_points, replace=False)
    else:
        select = np.arange(n)
    s = x_scores[select]
    m = y_mhat[select]
    y = labels[select]
    plt.figure(figsize=(8, 6))
    plt.scatter(s[y < 0.5], m[y < 0.5], s=2, alpha=0.3, label="background")
    plt.scatter(s[y > 0.5], m[y > 0.5], s=2, alpha=0.3, label="signal")
    plt.xlabel("Classifier score (signal prob)")
    plt.ylabel("m_hat = 1 - |m - m_top|/m_top")
    # Optional guideline lines
    if mhat_threshold is not None:
        plt.axhline(mhat_threshold, color="red", linestyle=":", linewidth=1.5, label="m_hat SR threshold")
    if score_cuts is not None:
        for k, cut in score_cuts.items():
            plt.axvline(cut, linestyle=(0,(4,2)), linewidth=1.2, label=f"score cut @ εS={k}%")
    plt.title(title + "\nGuides: horizontal = m_hat SR threshold; verticals = score cuts at εS=10/30/50%")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def make_hist_with_estimated_background(x: np.ndarray, labels: np.ndarray, weights: np.ndarray, regions: Dict[str, np.ndarray], bins: int, out: Path, title: str, x_label: Optional[str] = None, x_range: Optional[Tuple[float, float]] = None, log_scale: bool = False) -> None:
    """Stacked histogram of signal and true background in SR with a single estimated background overlay and a ratio panel.

    - SR = region A mask (score pass & in SR window).
    - True SR background shape plotted as histogram.
    - Estimated background overlay uses region C scaled by B/D (C-shape × B/D).
    - Bottom panel shows ratio = (estimated background) / (true background).
    - If log_scale is True, top panel y-axis uses log scale.
    """
    A = regions["A"]
    B = regions["B"]
    C = regions["C"]
    D = regions["D"]
    b_mask = labels < 0.5
    s_mask = labels > 0.5
    w = weights.astype(float)

    # Yield summaries for transfer factors
    B_bg = float(np.sum(w[B & b_mask]))
    C_bg = float(np.sum(w[C & b_mask]))
    D_bg = float(np.sum(w[D & b_mask]))
    eps = 1e-12
    tf_B_over_D = B_bg / (D_bg + eps)

    # Histogram range and bins
    if x_range is None:
        x_range = (float(np.min(x)), float(np.max(x)))
    nb = bins

    # Compute histograms
    b_counts, b_edges = np.histogram(x[A & b_mask], bins=nb, range=x_range, weights=w[A & b_mask])
    s_counts, _ = np.histogram(x[A & s_mask], bins=nb, range=x_range, weights=w[A & s_mask])
    bC_counts, _ = np.histogram(x[C & b_mask], bins=nb, range=x_range, weights=w[C & b_mask])
    est_from_C = tf_B_over_D * bC_counts

    centers = 0.5 * (b_edges[:-1] + b_edges[1:])
    width = (b_edges[1] - b_edges[0])

    # Figure with ratio panel
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3.0, 1.0], hspace=0.05)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    # Top panel: stacked S+B and estimated background overlay
    ax_top.bar(centers, b_counts, width=0.9 * width, color="tab:blue", alpha=0.5, label="True background (SR)")
    ax_top.bar(centers, s_counts, bottom=b_counts, width=0.9 * width, color="tab:orange", alpha=0.5, label="Signal (SR)")
    ax_top.step(centers, est_from_C, where="mid", color="green", lw=2, label="Estimated background (from C×B/D)")
    if log_scale:
        ax_top.set_yscale("log")
    ax_top.set_ylabel("Weighted events")
    ax_top.set_title(title + "\nABCD regions: A=pass&SR, B=pass&SB, C=fail&SR, D=fail&SB; TF: C×B/D")
    ax_top.legend(loc="best")

    # Bottom panel: ratio
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
    """Plot distributions for both raw logits (per class) and softmax score (class-1).

    Saves into subdirectories: out_root_dir/logit0, out_root_dir/logit1, out_root_dir/score.
    """
    # Logits per output (2) by class
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
    # Scores
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

    # Device selection
    device = setup_device(args.gpunum)
    print(f"Using device: {device}")

    # Output directories
    smear_sigma = float(args.smear)
    run_label = args.run_label or f"smear{int(smear_sigma)}"
    dirs = prepare_output_dirs(Path(args.output_root), run_label)

    # Data loading
    payload = load_and_preprocess(smear_sigma=smear_sigma, test=args.test)
    trainset = payload["trainset"] ; valset = payload["valset"] ; testset = payload["testset"]

    # Dataloaders (preserve batch size & shuffle semantics; eval uses shuffle=False)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=int(args.num_workers), persistent_workers=(int(args.num_workers) > 0))
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=int(args.num_workers), persistent_workers=(int(args.num_workers) > 0))
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=int(args.num_workers), persistent_workers=(int(args.num_workers) > 0))

    # Dataset summaries saved to history dir
    print_dataset_summary_split(
        "train", payload["train_features_np"], payload["train_labels_np"], payload["mhat_train"], dirs["history"]
    )
    print_dataset_summary_split(
        "val", payload["val_features_np"], payload["val_labels_np"], payload["mhat_val"], dirs["history"]
    )
    print_dataset_summary_split(
        "test", payload["test_features_np"], payload["test_labels_np"], payload["mhat_test"], dirs["history"]
    )

    # Model & optimizer
    model = SingleDiscoModel(
        n_features=11, n_classes=2,
        qnn_n_qubits=int(args.qnn_n_qubits),
        qnn_hidden_dim=int(args.qnn_hidden_dim),
        qnn_vqc_depth=int(args.qnn_vqc_depth),
        qnn_n_outputs=int(args.qnn_n_outputs),
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

    # Prepare a CSV-like plain log (legacy compatibility)
    legacy_log = open(dirs["history"] / args.logfile, "a")

    # Metrics that we track across epochs
    history_rows: List[Dict[str, Any]] = []

    # Training epochs
    for epoch in range(start_epoch, int(args.epochs)):
        print(f"Epoch {epoch:03d}")

        # Train and time
        train_stats = run_training_epoch(model, train_loader, optimizer, device, alpha=float(args.alpha))

        # Collect full arrays for train/val (no grad) for diagnostics
        train_arrays = collect_epoch_arrays(model, DataLoader(trainset, batch_size=args.batch_size, shuffle=False, pin_memory=True), device)
        val_arrays = collect_epoch_arrays(model, val_loader, device)

        # Compute validation losses using a pass over val loader for BCE/DisCo breakdown (no grad)
        model.eval()
        val_bce_losses: List[float] = []
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
                loss_total, loss_bce, loss_disco, _ = compute_losses_for_batch(
                    model, (features, labels, weights, binnums, masses), alpha=float(args.alpha)
                )
                val_bce_losses.append(float(loss_bce))
                val_disco_losses.append(float(loss_disco))
        val_duration = time.perf_counter() - start_val
        val_stats = {
            "val_loss_bce": float(np.mean(val_bce_losses)) if val_bce_losses else float("nan"),
            "val_loss_disco": float(np.mean(val_disco_losses)) if val_disco_losses else float("nan"),
            "val_loss_total": (float(np.mean(val_bce_losses)) + float(np.mean(val_disco_losses))) if val_bce_losses else float("nan"),
            "val_epoch_seconds": val_duration,
        }

        # Compute classification metrics (AUC, confusion) and background metrics at 10/30/50% for both splits
        split_metrics: Dict[str, Dict[str, Any]] = {}
        for split_name, arrays, mhat in (
            ("train", train_arrays, payload["mhat_train"]),
            ("val", val_arrays, payload["mhat_val"]),
        ):
            scores = arrays["scores"].astype(np.float64)
            labels = arrays["labels"].astype(np.float64)
            weights = arrays["weights"].astype(np.float64)
            logits = arrays["logits"].astype(np.float64)
            # ROC AUC (fast numpy-based)
            auc = compute_auc(scores, labels, weights)
            # Confusion at 0.5 and store cm-derived metrics
            cm_stats = weighted_confusion_stats(labels, scores, weights, threshold=0.5)
            cm = np.array([[cm_stats["tn"], cm_stats["fp"]], [cm_stats["fn"], cm_stats["tp"]]])
            acc = cm_stats["accuracy"]; prec = cm_stats["precision"]; rec = cm_stats["recall"]; f1 = cm_stats["f1"]
            # Working-point metrics
            cuts = {
                "10": weighted_threshold_at_signal_eff(scores, labels, weights, 0.10),
                "30": weighted_threshold_at_signal_eff(scores, labels, weights, 0.30),
                "50": weighted_threshold_at_signal_eff(scores, labels, weights, 0.50),
            }
            bgeff_rej_cont: Dict[str, Dict[str, float]] = {}
            for k, cut in cuts.items():
                if not math.isnan(cut):
                    bgeff_rej_cont[k] = compute_bg_eff_rej_contamination(scores, labels, weights, cut)
                else:
                    bgeff_rej_cont[k] = {"b_eff": float("nan"), "b_rej": float("nan"), "sig_contamination": float("nan")}
            # ABCD diagnostics with m_hat
            abcd_points: Dict[str, Any] = {}
            for k, eff in ("10", 0.10), ("30", 0.30), ("50", 0.50):
                abcd_points[k] = compute_abcd_metrics_with_mhat(scores, labels, weights, mhat, eff, m_top=172.5, delta_m=35.0)
            # JSD vs R (mass sculpting metric) using m_hat as the mass-like y-axis here
            # Build arrays for JSDvsR wrapper
            s_mask = labels > 0.5
            jsd_vs_r = {}
            for eff in (10, 30, 50):
                out = JSDvsR(sigscore=scores[s_mask], bgscore=scores[~s_mask], bgmass=mhat[~s_mask],
                             sigweights=weights[s_mask], bgweights=weights[~s_mask], sigeff=eff,
                             minmass=float(np.min(mhat)), maxmass=float(np.max(mhat)))
                jsd_vs_r[str(eff)] = {"b_rej": out[0], "inv_jsd": out[1], "jsd": (1.0 / out[1] if out[1] != 0 else float("inf"))}

            # Classification metrics at working-point thresholds (εS=10/30/50)
            metrics_wp: Dict[str, Dict[str, float]] = {}
            for k, cut in cuts.items():
                if not math.isnan(cut):
                    wp_stats = weighted_confusion_stats(labels, scores, weights, threshold=float(cut))
                    metrics_wp[k] = {
                        "accuracy": wp_stats["accuracy"],
                        "precision": wp_stats["precision"],
                        "recall": wp_stats["recall"],
                        "f1": wp_stats["f1"],
                    }
                else:
                    metrics_wp[k] = {"accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

            split_metrics[split_name] = {
                "auc": auc,
                "confusion": cm,
                "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
                "metrics_wp": metrics_wp,
                "bgeff_rej_cont": bgeff_rej_cont,
                "abcd": abcd_points,
                "jsd_vs_r": jsd_vs_r,
                "logits": logits,
                "scores": scores,
                "labels": labels,
                "weights": weights,
                "mhat": mhat,
            }

        # Produce and save plots for this epoch
        # 1) ROC curves
        for split_name in ("train", "val"):
            sm = split_metrics[split_name]
            split_dir = "train" if split_name == "train" else "valid"
            # Build ROC points from thresholds (approximate via cumulative)
            # For plotting we use sklearn-like curve with manual bins
            s = sm["scores"]; y = sm["labels"]; w = sm["weights"]
            # Sort by descending score
            order = np.argsort(-s)
            s_sorted, y_sorted, w_sorted = s[order], y[order], w[order]
            # Cumulative sums
            s_mask = y_sorted > 0.5
            b_mask = ~s_mask
            # Weighted totals
            S = np.sum(w_sorted[s_mask]) + 1e-12
            B = np.sum(w_sorted[b_mask]) + 1e-12
            tpr = np.cumsum(w_sorted * s_mask) / S
            fpr = np.cumsum(w_sorted * b_mask) / B
            out_dir_roc = dirs["plots_classification"] / split_dir / "roc"
            ensure_dir(out_dir_roc)
            out_path = out_dir_roc / f"roc_{split_name}_epoch_{epoch:03d}.png"
            plot_roc(fpr, tpr, sm["auc"], out_path, f"ROC ({split_name}) epoch {epoch}")

        # 2) Confusion matrices (at 0.5 threshold)
        for split_name in ("train", "val"):
            split_dir = "train" if split_name == "train" else "valid"
            cm = split_metrics[split_name]["confusion"]
            # Score 0.5
            out_dir_05_conf = dirs["plots_classification"] / split_dir / "Score0p5" / "confusion"
            out_dir_05_norm = dirs["plots_classification"] / split_dir / "Score0p5" / "confusion_normalized"
            ensure_dir(out_dir_05_conf)
            ensure_dir(out_dir_05_norm)
            out_path = out_dir_05_conf / f"confusion_{split_name}_epoch_{epoch:03d}.png"
            plot_confusion(cm, out_path, f"Confusion ({split_name}) @ threshold=0.5 epoch {epoch}")
            out_path_norm = out_dir_05_norm / f"confusion_{split_name}_norm_epoch_{epoch:03d}.png"
            plot_confusion(cm, out_path_norm, f"Confusion normalized ({split_name}) @ threshold=0.5 epoch {epoch}", normalize=True)
            # Confusions at working points (εS=10/30/50)
            for wp, cut in ( ("10", split_metrics[split_name]["abcd"]["10"].get("cut", float("nan"))),
                             ("30", split_metrics[split_name]["abcd"]["30"].get("cut", float("nan"))),
                             ("50", split_metrics[split_name]["abcd"]["50"].get("cut", float("nan"))) ):
                if math.isnan(cut):
                    continue
                # Recompute confusion for this cut from arrays
                s = split_metrics[split_name]["scores"]; y = split_metrics[split_name]["labels"]; w = split_metrics[split_name]["weights"]
                cms = weighted_confusion_stats(y, s, w, threshold=float(cut))
                cm_wp = np.array([[cms["tn"], cms["fp"]], [cms["fn"], cms["tp"]]])
                out_dir_wp_conf = dirs["plots_classification"] / split_dir / f"WP{wp}" / "confusion"
                out_dir_wp_norm = dirs["plots_classification"] / split_dir / f"WP{wp}" / "confusion_normalized"
                ensure_dir(out_dir_wp_conf)
                ensure_dir(out_dir_wp_norm)
                out_path_wp = out_dir_wp_conf / f"confusion_{split_name}_wp{wp}_epoch_{epoch:03d}.png"
                plot_confusion(cm_wp, out_path_wp, f"Confusion ({split_name}) @ εS={wp}% (threshold={cut:.4f}) epoch {epoch}")
                out_path_wp_norm = out_dir_wp_norm / f"confusion_{split_name}_wp{wp}_norm_epoch_{epoch:03d}.png"
                plot_confusion(cm_wp, out_path_wp_norm, f"Confusion normalized ({split_name}) @ εS={wp}% (threshold={cut:.4f}) epoch {epoch}", normalize=True)

        # 3) Logit and score distributions
        for split_name in ("train", "val"):
            sm = split_metrics[split_name]
            split_dir = "train" if split_name == "train" else "valid"
            dists_root = dirs["plots_classification"] / split_dir
            ensure_dir(dists_root)
            plot_logit_and_score_distributions(sm["logits"], sm["scores"], sm["labels"], dists_root, f"dists_{split_name}_epoch_{epoch:03d}", f"{split_name}")

        # 4) ABCD scatter plots (score vs m_hat)
        for split_name in ("train", "val"):
            sm = split_metrics[split_name]
            split_dir = "train" if split_name == "train" else "valid"
            out_dir_scatter = dirs["plots_abcd"] / split_dir / "scatter_score_vs_mhat"
            ensure_dir(out_dir_scatter)
            out_path = out_dir_scatter / f"scatter_score_vs_mhat_{split_name}_epoch_{epoch:03d}.png"
            # Gather score cuts and m_hat SR threshold
            score_cuts = {
                "10": split_metrics[split_name]["abcd"]["10"].get("cut", float("nan")),
                "30": split_metrics[split_name]["abcd"]["30"].get("cut", float("nan")),
                "50": split_metrics[split_name]["abcd"]["50"].get("cut", float("nan")),
            }
            mhat_thr = split_metrics[split_name]["abcd"]["30"].get("mhat_threshold", None)
            make_scatter_score_vs_mhat(
                sm["scores"], sm["mhat"], sm["labels"], out_path,
                f"score vs m_hat ({split_name}) epoch {epoch}",
                max_points=int(args.max_points_scatter), score_cuts=score_cuts, mhat_threshold=mhat_thr
            )

        # 5) Feature and overlay histograms
        #    Produce overlays for m_hat and score at εS=10,30,50; linear/log with ratio; per-split subdirectories
        for split_name in ("train", "val"):
            sm = split_metrics[split_name]
            split_dir = "train" if split_name == "train" else "valid"
            # Overlays for m_hat and score
            for wp in ("10", "30", "50"):
                abcd_wp = sm["abcd"][wp]
                regions = abcd_wp.get("region_masks", None)
                if regions is None:
                    continue
                # m_hat overlay per working point: linear and log
                for scale in ("linear", "log"):
                    out_dir_mhat = dirs["plots_abcd"] / split_dir / "hist_mhat" / scale / f"WP{wp}"
                    ensure_dir(out_dir_mhat)
                    out_path_mhat = out_dir_mhat / f"hist_mhat_overlay_wp{wp}_{split_name}_epoch_{epoch:03d}.png"
                    make_hist_with_estimated_background(
                        sm["mhat"], sm["labels"], sm["weights"], regions, bins=60, out=out_path_mhat,
                        title=f"m_hat overlays ({split_name}) @ εS={wp}% epoch {epoch}",
                        x_label=r"$\hat{m} = 1 - (|m_{jet} - m_{top}|/m_{top})$", x_range=(0.7, 1.1), log_scale=(scale == "log")
                    )
                # score overlay per working point: linear and log
                for scale in ("linear", "log"):
                    out_dir_score = dirs["plots_abcd"] / split_dir / "hist_score_overlay" / scale / f"WP{wp}"
                    ensure_dir(out_dir_score)
                    out_path_score = out_dir_score / f"hist_score_overlay_wp{wp}_{split_name}_epoch_{epoch:03d}.png"
                    make_hist_with_estimated_background(
                        sm["scores"], sm["labels"], sm["weights"], regions, bins=60, out=out_path_score,
                        title=f"score overlays ({split_name}) @ εS={wp}% epoch {epoch}",
                        x_label="model score", x_range=(-0.1, 1.1), log_scale=(scale == "log")
                    )

            # Plain 1D histograms for m_hat (signal vs background) into per-feature subdir
            feat_mhat_dir = dirs["plots_features"] / split_dir / "m_hat"
            ensure_dir(feat_mhat_dir)
            plt.figure(figsize=(8,5))
            plt.hist(sm["mhat"][sm["labels"] < 0.5], bins=60, alpha=0.5, label="background", density=False)
            plt.hist(sm["mhat"][sm["labels"] > 0.5], bins=60, alpha=0.5, label="signal", density=False)
            plt.xlabel("m_hat"); plt.ylabel("events"); plt.title(f"m_hat distribution ({split_name}) epoch {epoch}"); plt.legend()
            plt.tight_layout(); plt.savefig(feat_mhat_dir / f"mhat_hist_{split_name}_epoch_{epoch:03d}.png"); plt.close()

        # Use collected features for plotting per-epoch simple histograms (already gathered above)
        train_x_plot = train_arrays.get("features")
        train_y_plot = train_arrays.get("labels")
        val_x_plot = val_arrays.get("features")
        val_y_plot = val_arrays.get("labels")
        # Plot 1D histograms for each of the 11 input features
        def plot_feature_hists(x: np.ndarray, y: np.ndarray, split: str) -> None:
            nfeat = x.shape[1]
            for i in range(nfeat):
                plt.figure(figsize=(8,5))
                plt.hist(x[y < 0.5, i], bins=60, alpha=0.5, label="background", density=False)
                plt.hist(x[y > 0.5, i], bins=60, alpha=0.5, label="signal", density=False)
                name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature[{i}]"
                safe = sanitize_name(name)
                plt.xlabel(f"{name} (scaled)"); plt.ylabel("events"); plt.title(f"{name} distribution ({split}) epoch {epoch}"); plt.legend()
                split_dir = "train" if split == "train" else "valid"
                feat_dir = dirs["plots_features"] / split_dir / safe
                ensure_dir(feat_dir)
                plt.tight_layout(); plt.savefig(feat_dir / f"feature_{safe}_hist_{split}_epoch_{epoch:03d}.png"); plt.close()
        plot_feature_hists(train_x_plot, train_y_plot, "train")
        plot_feature_hists(val_x_plot, val_y_plot, "val")

        # On first epoch, also create one-time feature histograms over the full dataset using names
        if epoch == start_epoch:
            plot_feature_distributions_once(
                payload["train_features_np"], payload["val_features_np"], payload["test_features_np"],
                payload["train_labels_np"], payload["val_labels_np"], payload["test_labels_np"],
                FEATURE_NAMES, dirs["plots_features"]
            )

        # Track per-epoch summary row
        row: Dict[str, Any] = {
            "epoch": epoch,
            **train_stats,
            **val_stats,
            "val_auc": split_metrics["val"]["auc"],
            "train_auc": split_metrics["train"]["auc"],
        }
        # Include ABCD metrics and background eff/rej at working points
        for split_name in ("train", "val"):
            bgr = split_metrics[split_name]["bgeff_rej_cont"]
            for wp in ("10", "30", "50"):
                row[f"{split_name}_b_eff_{wp}"] = bgr[wp]["b_eff"]
                row[f"{split_name}_b_rej_{wp}"] = bgr[wp]["b_rej"]
                row[f"{split_name}_sig_cont_{wp}"] = bgr[wp]["sig_contamination"]
            # Store ABCD metrics and score cuts for all WPs (10/30/50)
            for wp in ("10", "30", "50"):
                ab = split_metrics[split_name]["abcd"][wp]
                row[f"{split_name}_score_cut_wp{wp}"] = ab.get("cut", float("nan"))
                row[f"{split_name}_mhat_threshold"] = ab.get("mhat_threshold", float("nan"))
                row[f"{split_name}_closure_{wp}"] = ab.get("closure", float("nan"))
                row[f"{split_name}_tf_B_over_D_{wp}"] = ab.get("tf_B_over_D", float("nan"))
                row[f"{split_name}_tf_C_over_D_{wp}"] = ab.get("tf_C_over_D", float("nan"))
                j = split_metrics[split_name]["jsd_vs_r"][wp]
                row[f"{split_name}_inv_jsd_{wp}"] = j["inv_jsd"]
                row[f"{split_name}_jsd_{wp}"] = j["jsd"]
                row[f"{split_name}_b_rej_jsd_{wp}"] = j["b_rej"]
                # Signal contamination metrics per region and normalized
                row[f"{split_name}_deltaA_{wp}"] = ab.get("delta_A", float("nan"))
                row[f"{split_name}_deltaB_{wp}"] = ab.get("delta_B", float("nan"))
                row[f"{split_name}_deltaC_{wp}"] = ab.get("delta_C", float("nan"))
                row[f"{split_name}_deltaD_{wp}"] = ab.get("delta_D", float("nan"))
                row[f"{split_name}_delta_rel_{wp}"] = ab.get("delta_rel", float("nan"))
            # Accuracy-style metrics
            m = split_metrics[split_name]["metrics"]
            row[f"{split_name}_accuracy"] = m["accuracy"]
            row[f"{split_name}_precision"] = m["precision"]
            row[f"{split_name}_recall"] = m["recall"]
            row[f"{split_name}_f1"] = m["f1"]
            # Add WP classification metrics
            for wp in ("10", "30", "50"):
                mwp = split_metrics[split_name]["metrics_wp"][wp]
                row[f"{split_name}_accuracy_wp{wp}"] = mwp["accuracy"]
                row[f"{split_name}_precision_wp{wp}"] = mwp["precision"]
                row[f"{split_name}_recall_wp{wp}"] = mwp["recall"]
                row[f"{split_name}_f1_wp{wp}"] = mwp["f1"]

        history_rows.append(row)

        # Save per-epoch checkpoint with rich payload
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history_rows,
        }
        ckpt_path = dirs["checkpoints"] / f"epoch_{epoch:03d}.pth"
        torch.save(ckpt, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

        # Optionally persist arrays needed to fully reproduce confusion matrices and ROC offline
        if args.save_epoch_arrays:
            npz_path_train = dirs["history_arrays"] / f"train_epoch_{epoch:03d}.npz"
            npz_path_val = dirs["history_arrays"] / f"val_epoch_{epoch:03d}.npz"
            ensure_dir(dirs["history_arrays"])
            np.savez_compressed(npz_path_train, scores=train_arrays["scores"], labels=train_arrays["labels"], weights=train_arrays["weights"], masses=train_arrays["masses"], features=train_arrays["features"]) 
            np.savez_compressed(npz_path_val, scores=val_arrays["scores"], labels=val_arrays["labels"], weights=val_arrays["weights"], masses=val_arrays["masses"], features=val_arrays["features"]) 
            print(f"Saved epoch arrays to {npz_path_train} and {npz_path_val}")

        # Write a concise line to legacy log CSV (key,value pairs akin to original state)
        legacy_log.write(f"epoch,{epoch},val_auc,{row['val_auc']},train_loss_total,{row['train_loss_total']},train_loss_bce,{row['train_loss_bce']},train_loss_disco,{row['train_loss_disco']}\n")
        legacy_log.flush()

        # Console summary
        print(
            f"AUC(val)={row['val_auc']:.3f} | loss(train,total/bce/disco)={row['train_loss_total']:.3f}/{row['train_loss_bce']:.3f}/{row['train_loss_disco']:.3f} | "
            f"loss(val,total/bce/disco)={row['val_loss_total']:.3f}/{row['val_loss_bce']:.3f}/{row['val_loss_disco']:.3f} | "
            f"closure30(val)={row['val_closure_30']:.3f} | invJSD30(val)={row['val_inv_jsd_30']:.3f} | b_rej30(val)={row['val_b_rej_30']:.2f}"
        )

    # Close log and write final history summary
    legacy_log.close()
    history_json = dirs["history"] / "history.json"
    save_json(history_rows, history_json)
    print(f"Saved history to {history_json}")

    # History plots (post-training): AUC vs epoch, losses, accuracy-style metrics, background metrics, JSD/invJSD
    import pandas as pd
    try:
        hist_df = pd.DataFrame(history_rows)
        x = hist_df["epoch"].values
        # History subdirectories
        hist_linear_dir = dirs["plots_history"] / "linear"
        hist_log_dir = dirs["plots_history"] / "log"
        ensure_dir(hist_linear_dir)
        ensure_dir(hist_log_dir)
        # AUC vs epoch
        plt.figure(figsize=(8,5))
        plt.plot(x, hist_df.get("train_auc", np.nan), label="train AUC")
        plt.plot(x, hist_df.get("val_auc", np.nan), label="val AUC")
        plt.xlabel("Epoch"); plt.ylabel("AUC"); plt.title("AUC vs epoch"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(hist_linear_dir / "auc_vs_epoch.png"); plt.yscale("log"); plt.savefig(hist_log_dir / "auc_vs_epoch_log.png"); plt.close()

        # Losses vs epoch (consistent styling: totals solid, BCE dashed, DisCo dotted; train blue, val orange)
        plt.figure(figsize=(10,6))
        # Totals
        plt.plot(x, hist_df.get("train_loss_total", np.nan), color="C0", linestyle="-", label="train total")
        plt.plot(x, hist_df.get("val_loss_total", np.nan), color="C1", linestyle="-", label="val total")
        # BCE
        plt.plot(x, hist_df.get("train_loss_bce", np.nan), color="C0", linestyle="--", label="train BCE")
        plt.plot(x, hist_df.get("val_loss_bce", np.nan), color="C1", linestyle="--", label="val BCE")
        # DisCo
        plt.plot(x, hist_df.get("train_loss_disco", np.nan), color="C0", linestyle=":", label="train DisCo")
        plt.plot(x, hist_df.get("val_loss_disco", np.nan), color="C1", linestyle=":", label="val DisCo")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss components vs epoch"); plt.legend(ncol=2); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(hist_linear_dir / "losses_vs_epoch.png"); plt.yscale("log"); plt.savefig(hist_log_dir / "losses_vs_epoch_log.png"); plt.close()

        # Accuracy/Precision/Recall/F1 vs epoch for train/val
        for metric in ("accuracy", "precision", "recall", "f1"):
            plt.figure(figsize=(8,5))
            plt.plot(x, hist_df.get(f"train_{metric}", np.nan), label=f"train {metric}")
            plt.plot(x, hist_df.get(f"val_{metric}", np.nan), label=f"val {metric}")
            plt.xlabel("Epoch"); plt.ylabel(metric); plt.title(f"{metric} vs epoch"); plt.legend(); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(hist_linear_dir / f"{metric}_vs_epoch.png"); plt.yscale("log"); plt.savefig(hist_log_dir / f"{metric}_vs_epoch_log.png"); plt.close()

        # Additional classification metrics vs epoch at working points 10/30/50 on single canvases
        for metric in ("accuracy", "precision", "recall", "f1"):
            plt.figure(figsize=(8,5))
            # Train lines in blue, val in orange; styles distinguish WPs
            for wp, style in [("10","--"),("30","-"),("50",":")]:
                plt.plot(x, hist_df.get(f"train_{metric}_wp{wp}", np.nan), color="C0", linestyle=style, label=f"train {metric} @ εS={wp}%")
                plt.plot(x, hist_df.get(f"val_{metric}_wp{wp}", np.nan), color="C1", linestyle=style, label=f"val {metric} @ εS={wp}%")
            plt.xlabel("Epoch"); plt.ylabel(metric); plt.title(f"{metric} vs epoch at εS=10/30/50% (train=blue, val=orange)")
            plt.legend(ncol=2); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(hist_linear_dir / f"{metric}_wps_vs_epoch.png"); plt.yscale("log"); plt.savefig(hist_log_dir / f"{metric}_wps_vs_epoch_log.png"); plt.close()

        # Background metrics vs epoch at three WPs for train/val on single canvases
        for mkey, ylabel in (("b_eff", "background efficiency ε_B"), ("b_rej", "background rejection 1/ε_B"), ("sig_cont", "signal contamination")):
            plt.figure(figsize=(8,5))
            for wp, style in [("10","--"),("30","-"),("50",":")]:
                plt.plot(x, hist_df.get(f"train_{mkey}_{wp}", np.nan), color="C0", linestyle=style, label=f"train @ εS={wp}%")
                plt.plot(x, hist_df.get(f"val_{mkey}_{wp}", np.nan), color="C1", linestyle=style, linewidth=2, label=f"val @ εS={wp}%")
            plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(f"{ylabel} vs epoch (εS=10/30/50%)")
            plt.legend(ncol=2); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(hist_linear_dir / f"{mkey}_vs_epoch.png"); plt.yscale("log"); plt.savefig(hist_log_dir / f"{mkey}_vs_epoch_log.png"); plt.close()

        # JSD, inverse JSD, and Closure vs epoch for 10/30/50 on single canvases (train & val)
        # JSD
        plt.figure(figsize=(8,5))
        for wp, style in [("10","--"),("30","-"),("50",":")]:
            plt.plot(x, hist_df.get(f"train_jsd_{wp}", np.nan), color="C0", linestyle=style, label=f"train JSD ({wp}%)")
            plt.plot(x, hist_df.get(f"val_jsd_{wp}", np.nan), color="C1",linestyle=style, linewidth=2, label=f"val JSD ({wp}%)")
        plt.xlabel("Epoch"); plt.ylabel("JSD"); plt.title("JSD vs epoch (εS=10/30/50%)")
        plt.legend(ncol=2); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(hist_linear_dir / "jsd_vs_epoch.png"); plt.yscale("log"); plt.savefig(hist_log_dir / "jsd_vs_epoch_log.png"); plt.close()

        # 1/JSD
        plt.figure(figsize=(8,5))
        for wp, style in [("10","--"),("30","-"),("50",":")]:
            plt.plot(x, hist_df.get(f"train_inv_jsd_{wp}", np.nan), color="C0", linestyle=style, label=f"train 1/JSD ({wp}%)")
            plt.plot(x, hist_df.get(f"val_inv_jsd_{wp}", np.nan), color="C1",linestyle=style, linewidth=2, label=f"val 1/JSD ({wp}%)")
        plt.xlabel("Epoch"); plt.ylabel("1/JSD"); plt.title("Inverse JSD vs epoch (εS=10/30/50%)")
        plt.legend(ncol=2); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(hist_linear_dir / "inv_jsd_vs_epoch.png"); plt.yscale("log"); plt.savefig(hist_log_dir / "inv_jsd_vs_epoch_log.png"); plt.close()

        # Closure
        plt.figure(figsize=(8,5))
        plt.axhspan(0.9, 1.1, color="grey", alpha=0.15, label="±10% band")
        for wp, style in [("10","--"),("30","-"),("50",":")]:
            plt.plot(x, hist_df.get(f"train_closure_{wp}", np.nan), color="C0", linestyle=style, label=f"train closure ({wp}%)")
            plt.plot(x, hist_df.get(f"val_closure_{wp}", np.nan), color="C1", linestyle=style, linewidth=2, label=f"val closure ({wp}%)")
        plt.xlabel("Epoch"); plt.ylabel("Closure N_pred/N_true"); plt.title("ABCD closure vs epoch (εS=10/30/50%)")
        plt.legend(ncol=2); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(hist_linear_dir / "closure_vs_epoch.png"); plt.yscale("log"); plt.savefig(hist_log_dir / "closure_vs_epoch_log.png"); plt.close()

        # Normalized signal contamination δ_rel vs epoch (εS=10/30/50%)
        plt.figure(figsize=(8,5))
        for wp, style in [("10","--"),("30","-"),("50",":")]:
            plt.plot(x, hist_df.get(f"train_delta_rel_{wp}", np.nan), color="C0", linestyle=style, label=f"train δ_rel ({wp}%)")
            plt.plot(x, hist_df.get(f"val_delta_rel_{wp}", np.nan), color="C1", linestyle=style, linewidth=2, label=f"val δ_rel ({wp}%)")
        plt.xlabel("Epoch"); plt.ylabel("δ_rel"); plt.title("Normalized signal contamination δ_rel vs epoch (εS=10/30/50%)")
        plt.legend(ncol=2); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(hist_linear_dir / "delta_rel_vs_epoch.png"); plt.yscale("log"); plt.savefig(hist_log_dir / "delta_rel_vs_epoch_log.png"); plt.close()

        # Per-region signal contamination δ_A/B/C/D vs epoch
        for region, key in [("A", "deltaA"), ("B", "deltaB"), ("C", "deltaC"), ("D", "deltaD")]:
            plt.figure(figsize=(8,5))
            for wp, style in [("10","--"),("30","-"),("50",":")]:
                plt.plot(x, hist_df.get(f"train_{key}_{wp}", np.nan), color="C0", linestyle=style, label=f"train δ_{region} ({wp}%)")
                plt.plot(x, hist_df.get(f"val_{key}_{wp}", np.nan), color="C1", linestyle=style, linewidth=2, label=f"val δ_{region} ({wp}%)")
            plt.xlabel("Epoch"); plt.ylabel(f"δ_{region}"); plt.title(f"Signal contamination δ_{region} vs epoch (εS=10/30/50%)")
            plt.legend(ncol=2); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(hist_linear_dir / f"delta_{region}_vs_epoch.png"); plt.yscale("log"); plt.savefig(hist_log_dir / f"delta_{region}_vs_epoch_log.png"); plt.close()
    except Exception as e:
        print(f"Failed to render history plots: {e}")

    # Final test-set evaluation (streaming, avoid memory blow-ups)
    print("Running final evaluation on the test set ...")
    test_arrays = collect_epoch_arrays(model, test_loader, device)
    test_scores = test_arrays["scores"].astype(np.float64)
    test_labels = test_arrays["labels"].astype(np.float64)
    test_weights = test_arrays["weights"].astype(np.float64)
    test_auc = compute_auc(test_scores, test_labels, test_weights)
    # ABCD test diagnostics at 30% using m_hat
    abcd_test = compute_abcd_metrics_with_mhat(test_scores, test_labels, test_weights, payload["mhat_test"], 0.30)
    jsd_test_out = JSDvsR(sigscore=test_scores[test_labels > 0.5], bgscore=test_scores[test_labels < 0.5], bgmass=payload["mhat_test"][test_labels < 0.5], sigweights=test_weights[test_labels > 0.5], bgweights=test_weights[test_labels < 0.5], sigeff=30, minmass=float(np.min(payload["mhat_test"])), maxmass=float(np.max(payload["mhat_test"])) )
    summary = {
        "test_auc": test_auc,
        "test_abcd_30": {
            "closure": abcd_test.get("closure", float("nan")),
            "b_rej": abcd_test.get("cls_at_cut", {}).get("b_rej", float("nan")),
        },
        "test_jsd_vs_r_30": {"b_rej": jsd_test_out[0], "inv_jsd": jsd_test_out[1]},
    }
    save_json(summary, dirs["history"] / "test_evaluation.json")
    print("Done.")


if __name__ == "__main__":
    main()


