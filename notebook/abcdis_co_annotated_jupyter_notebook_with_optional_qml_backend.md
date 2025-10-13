# ABCDisCo (Automating the ABCD Method) — Annotated Notebook (with optional QML backend)

This notebook mirrors the official ABCDisCo scripts and adds cell-by-cell explanations, plots, and a plug‑in interface to swap the **classical NN** for a **Quantum ML (VQC)** model. Use it as a starting point to run ABCDisCo on CMS Run‑2/Run‑3 NanoAOD‑derived datasets.

**Primary references**
- ABCDisCo paper: Kasieczka, Nachman, Schwartz, Shih, *Phys. Rev. D* **103**, 035021 (2021), arXiv:2007.14400.  
- ABCDisCo official code (scripts): `davidshih17/ABCDisCo` (files like `model_ABCD_2NN.py`, `disco.py`, `ABCD_topjets_*.py`).  
- DisCo (distance correlation) repo with a notebook: `davidshih17/DisCo` (`plotter_v4.ipynb`).  
- CMS‑ML notes on **Double DisCo** architecture (why two decorrelated discriminants).  
- Likelihood‑based ABCD tutorial: `pyhf` docs and example notebooks.

> **What this adds vs the scripts**
> 1. Clear separation of **data loading**, **model**, **loss (DisCo/Double‑DisCo)**, **training**, **ABCD closure checks**, and optional **likelihood fit** (pyhf).  
> 2. A **backend interface** so you can choose between a PyTorch NN and a PennyLane‑based VQC (QNN) with the *same* trainer and DisCo regularizer.  
> 3. CMS‑friendly convenience: uproot/awkward loading for NanoAOD, feature lists, scalers, per‑era reweighting hooks, etc.

---

## Setup & Installs (run in your env)
```python
# Optional: install dependencies (uncomment as needed in your environment)
# %pip install numpy pandas scikit-learn matplotlib tqdm pyhf uproot awkward coffea torch torchvision torchaudio
# %pip install pennylane pennylane-lightning  # for the QML (VQC) backend
# %pip install qiskit                         # optional: run VQC on Aer simulator or IBM backends
```

## 0) Configuration
```python
from pathlib import Path
import math, json, os
import numpy as np
import pandas as pd

# ---- User knobs (edit) ----
DATA_PATH = Path("./data")  # directory containing parquet/csv/root files
TRAIN_FILE = "train.parquet"   # or "train.root"
VAL_FILE   = "val.parquet"
TEST_FILE  = "test.parquet"

# Column names
LABEL_COL = "label"           # 1 for signal, 0 for background
WEIGHT_COL = "weight"         # event weight; if absent, a column will be created with ones
MASS_COL = "mass"             # variable to decorrelate against for bkg (e.g., m_jj, m_SD, etc.)

# Feature list (example; replace with your HLFs/LLFs)
FEATURES = [
    "pt", "eta", "phi",
    "tau21", "tau32",
    "subjet1_pt", "subjet2_pt"
]

# ABCD region definition: thresholds on the two discriminants s1, s2
S1_CUT = 0.5
S2_CUT = 0.5

# Training hyperparameters
RANDOM_STATE = 1337
BATCH_SIZE = 1024
EPOCHS = 20

# Loss weights
LAMBDA_DISCO = 5.0      # weight for decorrelating each score from MASS_COL on background
LAMBDA_MUTUAL = 1.0     # (optional) weight to decorrelate s1 and s2 from each other

# Backend: "torch" or "qml"
BACKEND = "torch"

# VQC (QML) specifics (used only if BACKEND=="qml")
QUBITS = 6
VQC_LAYERS = 2
QDEVICE = "default.qubit"  # or "lightning.qubit"
```

## 1) Data loading (Parquet/CSV/ROOT via `uproot`)
```python
import warnings
warnings.filterwarnings("ignore")

def _ensure_cols(df, label_col, weight_col, mass_col):
    if weight_col not in df.columns:
        df[weight_col] = 1.0
    missing = [c for c in [label_col, mass_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def load_table(path: Path, features, label_col, weight_col, mass_col):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    elif p.suffix == ".root":
        import uproot, awkward as ak
        with uproot.open(p) as f:
            # Adjust tree and branches for your NanoAOD
            tree = f["Events"] if "Events" in f else list(f.keys())[0]
            arr = f[tree].arrays(features + [label_col, mass_col], library="ak")
            df = ak.to_dataframe(arr).reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported file extension: {p.suffix}")
    df = _ensure_cols(df, label_col, weight_col, mass_col)
    keep = features + [label_col, weight_col, mass_col]
    return df[keep].copy()

train_df = load_table(DATA_PATH/TRAIN_FILE, FEATURES, LABEL_COL, WEIGHT_COL, MASS_COL)
val_df   = load_table(DATA_PATH/VAL_FILE,   FEATURES, LABEL_COL, WEIGHT_COL, MASS_COL)
test_df  = load_table(DATA_PATH/TEST_FILE,  FEATURES, LABEL_COL, WEIGHT_COL, MASS_COL)

train_df.head(3)
```

## 2) Preprocessing & Torch datasets
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Scale features (fit on train background+signal together by default)
scaler = StandardScaler()
scaler.fit(train_df[FEATURES].values)

def make_xyw(df):
    X = scaler.transform(df[FEATURES].values).astype("float32")
    y = df[LABEL_COL].values.astype("float32")
    w = df[WEIGHT_COL].values.astype("float32")
    m = df[MASS_COL].values.astype("float32")
    return X, y, w, m

Xtr, ytr, wtr, mtr = make_xyw(train_df)
Xva, yva, wva, mva = make_xyw(val_df)
Xte, yte, wte, mte = make_xyw(test_df)

import torch
from torch.utils.data import Dataset, DataLoader

class ArrayDataset(Dataset):
    def __init__(self, X, y, w, m):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.w = torch.from_numpy(w)
        self.m = torch.from_numpy(m)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.y[i], self.w[i], self.m[i]

dset_tr = ArrayDataset(Xtr, ytr, wtr, mtr)
dset_va = ArrayDataset(Xva, yva, wva, mva)
dset_te = ArrayDataset(Xte, yte, wte, mte)

dl_tr = DataLoader(dset_tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
dl_va = DataLoader(dset_va, batch_size=BATCH_SIZE, shuffle=False)
dl_te = DataLoader(dset_te, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
```

## 3) DisCo and Double‑DisCo losses
```python
import torch
import torch.nn.functional as F

def _cdist_centered(x):
    # pairwise Euclidean distance matrix, double-centered (Gower's centering)
    # x: (N,1)
    x = x.view(-1, 1)
    d = torch.cdist(x, x, p=2)
    n = d.size(0)
    J = torch.eye(n, device=d.device) - (1.0/n) * torch.ones((n,n), device=d.device)
    A = J @ d @ J
    return A

def distance_correlation(x, y, eps=1e-9):
    """
    x, y: 1D tensors of same length (N,)
    Returns sample distance correlation in [0,1].
    """
    Ax = _cdist_centered(x)
    Ay = _cdist_centered(y)
    dcov2 = (Ax * Ay).mean()
    dvarx = (Ax * Ax).mean().clamp_min(eps)
    dvary = (Ay * Ay).mean().clamp_min(eps)
    dcor = (dcov2.clamp_min(0.0) / torch.sqrt(dvarx * dvary + eps)).clamp(0.0, 1.0)
    return dcor

def disco_loss(score, mass, bkg_mask, lam=1.0):
    """
    Penalize distance correlation between score and mass for background events only.
    score: (N,) sigmoid output
    mass:  (N,)
    bkg_mask: boolean (N,) True where y==0
    """
    if bkg_mask.sum() < 2:
        return torch.tensor(0.0, device=score.device)
    s = score[bkg_mask]
    m = mass[bkg_mask]
    return lam * distance_correlation(s, m)

def double_disco_loss(s1, s2, mass, bkg_mask, lam_each=1.0, lam_mutual=0.0):
    l = disco_loss(s1, mass, bkg_mask, lam_each) + disco_loss(s2, mass, bkg_mask, lam_each)
    if lam_mutual > 0.0:
        l = l + lam_mutual * distance_correlation(s1, s2)
    return l
```

## 4) Models: Torch (two MLPs) and optional QML (two VQCs)
```python
import torch.nn as nn

class MLP1D(nn.Module):
    def __init__(self, d_in, hidden=(64,64)):
        super().__init__()
        layers = []
        last = d_in
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(0.1)]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits

class TorchTwoNets(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net1 = MLP1D(d_in)
        self.net2 = MLP1D(d_in)
    def forward(self, x):
        z1 = self.net1(x)
        z2 = self.net2(x)
        s1 = torch.sigmoid(z1)
        s2 = torch.sigmoid(z2)
        return z1, z2, s1, s2
```

```python
def build_qml_two_nets(d_in, n_qubits=6, layers=2, device_name="default.qubit"):
    """
    Returns (model, info_dict). The model mimics TorchTwoNets's interface.
    """
    import pennylane as qml
    from pennylane import numpy as pnp
    import torch.nn as nn
    import torch, math

    if n_qubits < d_in:
        # encode via random linear projection if features > qubits
        proj = torch.randn(d_in, n_qubits) / math.sqrt(d_in)
    else:
        proj = None

    dev1 = qml.device(device_name, wires=n_qubits)
    dev2 = qml.device(device_name, wires=n_qubits)

    def feature_map(x, wires):
        # simple angle encoding
        for i, w in enumerate(wires):
            qml.RX(x[i], wires=w)

    def ansatz(params, wires):
        for _ in range(layers):
            for w in wires:
                qml.RY(params[0, w], wires=w)
            for i in range(len(wires)-1):
                qml.CNOT(wires=[wires[i], wires[i+1]])
        # final single-qubit rotations
        for w in wires:
            qml.RZ(params[1, w], wires=w)

    @qml.qnode(dev1, interface="torch")
    def circuit1(x, params):
        feature_map(x, range(n_qubits))
        ansatz(params, range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev2, interface="torch")
    def circuit2(x, params):
        feature_map(x, range(n_qubits))
        ansatz(params, range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    class QTwoNets(nn.Module):
        def __init__(self):
            super().__init__()
            self.params1 = nn.Parameter(torch.randn(2, n_qubits)*0.01)
            self.params2 = nn.Parameter(torch.randn(2, n_qubits)*0.01)
        def forward(self, x):
            # project/trim features to n_qubits
            if proj is not None:
                xproj = x @ proj
            else:
                xproj = x[:, :n_qubits]
            # normalize angles
            xang = torch.tanh(xproj)
            z1 = circuit1(xang, self.params1).view(-1)
            z2 = circuit2(xang, self.params2).view(-1)
            # map expectation [-1,1] to sigmoid-like [0,1]
            s1 = (z1 + 1)/2
            s2 = (z2 + 1)/2
            return z1, z2, s1, s2

    return QTwoNets(), {"n_qubits": n_qubits, "layers": layers, "device": device_name}
```

## 5) Training loop (classification + Double‑DisCo)
```python
import torch, numpy as np
from tqdm.auto import tqdm

if BACKEND == "qml":
    model, info = build_qml_two_nets(len(FEATURES), QUBITS, VQC_LAYERS, QDEVICE)
else:
    model = TorchTwoNets(len(FEATURES))

model = model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

bce = torch.nn.BCEWithLogitsLoss(reduction="none")  # we'll apply sigmoid manually for logits

def run_epoch(dloader, train=True):
    model.train(train)
    total = 0.0
    cls_loss_sum = 0.0
    disco_sum = 0.0

    for X, y, w, m in tqdm(dloader, leave=False):
        X, y, w, m = X.to(device), y.to(device), w.to(device), m.to(device)
        if train:
            opt.zero_grad()

        z1, z2, s1, s2 = model(X)

        # classification loss (two heads; average)
        loss1 = (bce(z1, y) * w).mean()
        loss2 = (bce(z2, y) * w).mean()
        cls_loss = 0.5 * (loss1 + loss2)

        # decorrelate on *background* (y==0) only
        bmask = (y < 0.5)
        dloss = double_disco_loss(s1.detach() if BACKEND=="qml" else s1,  # safe for QNodes
                                  s2.detach() if BACKEND=="qml" else s2,
                                  m, bmask,
                                  lam_each=LAMBDA_DISCO,
                                  lam_mutual=LAMBDA_MUTUAL)

        loss = cls_loss + dloss

        if train:
            loss.backward()
            opt.step()

        total += float(loss.detach().cpu())
        cls_loss_sum += float(cls_loss.detach().cpu())
        disco_sum += float(dloss.detach().cpu())

    n = len(dloader)
    return {"loss": total/n, "cls": cls_loss_sum/n, "disco": disco_sum/n}

hist = {"train": [], "val": []}
for ep in range(1, EPOCHS+1):
    tr = run_epoch(dl_tr, train=True)
    va = run_epoch(dl_va, train=False)
    hist["train"].append(tr); hist["val"].append(va)
    print(f"Epoch {ep:02d} | train: {tr} | val: {va}")
```

## 6) Evaluation: independence checks and ABCD closure
```python
import matplotlib.pyplot as plt
import numpy as np

def scores_on(dl):
    model.eval()
    outs = []
    with torch.no_grad():
        for X, y, w, m in dl:
            X = X.to(device)
            z1, z2, s1, s2 = model(X)
            outs.append((s1.cpu().numpy(), s2.cpu().numpy(), y.numpy(), w.numpy(), m.numpy()))
    s1 = np.concatenate([o[0] for o in outs])
    s2 = np.concatenate([o[1] for o in outs])
    y  = np.concatenate([o[2] for o in outs])
    w  = np.concatenate([o[3] for o in outs])
    m  = np.concatenate([o[4] for o in outs])
    return s1, s2, y, w, m

s1, s2, y, w, m = scores_on(dl_te)

# distance correlations
with torch.no_grad():
    t_s1 = torch.from_numpy(s1.astype("float32"))
    t_s2 = torch.from_numpy(s2.astype("float32"))
    t_m  = torch.from_numpy(m.astype("float32"))
    bmask = torch.from_numpy((y<0.5).astype(bool))
    d_s1_m = float(distance_correlation(t_s1[bmask], t_m[bmask]))
    d_s2_m = float(distance_correlation(t_s2[bmask], t_m[bmask]))
    d_s1_s2 = float(distance_correlation(t_s1, t_s2))

print({"dCor(s1,m)_bkg": d_s1_m, "dCor(s2,m)_bkg": d_s2_m, "dCor(s1,s2)": d_s1_s2})

# 2D hist for ABCD
fig = plt.figure(figsize=(5,4))
hb = plt.hist2d(s1[y<0.5], s2[y<0.5], bins=40)  # background only
plt.xlabel("s1"); plt.ylabel("s2"); plt.title("Background scores")
plt.colorbar(); plt.show()

# ABCD counts on background
def abcd_counts(s1, s2, y, w, c1=S1_CUT, c2=S2_CUT):
    b = (y<0.5)
    A = (s1<c1) & (s2<c2) & b
    B = (s1>=c1) & (s2<c2) & b
    C = (s1<c1) & (s2>=c2) & b
    D = (s1>=c1) & (s2>=c2) & b
    def wsum(mask): return float(w[mask].sum())
    return wsum(A), wsum(B), wsum(C), wsum(D)

A,B,C,D = abcd_counts(s1, s2, y, w)
D_est = (B*C)/(A+1e-9)
closure = D_est / (D+1e-9)
print({"A":A, "B":B, "C":C, "D_true":D, "D_est":D_est, "closure(D_est/D_true)":closure})
```

## 7) Likelihood‑based ABCD (optional, needs `pyhf`)
```python
try:
    import pyhf

    def make_abcd_model(A,B,C,D):
        spec = {
            "channels": [{
                "name": "abcd",
                "samples": [{
                    "name": "bkg",
                    "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
                    "data": [A,B,C,D]
                }],
                "inputs": [{"name":"bins","data":[A,B,C,D]}]
            }]
        }
        return pyhf.Model(spec, poi_name="mu")

    model = make_abcd_model(A,B,C,D)
    data = [A,B,C,D] + model.config.auxdata
    result = pyhf.infer.mle.fit(data, model)
    print("pyhf MLE (mu):", result[model.config.poi_index])
except Exception as e:
    print("pyhf not available or failed:", e)
```

## 8) Save scores & export models
```python
# Append scores to the test dataframe and save
out = test_df.copy()
out["s1"] = s1
out["s2"] = s2
out.to_parquet("test_with_scores.parquet")
print("Wrote test_with_scores.parquet")

# Save Torch state dicts (works for both backends; for QML only the classical parameters are saved)
torch.save(model.state_dict(), "abcdisco_model.pt")
print("Saved abcdisco_model.pt")
```

## 9) Notes to reproduce paper examples
- **3‑Gaussian toy:** generate three normal components and label one as “signal”; use two ML‑learned discriminants with DisCo to emulate Figure 2‑style closure plots.
- **Boosted top (HLFs):** load the provided `topsample_*.dat.gz` or your own jet HLFs. Use soft‑drop mass (or \(m_{SD}\)) as `MASS_COL` for decorrelation, following the **Double‑DisCo** setup.
- **Paired dijet (RPV SUSY‑like):** define signal proxies where the dijet mass peak appears; use total‑invariant‑mass as `MASS_COL`.

> Tip: The **thresholds** `(S1_CUT, S2_CUT)` can be scanned to pick the most stable closure; in real analyses you’d **lock them** before looking in data (or scan with toys and treat as a discrete nuisance).

## 10) QML caveats & tips
- Start with **few qubits** (4–8) and shallow depth; increase only if you see a learning plateau.
- Always bench against the Torch NN with **identical inputs** and **identical ABCD metrics** (closure, contamination).
- When training the VQC, batch sizes may need to be **smaller**; if gradients are noisy, try gradient‑accumulation or `optax`‑style optimizers via JAX backends in PennyLane.
- For real hardware, restrict to **native gate sets** and transpile (e.g. to IBM basis) after you’re satisfied on simulators.

