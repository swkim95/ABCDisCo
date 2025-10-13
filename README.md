# ABCDisCo (Automating the ABCD Method)

This repository accompanies the studies reported in **T. Aarrestad *et al.*, "ABCDnn: ABCDisCo" (*Eur. Phys. J. C* **81**, 1003, 2021, [arXiv:2007.14400](https://arxiv.org/abs/2007.14400))**. It contains PyTorch implementations of the Double-DisCo decorrelation strategy for semi-supervised anomaly detection in collider data, along with utilities to port the workflow to Quantum Machine Learning (QML) backends using PennyLane.

## Repository layout

- `ABCD_topjets_HLF_DD.py`, `model_ABCD_2NN.py`, `data_loader.py`, `networks.py`, `disco.py`, `evaluation.py`: reference scripts used to produce the Double-DisCo baselines published in the paper.
- `topsample_*_tau.dat.gz`: reduced CMS top-tagging high-level feature (HLF) samples used throughout the examples.
- `notebook/ABCDisCo_single_disco_tutorial.ipynb`: single-network DisCo walkthrough that reproduces `ABCD_topjets_HLF_mD.py` before introducing mass decorrelation with optional PennyLane backends.
- `notebook/ABCDisCo_tutorial.ipynb`: double-network DisCo tutorial that mirrors `ABCD_topjets_HLF_DD.py` with dual heads, mutual decorrelation, and extended diagnostics.

## Running the tutorial notebooks

1. **Install dependencies** (CPU-friendly defaults):
   ```bash
   pip install numpy pandas scikit-learn matplotlib tqdm
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   # optional backends
   pip install pennylane pennylane-lightning
   pip install pyhf
   ```
2. **Launch a notebook** from the repository root:
   ```bash
   # Single-DisCo baseline (mass decorrelation only)
   jupyter notebook notebook/ABCDisCo_single_disco_tutorial.ipynb

   # Double-DisCo (dual discriminants with mutual decorrelation)
   jupyter notebook notebook/ABCDisCo_tutorial.ipynb
   ```
3. **Execute cells sequentially.** Each tutorial documents where the code originates:
   - **Single-DisCo notebook** (`ABCDisCo_single_disco_tutorial.ipynb`) follows `ABCD_topjets_HLF_mD.py` (lines 69–126) and `model.py` (lines 24–170). It emphasises the single score vs. jet-mass decorrelation, JSD scans, and saving `abcdisco_single_disco_model.pt` / `abcdisco_single_disco_scores.parquet`.
   - **Double-DisCo notebook** (`ABCDisCo_tutorial.ipynb`) mirrors `ABCD_topjets_HLF_DD.py` (lines 69–126) and `model_ABCD_2NN.py` (lines 29–208), adds mutual decorrelation diagnostics, ABCD closure plots, and exports `abcdisco_double_disco_model.pt` / `abcdisco_double_disco_scores.parquet`.
4. **Adjust hyperparameters** via each configuration cell:
   - Set `FULL_DATASET = True` and `EPOCHS = 200` to match the paper-level statistics (requires multi-hour GPU/CPU time).
   - In the single DisCo notebook, scan `LAMBDA_MASS` between 50 and 400 to study mass sculpting vs. performance.
   - In the double DisCo notebook, tune `LAMBDA_MUTUAL` (50–200) and `LAMBDA_MASS` to balance mutual and mass decorrelation, as done in the reference sweeps.
   - Switch `BACKEND = "qml"` (after installing PennyLane) in either notebook to activate the PennyLane variational head via `qml.qnn.TorchLayer`.
5. **Persist artefacts.** Both notebooks write parquet score tables and PyTorch weight files for downstream ABCD or `pyhf` studies as noted above.

## Data and reproducibility notes

- The included `topsample_*_tau.dat.gz` files are extracted from CMS Run-2 simulations and already formatted with the 13 HLFs used in the Double-DisCo study. No additional preprocessing is required beyond the min–max scaling implemented in the notebook/script.
- Random seeds are fixed (1337) to ease reproducibility on both CPU and GPU. When running QML backends, set `qml.seed(SEED)` inside the PennyLane device for deterministic behaviour.
- When scaling to full statistics, prefer GPU execution for the classical baseline and PennyLane's `lightning.qubit` backend for QML experiments.

## Further reading

- **ABCDisCo / Double-DisCo:** T. Aarrestad *et al.*, *Eur. Phys. J. C* **81**, 1003 (2021), [arXiv:2007.14400](https://arxiv.org/abs/2007.14400).
- **Distance correlation decorrelation:** M. D. Andrews *et al.*, *Phys. Rev. D* **101**, 094004 (2020), [arXiv:1905.08628](https://arxiv.org/abs/1905.08628).
- **Quantum extensions:** Explore PennyLane tutorials on variational quantum classifiers for strategies to extend the provided QML backend.
