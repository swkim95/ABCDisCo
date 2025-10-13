# ABCDisCo (Automating the ABCD Method)

This repository accompanies the studies reported in **T. Aarrestad *et al.*, "ABCDnn: ABCDisCo" (*Eur. Phys. J. C* **81**, 1003, 2021, [arXiv:2007.14400](https://arxiv.org/abs/2007.14400))**. It contains PyTorch implementations of the Double-DisCo decorrelation strategy for semi-supervised anomaly detection in collider data, along with utilities to port the workflow to Quantum Machine Learning (QML) backends using PennyLane.

## Repository layout

- `ABCD_topjets_HLF_DD.py`, `model_ABCD_2NN.py`, `data_loader.py`, `networks.py`, `disco.py`, `evaluation.py`: reference scripts used to produce the Double-DisCo baselines published in the paper.
- `topsample_*_tau.dat.gz`: reduced CMS top-tagging high-level feature (HLF) samples used throughout the examples.
- `notebook/ABCDisCo_tutorial.ipynb`: end-to-end tutorial notebook that mirrors the training/validation pipeline while adding diagnostics and QML hooks.

## Running the tutorial notebook

1. **Install dependencies** (CPU-friendly defaults):
   ```bash
   pip install numpy pandas scikit-learn matplotlib tqdm scipy
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   # optional backends
   pip install pennylane pennylane-lightning
   pip install pyhf
   ```
2. **Launch the notebook** from the repository root:
   ```bash
   jupyter notebook notebook/ABCDisCo_tutorial.ipynb
   ```
3. **Execute cells sequentially.** Each section cites the corresponding reference script:
   - *Configuration & data loading* reproduce `ABCD_topjets_HLF_DD.py` (lines 69–129) and `data_loader.py` (lines 1–63), including the global min–max scaling applied to all HLFs.
   - *Model construction* wraps `networks.DNNclassifier` (lines 8–78) inside Torch and PennyLane backends, maintaining interface compatibility with the classical Double-DisCo training loop.
   - *Loss & training* adapt `model_ABCD_2NN.py` (lines 29–208) and re-use the distance-correlation penalties from `disco.py` (lines 14–118).
   - *Evaluation* reproduces the closure metrics from `evaluation.py` (lines 1–141), yielding ROC curves, ABCD closure scans, and Jensen–Shannon divergence versus background rejection.
4. **Adjust hyperparameters** via the exposed configuration cell:
   - Set `FULL_DATASET = True` and `EPOCHS = 200` to match the paper-level statistics (requires multi-hour GPU/CPU time).
   - Tune `LAMBDA_MUTUAL` in the range 50–200 and optionally `LAMBDA_MASS` to scan decorrelation strengths, as done in the reference sweeps.
   - Switch `BACKEND = "qml"` (after installing PennyLane) to activate the variational quantum circuit head built on `qml.qnn.TorchLayer`.
5. **Persist artefacts.** The notebook writes:
   - `abcdisco_double_disco_model.pt`: trained PyTorch (or hybrid) state dict.
   - `abcdisco_double_disco_scores.parquet`: inference scores with labels, weights, and jet masses for downstream ABCD or `pyhf` studies.

## Data and reproducibility notes

- The included `topsample_*_tau.dat.gz` files are extracted from CMS Run-2 simulations and already formatted with the 13 HLFs used in the Double-DisCo study. No additional preprocessing is required beyond the min–max scaling implemented in the notebook/script.
- Random seeds are fixed (1337) to ease reproducibility on both CPU and GPU. When running QML backends, set `qml.seed(SEED)` inside the PennyLane device for deterministic behaviour.
- When scaling to full statistics, prefer GPU execution for the classical baseline and PennyLane's `lightning.qubit` backend for QML experiments.

## Further reading

- **ABCDisCo / Double-DisCo:** T. Aarrestad *et al.*, *Eur. Phys. J. C* **81**, 1003 (2021), [arXiv:2007.14400](https://arxiv.org/abs/2007.14400).
- **Distance correlation decorrelation:** M. D. Andrews *et al.*, *Phys. Rev. D* **101**, 094004 (2020), [arXiv:1905.08628](https://arxiv.org/abs/1905.08628).
- **Quantum extensions:** Explore PennyLane tutorials on variational quantum classifiers for strategies to extend the provided QML backend.
