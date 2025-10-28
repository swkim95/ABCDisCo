# ABCDisCo Agent Guidelines

## Purpose
This repository supports research into quantum machine learning (QML) and quantum neural networks (QNNs) with applications to high-energy physics analyses such as the CMS experiment. All future agent interactions must prioritize clear, physics-informed reasoning that bridges foundational concepts with practical implementations. The reference paper can be found here[https://arxiv.org/pdf/2007.14400]

## Communication Standards
- Treat every response as instruction for a physics Ph.D. student entering QML from a collider-physics background. Provide intuitive motivation followed by rigorous derivations or arguments.
- Structure explanations logically: begin with context, outline the physical or mathematical principles, and conclude with implications for CMS-style analyses or related workflows.
- Emphasize robustness. Justify claims with equations, algorithmic complexity estimates, or empirical evidence when relevant.
- When presenting algorithms or conceptual frameworks, relate them to familiar high-energy physics concepts (e.g., feature extraction, anomaly detection, calibration).

## References and Research
- Always refer to ABCDisCo paper[https://arxiv.org/pdf/2007.14400] - This should be the first and most important reference
- Every substantive answer must cite at least one verifiable reference (peer-reviewed paper, arXiv preprint, or conference proceeding) published in the last ten years when possible.
- Verify that each reference exists and is relevant before citing it. Prefer authoritative sources (Nature, PRX Quantum, Quantum, Physical Review, NeurIPS, ICML, QIP, etc.).
- When referencing, include full bibliographic details (authors, title, venue, year, and arXiv identifier if available).
- Highlight cutting-edge developments in QML/QNNs, especially those intersecting with particle physics or collider data analysis.

## Code and Implementation Guidelines
- Write code with clarity and pedagogical comments explaining the physical meaning of variables, qubit registers, loss functions, or encodings.
- Prefer modular, well-documented functions and classes. Include docstrings that state assumptions and expected inputs/outputs.
- Avoid wrapping import statements in try/except blocks.
- When implementing quantum circuits or simulations, annotate each block with the mathematical role it plays (e.g., feature map, ansatz layer, measurement routine).
- For classical components interfacing with quantum models, explain how they support tasks such as anomaly detection, calibration, or background estimation in collider physics.
- As QML/QNN is newly developed field, search web intensively when encountered undefined questions

## Testing and Reproducibility
- Provide reproducible scripts or notebooks for experiments. Include seed control and comments describing dataset splits or preprocessing steps.
- Summarize any tests or benchmarks run, noting dataset sizes, hardware assumptions, and key metrics (e.g., AUC, fidelity, log-likelihood ratios).

## Documentation and Reporting
- Keep README and supplementary documentation updated with conceptual overviews and usage examples tailored to newcomers transitioning from traditional HEP analyses.
- When proposing new research directions, outline potential benefits, experimental considerations, and possible systematic uncertainties relevant to CMS-like experiments.

Adhering to these guidelines ensures that future contributions remain scientifically rigorous, pedagogically valuable, and aligned with the goals of the ABCDisCo project.


## ABCD Definitions (Project Standard)

This project adopts a precise, ABCDisCo-consistent definition of regions and efficiencies for both Single-DisCo and Double-DisCo evaluations. Unless explicitly stated otherwise, all counts are to be interpreted as weighted event counts.

### Variables and use cases
- Single-DisCo (files `script/train_abcd_single.py`, `QNN_script/train_QNN_abcd_single.py`):
  - f: classifier score from the DNN/QNN (x-axis)
  - g: m_hat = 1 - |m_jet - m_top| / m_top (y-axis)
- Double-DisCo (files `doubleDisCo_script/train_abcd_double.py`, `doubleQNN_script/train_QNN_abcd_double.py`):
  - f: score of the first classifier (x-axis)
  - g: score of the second classifier (y-axis)

### Regions (ABCD)
- A (SR): pass both cuts → (f ≥ f_thr) and (g ≥ g_thr)
- B (CR): pass f, fail g → (f ≥ f_thr) and (g < g_thr)
- C (CR): fail f, pass g → (f < f_thr) and (g ≥ g_thr)
- D (CR): fail both → (f < f_thr) and (g < g_thr)

### Efficiencies
Let N_S(·) and N_B(·) denote weighted counts of signal and background, respectively, in the specified region(s). Let N_S(all) and N_B(all) denote the total weighted counts in the split under consideration.

- Signal efficiency of f:
  - ε_S(f) = [N_S(A) + N_S(B)] / N_S(all)
- Signal efficiency of g:
  - ε_S(g) = [N_S(A) + N_S(C)] / N_S(all)
- Background efficiency of f:
  - ε_B(f) = [N_B(A) + N_B(B)] / N_B(all)
  - Using the ABCD independence assumption for background, an equivalent estimator is:
    ε_B(f) = N_B(A) / [N_B(A) + N_B(C)]
- Background efficiency of g:
  - ε_B(g) = [N_B(A) + N_B(C)] / N_B(all)
  - Equivalently: ε_B(g) = N_B(A) / [N_B(A) + N_B(B)]

Background rejection is defined as 1/ε_B(·) for the corresponding variable.

### Signal contamination per region
For i ∈ {A, B, C, D}, define:
- δ_i = N_S(i) / N_B(i)

### Normalized (relative) signal contamination
- δ_rel = (δ_B + δ_C - δ_D) / δ_A

Notes:
- These definitions follow ABCDisCo: T. Aarrestad et al., “ABCDisCo: Automating the ABCD Method with Machine Learning,” Eur. Phys. J. C 81, 256 (2021), arXiv:2007.14400.
- When event weights are unity, the formulas reduce to simple event fractions.
