# Machine Learning Methods for Solving the Fokker-Planck Equation

Source code for the master's thesis by Alina Subotina, Taras Shevchenko National University of Kyiv, 2026.

---

## Overview

This repository contains the implementation of the DL-FP method (Xu et al., 2020) for solving the stationary Fokker–Planck equation using deep learning. The study investigates convergence robustness, initialization sensitivity, and the effect of transfer learning on multimodal probability distributions.

Two benchmark problems are considered:

- **1D double-well potential** — bistable system with two stable equilibria
- **1D triple-well potential** — tristable system in symmetric and asymmetric configurations

---

## Repository Structure

```
├── notebooks/
│   ├── double_well.ipynb               # Section 3.2 — Figures 3.1–3.5
│   ├── triple_well_symmetric.ipynb     # Sections 3.3.1–3.3.2 — Figures 3.6–3.10
│   └── triple_well_asymmetric.ipynb    # Section 3.3.3 — Figures 3.11–3.15
└── src/
    ├── models.py        # FPNet neural network architecture
    ├── losses.py        # DL-FP loss function (E1, E2, E3)
    ├── training.py      # Training loop (Adam, L-BFGS, hybrid)
    ├── utils.py         # Gradient computation, accL2 metric
    └── problems/
        ├── double_well.py   # Double-well potential, exact solution
        └── triple_well.py   # Triple-well potential, exact solution
```

---

## Requirements

```
python >= 3.10
torch
numpy
scipy
matplotlib
```

Install dependencies:

```bash
pip install torch numpy scipy matplotlib
```

---

## Usage

Run notebooks from the `notebooks/` directory. Each notebook is self-contained and imports from `src/` via a relative path (`sys.path.append('..')`).

```bash
cd notebooks
jupyter notebook double_well.ipynb
```

All experiments were run on an NVIDIA GeForce GTX 1650 GPU (4 GB VRAM) with CUDA. The code falls back to CPU automatically if no GPU is available.

---

## Method

The DL-FP method approximates the stationary probability density function p(x) with a feedforward neural network and minimizes a composite loss:

```
L(θ) = a1·E1 + a2·E2 + a3·E3
```

where E1 is the PDE residual, E2 enforces normalization (∫p dx = 1), and E3 penalizes boundary violations. The softplus transform is applied to the network output to ensure p(x) > 0.

---

## Reference

Xu, Y., Zhang, H., Li, Y., Zhou, K., Liu, Q., & Kurths, J. (2020). Solving Fokker-Planck equation using deep learning. *Chaos*, 30(1), 013133. https://doi.org/10.1063/1.5132840
