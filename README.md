# Deep Bayesian Active Learning (Gal et al., 2017) — MNIST Reproduction

This repository contains an **anonymous reproduction** of the MNIST active learning experiments from:

- Y. Gal, R. Islam, Z. Ghahramani (2017). *Deep Bayesian Active Learning with Image Data*.

Focus: reproducing the **Section 5.1 and 5.2** MNIST results (multiple acquisition functions + deterministic baseline).

Tested with: Python 3.14.2; PyTorch 2.9.1+cu128; torchvision 0.24.1+cu128; torch CUDA 12.8 (build); NVIDIA driver 577.13 (CUDA 12.9); cuDNN 91002; GPU NVIDIA GeForce RTX 5070 Ti Laptop GPU.


---

## What's included

- **MC Dropout** active learning on MNIST
- **Five acquisition strategies**:
  - BALD
  - Max Entropy
  - Variation Ratios
  - Mean STD
  - Random
- **Deterministic baseline** (dropout disabled at test time / no MC sampling) to isolate epistemic uncertainty effects

---

## Repository structure

- `run_active_learning.py` — MNIST active learning loop (train → score pool → acquire → repeat)
- `run_mnist_regression.py` — additional MNIST-related experiments
- `test_cuda.py` — quick CUDA/GPU sanity check - AI-generated file to test GPU
- `.gitignore` — excludes `venv/`, local data, results, etc.

Note: MNIST is downloaded programmatically by the code (no dataset files are included in this repo).

---

## Novel extensions

Explain the novel extensions.
