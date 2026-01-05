# Deep Bayesian Active Learning (Gal et al., 2017) — MNIST Reproduction + Extensions

This repo contains the code I used for my miniproject report:
- Reproduction of **Gal, Islam, Ghahramani (2017)** MNIST active learning experiments **§5.1–§5.2**
- The required **minimal extension** (Bayesian last-layer / “neural linear” regression baseline)
- A **novel extension** (diversity-aware batch acquisition)

I reproduce MNIST deep Bayesian active learning with MC dropout and uncertainty acquisition, add a frozen-feature Bayesian last-layer regression baseline (Exact vs analytic MFVI) using predictive variance acquisition, and propose a simple greedy diversity penalty on batch acquisition to reduce redundancy.
The goal is to make the experimental protocol in the report directly reproducible from these scripts.

---

## Reference paper reproduced
- Y. Gal, R. Islam, Z. Ghahramani (2017). *Deep Bayesian Active Learning with Image Data* (ICML).

---

## What I ran (exact environment)
Tested with: Python 3.14.2; PyTorch 2.9.1+cu128; torchvision 0.24.1+cu128; torch CUDA 12.8 (build); NVIDIA driver 577.13 (CUDA 12.9); cuDNN 91002; GPU NVIDIA GeForce RTX 5070 Ti Laptop GPU.

---

## Experiments implemented (matches the report)

### 1) Reproduction: MNIST deep Bayesian active learning (§5.1–§5.2)
Implemented in `run_active_learning.py`.

- Dataset: MNIST (train 60,000 / test 10,000), `ToTensor()` scaling to [0,1].
- Initial labelled set: 20 points (2 per class); validation set: 100 points; pool is the remainder.
- Pool-based protocol: 100 rounds, acquire K=10 per round (0→1000 acquired).
- Repeats: 3 repeats with seeds {33, 34, 35}; curves reported as mean (std where plotted).
- Model: CNN with dropout; MC dropout at acquisition/evaluation time (dropout enabled).
- MC samples: T = 20 stochastic forward passes.
- Training budget: fixed-step optimisation with 1000 gradient updates per acquisition round (batch size 64).
- Weight decay: selected by grid search on the 100-point validation set (details in the report spec sheet).

Acquisition rules implemented:
- BALD
- MaxEntropy
- VarRatios
- MeanSTD
- Random

Deterministic baseline (§5.2):
- Same architecture/training, but dropout disabled at test time (single forward pass, no MC).
- Deterministic BALD: treated as random tie-breaking because mutual information collapses to a constant under a point-estimate model.

Outputs produced by `run_active_learning.py` include:
- saved curves per strategy and repeat (`.npz`)
- summary `.npz` (means/stds)
- Figure 5.1 plot image
- Figure 5.2 plot images (one per strategy in §5.2)
- label-efficiency table dump (`.npz`)

MNIST is downloaded automatically into `./data/` by torchvision.

---

### 2) Minimal extension: frozen features + Bayesian last layer (regression baseline)
Implemented in `run_minimal_extension.py`.

- MNIST is cast as **multi-output regression** (one-hot vectors treated as continuous targets in 10 dimensions); metric is **RMSE**.
- Inputs are normalised with the standard MNIST mean/std (0.1307, 0.3081).
- Frozen feature extractor ϕθ(x): pretrained once via **self-supervised rotation prediction** on unlabeled MNIST, then frozen and features cached.
  - Rotation pretraining uses a deterministic label assignment `r = idx % 4` with rotations {0°, 90°, 180°, 270°}.
- Bayesian inference is performed only in the final linear layer W (Gaussian prior + isotropic Gaussian likelihood).
- Acquisition: **predictive variance** (trace of predictive covariance).
- Two inference methods compared for the last layer:
  - Exact conjugate Bayesian linear regression
  - Analytic mean-field variational approximation (MFVI; diagonal covariance)
- Hyperparameters (α, σ²) are tuned by grid search on the 100-point validation set (Gaussian validation NLL).

---

### 3) Novel extension: diversity-aware batch acquisition (`_Div`)
Implemented in `run_novel_extension.py`.

- Adds a lightweight diversity modification that can be applied to any scalar uncertainty score:
  - shortlist the top-M pool points by uncertainty
  - greedily select a batch of size k using a cosine-similarity penalty in embedding space:  
    `u(x) − λ * max_{x' in S} cos(e(x), e(x'))`
- Uncertainty scores `u(x)` are computed with MC dropout (dropout enabled), while embeddings `e(x)` are computed with dropout disabled for stability.
- Evaluated across four acquisition families: BALD, MaxEntropy, VarRatios, MeanSTD (base vs `_Div`).
- Regularisation protocol: within each repeat/seed, weight decay is tuned once on the validation set by minimising the MC-estimated predictive NLL under MC dropout (T=20), and the selected value is reused for all strategies (including `_Div` variants) within that repeat.

---

## Repository structure
- `run_active_learning.py` — reproduction of Gal et al. MNIST §5.1–§5.2 (MC dropout + deterministic baseline)
- `run_minimal_extension.py` — frozen-feature Bayesian last-layer regression baseline (Exact vs analytic MFVI)
- `run_novel_extension.py` — diversity-aware batch acquisition variants across uncertainty strategies
- `.gitignore` — excludes local data, results, etc.

---

## How to run (in powershell)
- Reproduction (§5.1–§5.2): `python .\run_active_learning.py`
- Minimal extension: `python .\run_minimal_extension.py`
- Novel extension: `python .\run_novel_extension.py`
