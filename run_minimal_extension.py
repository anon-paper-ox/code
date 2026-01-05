# ============================================================
# Active Learning with Frozen Features + Bayesian Last Layer (MNIST Regression)
# ============================================================

# -------------------------
# Overview (what this file does)
# -------------------------
# - Treat MNIST labels as one-hot vectors in R^10 and fit them with a Gaussian regression model.
# - Precompute frozen ConvNet features phi(x) once (optionally via rotation pretraining on the train images).
# - Do Bayesian linear regression on the last layer: exact posterior vs diagonal MFVI approximation.
# - Active learning score: predictive variance (trace), which is simple under isotropic noise.



import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# -------------------------
# Seeding / reproducibility
# Note: cuDNN deterministic mode can be slower; this is "best effort" reproducibility.
# -------------------------

def set_global_determinism(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id: int):
    # Ensures numpy/random are seeded inside each DataLoader worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



# -------------------------
# Experiment constants
# -------------------------
# Pool-based AL loop: start with a tiny stratified seed set, then acquire ACQUISITION_SIZE per round.
# We tune last-layer hyperparameters on a small labeled validation set (VAL_SIZE).
# Hyperparameters (alpha, sigma2) are tuned once per repeat using init+val, then kept fixed during acquisitions.

NUM_CLASSES = 10  # output dim for one-hot regression
INITIAL_POINTS_PER_CLASS = 2
VAL_SIZE = 100

NUM_REPEATS = 3

TOTAL_ROUNDS = 100
ACQUISITION_SIZE = 10

SEED = 33
set_global_determinism(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = (DEVICE.type == "cuda")

BATCH_SIZE_FEATURES = 512    # feature precompute throughput
BATCH_SIZE_PRETRAIN = 256    # rotation pretraining throughput

# Normalisation is standard for MNIST; helps feature learning.
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

MNIST = datasets.MNIST

NUM_WORKERS = 4
PREFETCH_FACTOR = 4

LOG_EVERY_ROUNDS = 10

# Acquisition strategies
ACQ_STRATEGIES = ["PredVar", "Random"]

# Two required last-layer inference methods
# Note: for this linear-Gaussian last layer, MFVI matches the exact posterior mean; it mainly differs in uncertainty.

INFERENCE_METHODS = ["exact", "mfvi"]

# Last-layer hyperparameters (tuned on the validation set each repeat):
# - alpha:  prior precision
# - sigma2: observation noise variance
ALPHA_GRID  = [1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0, 10.0]
SIGMA2_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]

JITTER = 1e-8  # numerical stabiliser for Cholesky
EPS = 1e-8   # numerical floor for sqrt/log/clamp in float32
EPS_DENOM = 1e-30 # denominator safety (e.g. 1/(x + eps))

# Feature pretraining (self-supervised rotation prediction)
PRETRAIN_FEATURE_EXTRACTOR = True
PRETRAIN_EPOCHS = 3
PRETRAIN_LR = 1e-3


# -------------------------
# IO helpers
# Save per-(strategy,inference) curves as .npz and keep a summary file for plotting.
# atomic_savez avoids half-written files if a run is interrupted.
# -------------------------

def get_results_dir(experiment_id: str) -> str:
    return f"results_{experiment_id}_seed_{SEED}"

def results_path(experiment_id: str, strategy: str, inference: str) -> str:
    base = get_results_dir(experiment_id)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"results_{strategy}_{inference}.npz")

def summary_path(experiment_id: str) -> str:
    base = get_results_dir(experiment_id)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"summary_{experiment_id}.npz")

def figure_path(experiment_id: str) -> str:
    base = get_results_dir(experiment_id)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"figure_{experiment_id}.png")

def atomic_savez(path: str, **kwargs):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **kwargs)
    os.replace(tmp, path)

def load_npz_or_raise(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing results file: {path}")
    return np.load(path, allow_pickle=True)


# -------------------------
# Data split (init / val / pool)
# - init: INITIAL_POINTS_PER_CLASS per class
# - val:  VAL_SIZE labeled points (for hyperparameter tuning)
# - pool: remaining unlabeled points (for acquisition)
# Deterministic given `seed` (we re-seed inside for repeatability of init/val/pool splits).
# -------------------------

def split_indices_stratified(train_full: Dataset, seed: int) -> Tuple[List[int], List[int], List[int]]:
    set_global_determinism(seed)
    targets = train_full.targets.cpu().numpy()

    by_class = [[] for _ in range(NUM_CLASSES)]
    for i, y in enumerate(targets):
        by_class[int(y)].append(i)

    init_idx: List[int] = []
    for c in range(NUM_CLASSES):
        chosen = random.sample(by_class[c], INITIAL_POINTS_PER_CLASS)
        init_idx.extend(chosen)
        for idx in chosen:
            by_class[c].remove(idx)

    remaining: List[int] = []
    for c in range(NUM_CLASSES):
        remaining.extend(by_class[c])

    random.shuffle(remaining)
    val_idx  = remaining[:VAL_SIZE]
    pool_idx = remaining[VAL_SIZE:]
    return init_idx, val_idx, pool_idx


# -------------------------
# Frozen feature extractor (basis functions)
# FeatureNet defines phi(x). We train it once (optional) and then freeze it for the AL loop.
# -------------------------
class FeatureNet(nn.Module):
    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1   = nn.Linear(32 * 11 * 11, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


# -------------------------
# Optional self-supervised pretraining: rotation prediction
# We convert MNIST into a 4-way rotation classification task.
# Deterministic assignment (no 4× expansion): for each image at dataset index `idx`,
# we set the rotation label r = idx % 4 and rotate by {0°, 90°, 180°, 270°} accordingly.
# Label mapping: r ∈ {0,1,2,3} corresponds to rotation ∈ {0°, 90°, 180°, 270°}.
# -------------------------
class RotationDataset(Dataset):

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x,_ = self.base[idx]
        r = idx % 4
        x = torch.rot90(x, k=r, dims=(1,2))
        return x, r



# DataLoader settings (throughput-oriented; does not affect the math).
def dataloader_kwargs(seed: int):
    g = torch.Generator()
    g.manual_seed(seed)

    kwargs = dict(
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS,
        generator=g,
    )
    if NUM_WORKERS > 0:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=PREFETCH_FACTOR,
            worker_init_fn=seed_worker,
        )
    return kwargs



# Rotation pretraining: train FeatureNet + a small 4-class head, then freeze FeatureNet weights.
def pretrain_feature_extractor(feature_net: FeatureNet, train_full: Dataset):
    feature_net = feature_net.to(DEVICE)
    head = nn.Linear(feature_net.fc1.out_features, 4).to(DEVICE)

    ds = RotationDataset(train_full)
    loader = DataLoader(ds, batch_size=BATCH_SIZE_PRETRAIN, shuffle=True, **dataloader_kwargs(SEED))

    opt = torch.optim.Adam(list(feature_net.parameters()) + list(head.parameters()), lr=PRETRAIN_LR)

    feature_net.train()
    head.train()

    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        total_loss, total = 0.0, 0
        for x, r in loader:
            x = x.to(DEVICE, non_blocking=True)
            r = r.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            z = feature_net(x)
            logits = head(z)
            loss = F.cross_entropy(logits, r)
            loss.backward()
            opt.step()

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total += bs

        print(f"[PRETRAIN] epoch {epoch}/{PRETRAIN_EPOCHS} rotation CE={total_loss/max(total,1):.4f}")

    feature_net.eval()
    for p in feature_net.parameters():
        p.requires_grad_(False)
    return feature_net


# -------------------------
# Feature precompute
# We embed all train/test images once into Phi (and append a bias column).
# This keeps the AL loop fast: last-layer updates are just linear algebra.
# -------------------------
@torch.inference_mode()
def compute_features(feature_net: FeatureNet, dataset: Dataset, feat_dim: int) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE_FEATURES, shuffle=False, **dataloader_kwargs(SEED))
    feats = []
    feature_net.eval()

    for x, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        z = feature_net(x)  # (B, D)
        feats.append(z)

    Phi = torch.cat(feats, dim=0)  # (N, D)
    ones = torch.ones(Phi.size(0), 1, device=Phi.device, dtype=Phi.dtype)
    Phi = torch.cat([Phi, ones], dim=1)  # (N, D+1)
    # sanity check: D features + 1 bias column
    assert Phi.size(1) == feat_dim + 1
    return Phi


def one_hot_targets_from_digits(digits: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    return F.one_hot(digits.to(torch.int64), num_classes=num_classes).to(torch.float32)


# Precompute:
# - Phi_train_all / Phi_test: frozen features with a bias column (D+1)
# - Y_train_all / Y_test: one-hot targets in R^10
# All tensors are kept on DEVICE for fast posterior updates.
def precompute_features_and_targets(train_full: Dataset, test_dataset: Dataset, feat_dim: int = 128):
    feature_net = FeatureNet(feat_dim=feat_dim)

    if PRETRAIN_FEATURE_EXTRACTOR:
        feature_net = pretrain_feature_extractor(feature_net, train_full)
    else:
        # No pretraining: use random frozen features as a weak baseline.
        feature_net = feature_net.to(DEVICE)
        feature_net.eval()
        for p in feature_net.parameters():
            p.requires_grad_(False)

    Phi_train_all = compute_features(feature_net, train_full, feat_dim=feat_dim)
    Phi_test      = compute_features(feature_net, test_dataset, feat_dim=feat_dim)

    Y_train_all = one_hot_targets_from_digits(train_full.targets, NUM_CLASSES).to(DEVICE)
    Y_test      = one_hot_targets_from_digits(test_dataset.targets, NUM_CLASSES).to(DEVICE)

    return Phi_train_all, Y_train_all, Phi_test, Y_test


# -------------------------
# Bayesian last-layer inference
# Multi-output linear regression with shared feature covariance across outputs (isotropic prior/noise).
# -------------------------
@dataclass
class PosteriorExact:
    mean_W: torch.Tensor   # (D, K)
    chol_A: torch.Tensor   # (D, D) lower-triangular Cholesky factor of A
    alpha: float
    sigma2: float


@dataclass
class PosteriorMFVI:
    mean_W: torch.Tensor  # (D, K)
    diag_S: torch.Tensor  # (D,)   diagonal covariance in feature space (shared across outputs)
    alpha: float
    sigma2: float


# Exact Bayesian ridge regression on the last layer.
# A = alpha I + (1/sigma2) Phi^T Phi
# mean_W = A^{-1} (1/sigma2) Phi^T Y
# Shared covariance across outputs: Cov(vec(W)) = I_K ⊗ A^{-1}
def fit_last_layer_exact(Phi: torch.Tensor, Y: torch.Tensor, alpha: float, sigma2: float) -> PosteriorExact:
    Phi64 = Phi.to(torch.float64)
    Y64   = Y.to(torch.float64)
    D = Phi64.size(1)

    A = alpha * torch.eye(D, device=Phi64.device, dtype=torch.float64)
    A = A + (Phi64.T @ Phi64) / sigma2
    A = A + JITTER * torch.eye(D, device=A.device, dtype=A.dtype)

    L = torch.linalg.cholesky(A)  # A = L L^T (L is lower-triangular)

    rhs = (Phi64.T @ Y64) / sigma2
    mean_W = torch.cholesky_solve(rhs, L)  # A^{-1} rhs

    return PosteriorExact(
        mean_W=mean_W.to(torch.float32),
        chol_A=L.to(torch.float32),
        alpha=float(alpha),
        sigma2=float(sigma2),
    )



# Diagonal MFVI approximation for the last layer.
# - We reuse the exact posterior mean (optimal for linear-Gaussian).
# - Variances use the standard diagonal fixed point:
#     diag_S[j] = 1 / (alpha + (1/sigma2) * sum_n Phi[n,j]^2)
# Note: diag_S is not diag(A^{-1}); it is 1 / diag(A).
def fit_last_layer_mfvi(Phi: torch.Tensor, Y: torch.Tensor, alpha: float, sigma2: float) -> PosteriorMFVI:
    post_exact = fit_last_layer_exact(Phi, Y, alpha=alpha, sigma2=sigma2)

    Phi2_sum = (Phi.to(torch.float64) ** 2).sum(dim=0)  # (D,)
    diag_prec = alpha + Phi2_sum / sigma2               # (D,)
    diag_S = (1.0 / (diag_prec + EPS_DENOM)).to(torch.float32)

    return PosteriorMFVI(
        mean_W=post_exact.mean_W,
        diag_S=diag_S,
        alpha=float(alpha),
        sigma2=float(sigma2),
    )


# -------------------------
# Predictions / metrics
# RMSE is computed on one-hot vectors; NLL uses the same isotropic Gaussian model as training.
# -------------------------
@torch.inference_mode()
def predict_mean(Phi: torch.Tensor, mean_W: torch.Tensor) -> torch.Tensor:
    return Phi @ mean_W  # (N, K)


# Predictive variance for one output dimension (exact):
#   v(x) = sigma2 + phi^T A^{-1} phi, computed via Cholesky factor A = L L^T
@torch.inference_mode()
def predictive_var_scalar_exact(Phi: torch.Tensor, post: PosteriorExact) -> torch.Tensor:
    # quad = phi^T A^{-1} phi = || L^{-1} phi ||^2 where A = L L^T
    # Solve L * X = Phi^T  -> X = L^{-1} Phi^T
    X = torch.linalg.solve_triangular(post.chol_A, Phi.T, upper=False)  # (D, N)
    quad = (X * X).sum(dim=0)  # (N,)
    return quad + post.sigma2


# Predictive variance for one output dimension (MFVI):
#   v(x) = sigma2 + sum_j phi_j^2 * diag_S[j]
@torch.inference_mode()
def predictive_var_scalar_mfvi(Phi: torch.Tensor, post: PosteriorMFVI) -> torch.Tensor:
    quad = (Phi * Phi * post.diag_S.unsqueeze(0)).sum(dim=1)
    return quad + post.sigma2


# Under isotropic likelihood, Cov[y*|x,D] = var_scalar(x) * I_K, so trace = K * var_scalar.
@torch.inference_mode()
def acquisition_trace_from_var_scalar(var_scalar: torch.Tensor, K: int = NUM_CLASSES) -> torch.Tensor:
    # Multiplying by K does not change the ranking; we keep it so the score equals trace(Cov[y|x]).
    return K * var_scalar

# RMSE on one-hot targets is just a simple scalar metric for this regression framing.
@torch.inference_mode()
def rmse_from_preds(mu: torch.Tensor, Y: torch.Tensor) -> float:
    mse = (mu - Y).pow(2).mean()
    return float(torch.sqrt(mse + EPS).item())


# Mean NLL for isotropic Gaussian outputs with per-sample scalar variance:
#   NLL(x) = 0.5 * [ K log(2*pi*var) + ||y-mu||^2 / var ]
@torch.inference_mode()
def nll_gaussian_isotropic(mu: torch.Tensor, var_scalar: torch.Tensor, Y: torch.Tensor) -> float:
    K = Y.size(1)
    var = var_scalar.clamp_min(EPS)
    sq = (Y - mu).pow(2).sum(dim=1)
    nll = 0.5 * (K * torch.log(2 * torch.pi * var) + sq / var)
    return float(nll.mean().item())


# -------------------------
# Hyperparameter tuning (alpha, sigma2)
# Simple grid search on the validation set using Gaussian NLL.
# -------------------------
def tune_hyperparams(
    Phi_train_all: torch.Tensor,
    Y_train_all: torch.Tensor,
    val_idx: List[int],
    init_idx: List[int],
    inference_method: str,
    seed: int
) -> Tuple[float, float]:
    set_global_determinism(seed)

    idx_init = torch.as_tensor(init_idx, device=Phi_train_all.device, dtype=torch.long)
    idx_val  = torch.as_tensor(val_idx,  device=Phi_train_all.device, dtype=torch.long)

    Phi_init = Phi_train_all.index_select(0, idx_init)
    Y_init   = Y_train_all.index_select(0, idx_init)

    Phi_val = Phi_train_all.index_select(0, idx_val)
    Y_val   = Y_train_all.index_select(0, idx_val)

    best_alpha, best_sigma2 = None, None
    best_val = float("inf")

    for alpha in ALPHA_GRID:
        for sigma2 in SIGMA2_GRID:
            if inference_method == "exact":
                post = fit_last_layer_exact(Phi_init, Y_init, alpha=alpha, sigma2=sigma2)
                mu = predict_mean(Phi_val, post.mean_W)
                var = predictive_var_scalar_exact(Phi_val, post)

            elif inference_method == "mfvi":
                post = fit_last_layer_mfvi(Phi_init, Y_init, alpha=alpha, sigma2=sigma2)
                mu = predict_mean(Phi_val, post.mean_W)
                var = predictive_var_scalar_mfvi(Phi_val, post)

            else:
                raise ValueError(inference_method)

            val_nll = nll_gaussian_isotropic(mu, var, Y_val)

            if val_nll < best_val:
                best_val = val_nll
                best_alpha, best_sigma2 = float(alpha), float(sigma2)

    assert best_alpha is not None and best_sigma2 is not None
    print(f"[TUNE] {inference_method}: best alpha={best_alpha:.2e}, sigma2={best_sigma2:.2e} (val NLL={best_val:.4f})")
    return best_alpha, best_sigma2


# -------------------------
# Active learning loop
# Features stay fixed. Each round: score pool -> acquire -> refit last-layer posterior -> evaluate on test.
# -------------------------
def run_active_learning_once(
    Phi_train_all: torch.Tensor,
    Y_train_all: torch.Tensor,
    Phi_test: torch.Tensor,
    Y_test: torch.Tensor,
    init_idx: List[int],
    pool_idx: List[int],
    strategy: str,
    inference_method: str,
    alpha: float,
    sigma2: float,
    seed: int
) -> List[float]:
    set_global_determinism(seed)

    train_idx = list(init_idx)
    pool_idx = list(pool_idx)

    rmse_curve: List[float] = []

    # --- initial fit + test eval ---
    idx_train = torch.as_tensor(train_idx, device=Phi_train_all.device, dtype=torch.long)
    Phi_train = Phi_train_all.index_select(0, idx_train)
    Y_train   = Y_train_all.index_select(0, idx_train)

    if inference_method == "exact":
        post = fit_last_layer_exact(Phi_train, Y_train, alpha=alpha, sigma2=sigma2)
        mu_test = predict_mean(Phi_test, post.mean_W)
    elif inference_method == "mfvi":
        post = fit_last_layer_mfvi(Phi_train, Y_train, alpha=alpha, sigma2=sigma2)
        mu_test = predict_mean(Phi_test, post.mean_W)
    else:
        raise ValueError(inference_method)

    rmse_curve.append(rmse_from_preds(mu_test, Y_test))

    # --- acquisition rounds ---
    for rnd in range(1, TOTAL_ROUNDS + 1):
        if strategy == "Random":
            acquired = random.sample(pool_idx, ACQUISITION_SIZE)

        # Score pool points by predictive variance of the regression output y (trace under isotropic noise).
        elif strategy == "PredVar":
            idx_pool = torch.as_tensor(pool_idx, device=Phi_train_all.device, dtype=torch.long)
            Phi_pool = Phi_train_all.index_select(0, idx_pool)

            if inference_method == "exact":
                var_scalar = predictive_var_scalar_exact(Phi_pool, post)
            else:
                var_scalar = predictive_var_scalar_mfvi(Phi_pool, post)

            scores = acquisition_trace_from_var_scalar(var_scalar, K=NUM_CLASSES)

            top_local = torch.topk(scores, k=ACQUISITION_SIZE, largest=True).indices.tolist()
            acquired = [pool_idx[i] for i in top_local]

        else:
            raise ValueError(strategy)

        # update sets
        acquired_set = set(acquired)
        train_idx.extend(acquired)
        pool_idx = [i for i in pool_idx if i not in acquired_set]

        # refit posterior
        idx_train = torch.as_tensor(train_idx, device=Phi_train_all.device, dtype=torch.long)
        Phi_train = Phi_train_all.index_select(0, idx_train)
        Y_train   = Y_train_all.index_select(0, idx_train)

        if inference_method == "exact":
            post = fit_last_layer_exact(Phi_train, Y_train, alpha=alpha, sigma2=sigma2)
        else:
            post = fit_last_layer_mfvi(Phi_train, Y_train, alpha=alpha, sigma2=sigma2)

        mu_test = predict_mean(Phi_test, post.mean_W)
        rmse_curve.append(rmse_from_preds(mu_test, Y_test))

        if rnd % LOG_EVERY_ROUNDS == 0 or rnd == TOTAL_ROUNDS:
            print(f"[{strategy} | {inference_method} | seed={seed}] round {rnd}/{TOTAL_ROUNDS} RMSE={rmse_curve[-1]:.4f}")

    return rmse_curve


# -------------------------
# Run one setting (strategy + inference)
# Each repeat: new split + tune hyperparameters on val + run AL loop; results are saved incrementally.
# -------------------------

def run_one_setting(
    experiment_id: str,
    strategy: str,
    inference_method: str,
    Phi_train_all: torch.Tensor,
    Y_train_all: torch.Tensor,
    Phi_test: torch.Tensor,
    Y_test: torch.Tensor,
    train_full: Dataset,
    overwrite: bool = False
):
    save_path = results_path(experiment_id, strategy, inference_method)

    if overwrite and os.path.exists(save_path):
        os.remove(save_path)
        print(f"[OVERWRITE] Deleted existing {save_path}")

    curves_done: List[np.ndarray] = []
    hypers_done: List[Tuple[float, float]] = []
    seeds_done: List[int] = []

    if (not overwrite) and os.path.exists(save_path):
        old = np.load(save_path, allow_pickle=True)
        curves_done = [c for c in old["curves"]]
        hypers_done = [tuple(x) for x in old["best_hypers_per_repeat"]]
        seeds_done  = old["seeds"].tolist()
        print(f"[RESUME] Found {len(curves_done)}/{NUM_REPEATS} repeats in {save_path}")

    start_repeat = len(curves_done)

    for r in range(start_repeat, NUM_REPEATS):
        repeat_seed = SEED + r
        print("\n" + "=" * 80)
        print(f"Experiment={experiment_id} | Strategy={strategy} | Inference={inference_method} | Repeat {r+1}/{NUM_REPEATS} | seed={repeat_seed}")
        print("=" * 80)

        init_idx, val_idx, pool_idx = split_indices_stratified(train_full, repeat_seed)

        alpha_best, sigma2_best = tune_hyperparams(
            Phi_train_all, Y_train_all,
            val_idx=val_idx, init_idx=init_idx,
            inference_method=inference_method,
            seed=repeat_seed
        )

        rmse_curve = run_active_learning_once(
            Phi_train_all=Phi_train_all,
            Y_train_all=Y_train_all,
            Phi_test=Phi_test,
            Y_test=Y_test,
            init_idx=init_idx,
            pool_idx=pool_idx,
            strategy=strategy,
            inference_method=inference_method,
            alpha=alpha_best,
            sigma2=sigma2_best,
            seed=repeat_seed,
        )

        curves_done.append(np.array(rmse_curve, dtype=np.float32))
        hypers_done.append((alpha_best, sigma2_best))
        seeds_done.append(repeat_seed)

        atomic_savez(
            save_path,
            strategy=strategy,
            inference=inference_method,
            curves=np.stack(curves_done, axis=0).astype(np.float32),
            best_hypers_per_repeat=np.array(hypers_done, dtype=np.float64),
            seeds=np.array(seeds_done, dtype=np.int64),
        )
        print(f"[SAVED] {save_path} (repeats completed: {len(curves_done)}/{NUM_REPEATS})")

    print(f"\n[DONE] Experiment={experiment_id} Strategy={strategy} Inference={inference_method} complete.")


# -------------------------
# Summaries + plot
# Summarize curves across repeats (mean/std) and plot RMSE vs acquired labels.
# -------------------------

def acquired_axis() -> np.ndarray:
    # 0, 10, 20, ..., 1000 for TOTAL_ROUNDS=100 and ACQUISITION_SIZE=10
    return np.arange(0, (TOTAL_ROUNDS + 1) * ACQUISITION_SIZE, ACQUISITION_SIZE, dtype=np.int64)


def summarize(experiment_id: str):
    x = acquired_axis()
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    for strat in ACQ_STRATEGIES:
        results[strat] = {}
        for inf in INFERENCE_METHODS:
            data = load_npz_or_raise(results_path(experiment_id, strat, inf))
            curves = data["curves"].astype(np.float32)
            results[strat][inf] = {
                "curves": curves,
                "mean": curves.mean(axis=0),
                "std":  curves.std(axis=0),
            }

    atomic_savez(summary_path(experiment_id), x_acquired=x, results=results)


def plot_figure(experiment_id: str):
    data = load_npz_or_raise(summary_path(experiment_id))
    x = data["x_acquired"]
    results = data["results"].item()

    # Pretty legend names
    strategy_name = {
        "PredVar": "Predictive Variance",
        "Random": "Random",
    }
    inference_name = {
        "exact": "Exact (Conjugate)",
        "mfvi": "MFVI (Diagonal)",
    }

    plt.figure(figsize=(6.6, 4.8))
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=1.0, alpha=0.25)

    for strat in ACQ_STRATEGIES:
        for inf in INFERENCE_METHODS:
            mean = results[strat][inf]["mean"]
            # std  = results[strat][inf]["std"]  # no shading anymore

            label = f"{strategy_name.get(strat, strat)} — {inference_name.get(inf, inf)}"
            ax.plot(x, mean, linewidth=2.2, label=label)

            # Removed shading:
            # ax.fill_between(x, mean - std, mean + std, alpha=0.15, linewidth=0)

    ax.set_xlabel("Acquired images")
    ax.set_ylabel("Test RMSE (lower is better)")
    ax.set_xlim(0, int(x[-1]))

    # Slightly nicer legend styling
    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        fontsize=9,
        title="Acquisition — Inference",
        title_fontsize=9,
    )

    out = figure_path(experiment_id)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")



# -------------------------
# Main
# Precompute frozen features once, then run the AL loop purely in feature space.
# -------------------------
if __name__ == "__main__":
    train_full = MNIST(root="./data", train=True, download=True, transform=TRANSFORM)
    test_dataset = MNIST(root="./data", train=False, download=True, transform=TRANSFORM)

    print("[SETUP] Computing frozen features + one-hot regression targets ...")
    Phi_train_all, Y_train_all, Phi_test, Y_test = precompute_features_and_targets(train_full, test_dataset, feat_dim=128)
    print(f"[SETUP] Phi_train_all: {tuple(Phi_train_all.shape)}, Y_train_all: {tuple(Y_train_all.shape)}")
    print(f"[SETUP] Phi_test:      {tuple(Phi_test.shape)},      Y_test:      {tuple(Y_test.shape)}")

    experiment_id = "minimal_extension_neural_linear"

    for strategy in ACQ_STRATEGIES:
        for inference_method in INFERENCE_METHODS:
            run_one_setting(
                experiment_id=experiment_id,
                strategy=strategy,
                inference_method=inference_method,
                Phi_train_all=Phi_train_all,
                Y_train_all=Y_train_all,
                Phi_test=Phi_test,
                Y_test=Y_test,
                train_full=train_full,
                overwrite=True,
            )

    summarize(experiment_id)
    plot_figure(experiment_id)
