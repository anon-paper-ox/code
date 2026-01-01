# ============================================================
# Active Learning with Frozen Features + Bayesian Last Layer (MNIST Regression)
# Minimal extension: analytic exact vs analytic MFVI on last layer
# Acquisition: predictive variance
# Metric: RMSE
# ============================================================

import os
import random
import shutil
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Sampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# -------------------------
# 🔇 Silence PyTorch DataLoader cleanup spam (keeps workers=4 + persistent=True)
# -------------------------
def silence_pytorch_dataloader_cleanup_error():
    import torch.utils.data.dataloader as dl
    cls = getattr(dl, "_MultiProcessingDataLoaderIter", None)
    if cls is None:
        return
    orig_del = cls.__del__
    def safe_del(self):
        try:
            orig_del(self)
        except AssertionError:
            pass
    cls.__del__ = safe_del


def silence_multiprocessing_connection_cleanup_error():
    import multiprocessing.connection as mp_conn
    base = getattr(mp_conn, "_ConnectionBase", None)
    if base is None:
        return
    orig_del = base.__del__
    def safe_del(self):
        try:
            orig_del(self)
        except OSError as e:
            if getattr(e, "errno", None) == 9:
                return
            raise
    base.__del__ = safe_del


silence_pytorch_dataloader_cleanup_error()
silence_multiprocessing_connection_cleanup_error()


# -------------------------
# Experiment constants
# -------------------------
NUM_CLASSES = 10              # output dim for regression
INITIAL_POINTS_PER_CLASS = 2
VAL_SIZE = 100

NUM_REPEATS = 3

TOTAL_ROUNDS = 100
ACQUISITION_SIZE = 10

SEED = 45
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = (DEVICE.type == "cuda")

BATCH_SIZE_EVAL  = 512

TRANSFORM = transforms.ToTensor()
MNIST = datasets.MNIST

# ✅ requested
NUM_WORKERS = 4
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4

LOG_EVERY_ROUNDS = 10

# Predictive variance acquisition baseline + (optional) random sanity check
ACQ_STRATEGIES = ["PredVar", "Random"]

# Two required last-layer inference methods
INFERENCE_METHODS = ["exact", "mfvi"]

# Hyperparameter grids (prior precision alpha, noise variance sigma^2)
# (You can adjust; these are reasonable starting points.)
ALPHA_GRID  = [1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0, 10.0]
SIGMA2_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]

EPS = 1e-12
JITTER = 1e-8  # for numerical stability in Cholesky

# -------------------------
# Quick smoke test config
# -------------------------
QUICK_TEST = False
if QUICK_TEST:
    NUM_REPEATS = 2
    TOTAL_ROUNDS = 20
    ACQUISITION_SIZE = 10
    ALPHA_GRID  = [1e-3, 1e-2, 1e-1]
    SIGMA2_GRID = [1e-2, 1e-1, 1.0]
    LOG_EVERY_ROUNDS = 1
    ACQ_STRATEGIES = ["PredVar"]


# -------------------------
# Paths / IO
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

def _load_npz_or_raise(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing results file: {path}")
    return np.load(path, allow_pickle=True)

def atomic_savez(path, **kwargs):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **kwargs)
    os.replace(tmp, path)


# -------------------------
# Seeding helpers
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_torch_seed_only(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# Samplers (only for feature precompute loaders)
# -------------------------
class MutableSubsetSequentialSampler(Sampler[int]):
    def __init__(self, indices=None):
        self.indices = list(indices) if indices is not None else []

    def set_indices(self, indices):
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def _dataloader_kwargs():
    kwargs = dict(
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
    )
    if NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return kwargs


# -------------------------
# Data split
# -------------------------
def split_indices(train_full, seed):
    set_seed(seed)
    targets = train_full.targets.cpu().numpy()

    by_class = [[] for _ in range(NUM_CLASSES)]
    for i, y in enumerate(targets):
        by_class[int(y)].append(i)

    init_idx = []
    for c in range(NUM_CLASSES):
        chosen = random.sample(by_class[c], INITIAL_POINTS_PER_CLASS)
        init_idx.extend(chosen)
        for idx in chosen:
            by_class[c].remove(idx)

    remaining = []
    for c in range(NUM_CLASSES):
        remaining.extend(by_class[c])

    random.shuffle(remaining)
    val_idx  = remaining[:VAL_SIZE]
    pool_idx = remaining[VAL_SIZE:]
    return init_idx, val_idx, pool_idx


# -------------------------
# Frozen Feature Extractor (basis functions)
# -------------------------
class FrozenFeatureNet(nn.Module):
    """
    Frozen neural network used as a basis-function map phi(x).
    Output is a feature vector (no dropout; deterministic).
    """
    def __init__(self, feat_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1   = nn.Linear(32 * 11 * 11, feat_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


# -------------------------
# Precompute features for ALL points (fast AL)
# -------------------------
def make_full_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        **_dataloader_kwargs(),
    )

def one_hot_targets_from_digits(digits: torch.Tensor, num_classes=10) -> torch.Tensor:
    # digits: (N,) int64 -> (N, K) float32
    return F.one_hot(digits.to(torch.int64), num_classes=num_classes).to(torch.float32)

def precompute_features_and_targets(train_full, test_dataset, feat_dim=128):
    """
    Returns:
      Phi_train: (N_train, D) tensor on DEVICE
      Y_train:   (N_train, K) tensor on DEVICE (continuous one-hot)
      Phi_test:  (N_test, D) tensor on DEVICE
      Y_test:    (N_test, K) tensor on DEVICE
    We also append a bias feature (constant 1) into Phi.
    """
    set_torch_seed_only(SEED)  # stable frozen-net init
    feature_net = FrozenFeatureNet(feat_dim=feat_dim).to(DEVICE)
    feature_net.eval()
    for p in feature_net.parameters():
        p.requires_grad_(False)

    def compute_phi(dataset):
        loader = make_full_loader(dataset)
        feats = []
        with torch.inference_mode():
            for x, _ in loader:
                x = x.to(DEVICE, non_blocking=True)
                z = feature_net(x)  # (B, feat_dim)
                feats.append(z)
        Phi = torch.cat(feats, dim=0)  # (N, feat_dim)
        # append bias
        ones = torch.ones(Phi.size(0), 1, device=Phi.device, dtype=Phi.dtype)
        Phi = torch.cat([Phi, ones], dim=1)  # (N, feat_dim+1)
        return Phi

    Phi_train = compute_phi(train_full)
    Phi_test  = compute_phi(test_dataset)

    Y_train = one_hot_targets_from_digits(train_full.targets, NUM_CLASSES).to(DEVICE)
    Y_test  = one_hot_targets_from_digits(test_dataset.targets, NUM_CLASSES).to(DEVICE)

    return Phi_train, Y_train, Phi_test, Y_test


# -------------------------
# Bayesian last-layer inference (multi-output linear regression)
# Model: Y = Phi W + eps, eps ~ N(0, sigma^2 I_K)
# Prior: W ~ N(0, alpha^{-1} I)  (independent across outputs)
# -------------------------
@dataclass
class PosteriorExact:
    mean_W: torch.Tensor     # (D, K)
    A_inv: torch.Tensor      # (D, D) posterior covariance per output
    alpha: float
    sigma2: float

@dataclass
class PosteriorMFVI:
    mean_W: torch.Tensor     # (D, K) (matches exact mean)
    diag_S: torch.Tensor     # (D,) diagonal covariance approximation
    alpha: float
    sigma2: float

def fit_last_layer_exact(Phi: torch.Tensor, Y: torch.Tensor, alpha: float, sigma2: float) -> PosteriorExact:
    """
    Exact conjugate Bayesian linear regression for multi-output with shared Phi.
    """
    # use float64 for stable linear algebra
    Phi64 = Phi.to(torch.float64)
    Y64   = Y.to(torch.float64)
    D = Phi64.size(1)

    A = alpha * torch.eye(D, device=Phi64.device, dtype=torch.float64) + (Phi64.T @ Phi64) / sigma2
    A = A + JITTER * torch.eye(D, device=A.device, dtype=A.dtype)

    L = torch.linalg.cholesky(A)
    rhs = (Phi64.T @ Y64) / sigma2  # (D, K)
    mean_W = torch.cholesky_solve(rhs, L)  # A^{-1} rhs
    A_inv = torch.cholesky_inverse(L)

    return PosteriorExact(mean_W=mean_W.to(torch.float32),
                          A_inv=A_inv.to(torch.float32),
                          alpha=float(alpha),
                          sigma2=float(sigma2))

def fit_last_layer_mfvi(Phi: torch.Tensor, Y: torch.Tensor, alpha: float, sigma2: float) -> PosteriorMFVI:
    """
    Analytic mean-field VI for linear-Gaussian model:
      - Optimal mean equals exact posterior mean
      - Optimal diagonal covariance: diag_S[j] = 1 / (alpha + (1/sigma2) * sum_n Phi[n,j]^2)
    """
    post_exact = fit_last_layer_exact(Phi, Y, alpha, sigma2)

    # diag of precision: alpha + (1/sigma2) sum_n phi^2
    Phi2_sum = (Phi.to(torch.float64) ** 2).sum(dim=0)  # (D,)
    diag_prec = alpha + Phi2_sum / sigma2
    diag_S = (1.0 / (diag_prec + 1e-30)).to(torch.float32)  # (D,)

    return PosteriorMFVI(mean_W=post_exact.mean_W,
                         diag_S=diag_S,
                         alpha=float(alpha),
                         sigma2=float(sigma2))


# -------------------------
# Predictions / metrics
# -------------------------
def predict_mean(Phi: torch.Tensor, mean_W: torch.Tensor) -> torch.Tensor:
    # Phi: (N,D), mean_W: (D,K) => (N,K)
    return Phi @ mean_W

def predictive_variance_exact(Phi: torch.Tensor, post: PosteriorExact) -> torch.Tensor:
    """
    Returns scalar predictive variance per point (same for each output under isotropic noise & independent outputs):
      var(x) = sigma2 + phi^T A_inv phi
    """
    tmp = Phi @ post.A_inv  # (N,D)
    quad = (tmp * Phi).sum(dim=1)  # (N,)
    return quad + post.sigma2

def predictive_variance_mfvi(Phi: torch.Tensor, post: PosteriorMFVI) -> torch.Tensor:
    """
    Mean-field predictive variance:
      var(x) = sigma2 + sum_j phi_j^2 * diag_S[j]
    """
    quad = (Phi * Phi * post.diag_S.unsqueeze(0)).sum(dim=1)  # (N,)
    return quad + post.sigma2

def rmse_from_preds(mu: torch.Tensor, Y: torch.Tensor) -> float:
    # global RMSE over all entries
    mse = (mu - Y).pow(2).mean()
    return float(torch.sqrt(mse + 1e-30).item())

def nll_gaussian_diag(mu: torch.Tensor, var_scalar: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Multi-output NLL with isotropic variance across outputs:
      y in R^K, var is scalar per sample:
      NLL = 0.5 * [ K log(2*pi*var) + ||y-mu||^2 / var ]
    """
    K = Y.size(1)
    var = var_scalar.clamp_min(1e-30)
    sq = (Y - mu).pow(2).sum(dim=1)  # (N,)
    nll = 0.5 * (K * torch.log(2 * torch.pi * var) + sq / var)
    return float(nll.mean().item())


# -------------------------
# Hyperparameter tuning on (init_idx -> train), (val_idx -> validate)
# Tune by validation NLL (uses predictive variance, so sigma2 matters)
# -------------------------
def tune_hyperparams(Phi_train_all, Y_train_all, val_idx, init_idx, inference_method: str, seed: int):
    set_seed(seed)

    Phi_init = Phi_train_all[torch.tensor(init_idx, device=Phi_train_all.device)]
    Y_init   = Y_train_all[torch.tensor(init_idx, device=Y_train_all.device)]

    Phi_val = Phi_train_all[torch.tensor(val_idx, device=Phi_train_all.device)]
    Y_val   = Y_train_all[torch.tensor(val_idx, device=Y_train_all.device)]

    best = None
    best_val = float("inf")

    for alpha in ALPHA_GRID:
        for sigma2 in SIGMA2_GRID:
            if inference_method == "exact":
                post = fit_last_layer_exact(Phi_init, Y_init, alpha=alpha, sigma2=sigma2)
                mu = predict_mean(Phi_val, post.mean_W)
                var = predictive_variance_exact(Phi_val, post)
                val_nll = nll_gaussian_diag(mu, var, Y_val)

            elif inference_method == "mfvi":
                post = fit_last_layer_mfvi(Phi_init, Y_init, alpha=alpha, sigma2=sigma2)
                mu = predict_mean(Phi_val, post.mean_W)
                var = predictive_variance_mfvi(Phi_val, post)
                val_nll = nll_gaussian_diag(mu, var, Y_val)

            else:
                raise ValueError(inference_method)

            if val_nll < best_val:
                best_val = val_nll
                best = (float(alpha), float(sigma2))

    alpha_best, sigma2_best = best
    print(f"[TUNE] {inference_method}: best alpha={alpha_best:.2e}, sigma2={sigma2_best:.2e} (val NLL={best_val:.4f})")
    return alpha_best, sigma2_best


# -------------------------
# Acquisition scoring
# -------------------------
def score_pool_predvar(Phi_pool: torch.Tensor, inference_method: str, post):
    if inference_method == "exact":
        return predictive_variance_exact(Phi_pool, post)
    elif inference_method == "mfvi":
        return predictive_variance_mfvi(Phi_pool, post)
    else:
        raise ValueError(inference_method)


# -------------------------
# Active learning loop
# -------------------------
def run_active_learning_once(
    Phi_train_all, Y_train_all, Phi_test, Y_test,
    init_idx, pool_idx,
    strategy: str,
    inference_method: str,
    alpha: float, sigma2: float,
    seed: int
):
    set_seed(seed)
    train_idx = list(init_idx)
    pool_idx  = list(pool_idx)

    rmse_curve = []

    # initial posterior + test eval
    Phi_train = Phi_train_all[torch.tensor(train_idx, device=Phi_train_all.device)]
    Y_train   = Y_train_all[torch.tensor(train_idx, device=Y_train_all.device)]

    if inference_method == "exact":
        post = fit_last_layer_exact(Phi_train, Y_train, alpha=alpha, sigma2=sigma2)
        mu_test = predict_mean(Phi_test, post.mean_W)
    else:
        post = fit_last_layer_mfvi(Phi_train, Y_train, alpha=alpha, sigma2=sigma2)
        mu_test = predict_mean(Phi_test, post.mean_W)

    rmse_curve.append(rmse_from_preds(mu_test, Y_test))

    for _round in range(1, TOTAL_ROUNDS + 1):
        if strategy == "Random":
            acquired = random.sample(pool_idx, ACQUISITION_SIZE)
        elif strategy == "PredVar":
            Phi_pool = Phi_train_all[torch.tensor(pool_idx, device=Phi_train_all.device)]
            scores = score_pool_predvar(Phi_pool, inference_method, post)  # tensor (|pool|,)

            k = ACQUISITION_SIZE
            top_local = torch.topk(scores, k=k, largest=True).indices.tolist()
            acquired = [pool_idx[i] for i in top_local]
        else:
            raise ValueError(strategy)

        acquired_set = set(acquired)
        train_idx.extend(acquired)
        pool_idx = [i for i in pool_idx if i not in acquired_set]

        # refit posterior (closed-form; cheap)
        Phi_train = Phi_train_all[torch.tensor(train_idx, device=Phi_train_all.device)]
        Y_train   = Y_train_all[torch.tensor(train_idx, device=Y_train_all.device)]

        if inference_method == "exact":
            post = fit_last_layer_exact(Phi_train, Y_train, alpha=alpha, sigma2=sigma2)
        else:
            post = fit_last_layer_mfvi(Phi_train, Y_train, alpha=alpha, sigma2=sigma2)

        mu_test = predict_mean(Phi_test, post.mean_W)
        rmse_curve.append(rmse_from_preds(mu_test, Y_test))

        if _round % LOG_EVERY_ROUNDS == 0 or _round == TOTAL_ROUNDS:
            print(f"[{strategy} | {inference_method} | seed={seed}] round {_round}/{TOTAL_ROUNDS} RMSE={rmse_curve[-1]:.4f}")

    return rmse_curve


# -------------------------
# Run one strategy + inference method (with resume support)
# -------------------------
def run_one_setting(experiment_id, strategy, inference_method,
                    Phi_train_all, Y_train_all, Phi_test, Y_test,
                    train_full, overwrite=False):
    save_path = results_path(experiment_id, strategy, inference_method)

    if overwrite and os.path.exists(save_path):
        os.remove(save_path)
        print(f"[OVERWRITE] Deleted existing {save_path}")

    curves_done, hypers_done, seeds_done = [], [], []

    if (not overwrite) and os.path.exists(save_path):
        old = np.load(save_path, allow_pickle=True)
        curves_done = list(old["curves"])
        hypers_done = old["best_hypers_per_repeat"].tolist()
        seeds_done  = old["seeds"].tolist()
        print(f"[RESUME] Found {len(curves_done)}/{NUM_REPEATS} repeats in {save_path}")

    start_repeat = len(curves_done)

    for r in range(start_repeat, NUM_REPEATS):
        repeat_seed = SEED + r
        print("\n" + "=" * 70)
        print(f"Experiment={experiment_id} | Strategy={strategy} | Inference={inference_method} | Repeat {r+1}/{NUM_REPEATS} | seed={repeat_seed}")
        print("=" * 70)

        init_idx, val_idx, pool_idx = split_indices(train_full, repeat_seed)
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
# Summaries + plots
# -------------------------
def acquired_axis():
    return np.arange(0, (TOTAL_ROUNDS + 1) * ACQUISITION_SIZE, ACQUISITION_SIZE)

def summarize(experiment_id: str):
    x = acquired_axis()
    results = {}
    for strat in ACQ_STRATEGIES:
        results[strat] = {}
        for inf in INFERENCE_METHODS:
            data = _load_npz_or_raise(results_path(experiment_id, strat, inf))
            curves = data["curves"].astype(np.float32)
            results[strat][inf] = {
                "curves": curves,
                "mean": curves.mean(axis=0),
                "std": curves.std(axis=0)
            }
    atomic_savez(summary_path(experiment_id), x_acquired=x, results=results)

def plot_figure(experiment_id: str):
    data = _load_npz_or_raise(summary_path(experiment_id))
    x = data["x_acquired"]
    results = data["results"].item()

    plt.figure(figsize=(6.4, 4.8))
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=1.0, alpha=0.25)

    # Plot PredVar for both inference methods (and optionally Random)
    for strat in ACQ_STRATEGIES:
        for inf in INFERENCE_METHODS:
            mean = results[strat][inf]["mean"]
            std  = results[strat][inf]["std"]
            label = f"{strat}-{inf}"
            ax.plot(x, mean, linewidth=2.2, label=label)
            ax.fill_between(x, mean - std, mean + std, alpha=0.15, linewidth=0)

    ax.set_xlabel("Acquired images")
    ax.set_ylabel("Test RMSE (lower is better)")
    ax.set_xlim(0, int(x[-1]))
    ax.legend(loc="upper right", frameon=True, fancybox=False, framealpha=1.0, fontsize=9)

    out = figure_path(experiment_id)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    train_full = MNIST(root="./data", train=True, download=True, transform=TRANSFORM)
    test_dataset = MNIST(root="./data", train=False, download=True, transform=TRANSFORM)

    # Precompute frozen features and one-hot regression targets
    print("[SETUP] Precomputing frozen features (this happens once)...")
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
