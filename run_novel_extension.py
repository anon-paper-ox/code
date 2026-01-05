# ============================================================
# Deep Bayesian Active Learning (MNIST) with MC Dropout
# Novel Extension A: Diversity-aware batch acquisition
#
# Based on the reproduction code for:
#   Y. Gal, R. Islam, Z. Ghahramani (2017). "Deep Bayesian Active Learning with Image Data"
#
# This file intentionally keeps the same overall structure as the reproduction runner, but
# adds a single novel idea: within-batch diversity regularisation during acquisition.
#
# ----------------------------
# CHANGE LOG (Novel Extension A)
# ----------------------------
# [CHANGE 1] IndexedDataset wrapper so DataLoaders return (x, y, idx).
#            Implemented in: IndexedDataset, unpack_batch, and any loop that reads batches.
#
# [CHANGE 2] Batch unpacking helper that supports both (x, y) and (x, y, idx).
#            Implemented in: unpack_batch, used in training/eval/scoring loops.
#
# [CHANGE 3] SimpleBCNN.forward(...) optionally returns penultimate features.
#            Implemented in: SimpleBCNN.forward(return_features=True).
#
# [CHANGE 4] New strategy suffix "<BASE>_Div" enabling diversity-aware acquisition.
#            Implemented in: is_diverse_strategy, base_strategy_name, and AL loop branch.
#
# [CHANGE 5] Novel-A runner block in __main__ (tune WD once per repeat, reuse across strategies).
#            Implemented in: the __main__ runner at the bottom of this file.
#
# [CHANGE 6] Diversity hyperparameters controlling shortlist size and penalty strength.
#            Implemented in: DIVERSE_TOP_M, DIVERSE_LAMBDA, and diversity selection functions.
#
# Core novelty (high level):
#   - Uncertainty is computed with MC dropout (dropout ON).
#   - Embeddings are computed deterministically (dropout OFF).
#   - Batch selection is greedy on: u(x) - lambda * max_{x' in S} cos(e(x), e(x')).
# ============================================================

import os
import random
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Sampler, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# -------------------------
# Dataset wrapper: expose dataset indices
# DataLoader yields (x, y, idx) so acquisition can select exact dataset items.
# -------------------------
class IndexedDataset(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, y, idx

    @property
    def targets(self):
        return self.base.targets


# -------------------------
# Batch unpacking
# Supports datasets returning either:
#   - (x, y)      (standard torchvision datasets)
#   - (x, y, idx) (IndexedDataset wrapper)
# Returns: x, y, idx_or_none
# -------------------------
def unpack_batch(batch):
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        x, y, idx = batch
        return x, y, idx
    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
        x, y = batch
        return x, y, None
    else:
        raise ValueError(f"Unexpected batch format of length {len(batch)}")


# -------------------------
# Experiment constants (same defaults as reproduction)
# -------------------------
NUM_CLASSES = 10

INITIAL_POINTS_PER_CLASS = 2
VAL_SIZE = 100

NUM_REPEATS = 3

TOTAL_ROUNDS = 100
ACQUISITION_SIZE = 10

# MC dropout forward passes (T)
MC_SAMPLES = 20

# Training schedule: fixed number of SGD updates per round
TRAIN_STEPS = 1000
TUNE_STEPS = 500

LR = 1e-3
WEIGHT_DECAY_GRID = [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

EPS = 1e-10
SEED = 33

BATCH_SIZE_TRAIN = 64
BATCH_SIZE_EVAL = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = (DEVICE.type == "cuda")

TRANSFORM = transforms.ToTensor()
MNIST = datasets.MNIST

NUM_WORKERS = 4
PERSISTENT_WORKERS = (NUM_WORKERS > 0)
PREFETCH_FACTOR = 4

LOG_EVERY_ROUNDS = 10

NOVEL_A_EXPERIMENT_ID = "novel_A"

BASE_STRATEGIES = [
    "BALD",
    "VarRatios",
    "MaxEntropy",
    "MeanSTD",
]

NOVEL_A_STRATEGIES = (
    BASE_STRATEGIES
    + [f"{s}_Div" for s in BASE_STRATEGIES]
)


# -------------------------
# Novel Extension A hyperparameters
#
# Procedure per acquisition round:
#   1) Score whole pool by uncertainty u(x) (dropout ON, MC)
#   2) Shortlist top-M by u(x)
#   3) Compute embeddings e(x) for shortlist (dropout OFF)
#   4) Greedy pick k points with diversity penalty
# -------------------------
DIVERSE_TOP_M = 500
DIVERSE_LAMBDA = 0.5
DIVERSE_EPS = 1e-12


# -------------------------
# Quick-test mode (optional)
# Reduces compute for sanity-checking end-to-end execution.
# -------------------------
QUICK_TEST = False

if QUICK_TEST:
    NUM_REPEATS = 2
    TOTAL_ROUNDS = 20
    ACQUISITION_SIZE = 10
    TRAIN_STEPS = 100
    TUNE_STEPS = 75
    MC_SAMPLES = 10
    WEIGHT_DECAY_GRID = [1e-4]
    LOG_EVERY_ROUNDS = 4

    BASE_STRATEGIES = ["BALD"]
    NOVEL_A_STRATEGIES = (
        BASE_STRATEGIES
        + [f"{s}_Div" for s in BASE_STRATEGIES]
    )
    DIVERSE_TOP_M = 200



# -------------------------
# Output paths + atomic saving
# -------------------------
def get_results_dir(experiment_id: str) -> str:
    return f"results_{experiment_id}_seed_{SEED}"


def results_path(experiment_id: str, strategy: str) -> str:
    base = get_results_dir(experiment_id)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"results_{strategy}.npz")


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
# Seeding
# set_seed: full RNG (python/numpy/torch)
# set_torch_seed_only: torch-only (used for deterministic init weights)
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
# Samplers
#
# - MutableSubsetSequentialSampler: iterate a fixed index list in order (pool/val)
# - MutableSubsetReplacementSampler: sample with replacement (train) to enforce
#   a fixed number of SGD updates per round: steps * batch_size draws.
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


class MutableSubsetReplacementSampler(Sampler[int]):
    def __init__(self, indices=None, seed: int = 0, num_samples: int = 0):
        self.indices = list(indices) if indices is not None else []
        self.seed = int(seed)
        self.num_samples = int(num_samples)

    def set_indices(self, indices):
        self.indices = list(indices)

    def set_seed(self, seed: int):
        self.seed = int(seed)

    def set_num_samples(self, num_samples: int):
        self.num_samples = int(num_samples)

    def __iter__(self):
        if len(self.indices) == 0 or self.num_samples <= 0:
            return iter([])

        g = torch.Generator()
        g.manual_seed(self.seed)

        draws = torch.randint(0, len(self.indices), (self.num_samples,), generator=g).tolist()
        return (self.indices[i] for i in draws)

    def __len__(self):
        return self.num_samples


# -------------------------
# Shared loaders
# One dataset (train_full) + three samplers:
#   - train_sampler: replacement sampling over current labelled set
#   - val_sampler: sequential over validation set
#   - pool_sampler: sequential over pool set (for scoring/embeddings)
# -------------------------
@dataclass
class SharedLoaders:
    train_loader: DataLoader
    train_sampler: MutableSubsetReplacementSampler
    val_loader: DataLoader
    val_sampler: MutableSubsetSequentialSampler
    pool_loader: DataLoader
    pool_sampler: MutableSubsetSequentialSampler


def _dataloader_kwargs():
    kwargs = dict(
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
    )
    if NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return kwargs


def build_shared_loaders(train_full) -> SharedLoaders:
    train_sampler = MutableSubsetReplacementSampler(indices=[], seed=0, num_samples=0)
    val_sampler = MutableSubsetSequentialSampler(indices=[])
    pool_sampler = MutableSubsetSequentialSampler(indices=[])

    train_loader = DataLoader(
        train_full,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=False,
        sampler=train_sampler,
        drop_last=True,
        **_dataloader_kwargs(),
    )
    val_loader = DataLoader(
        train_full,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        sampler=val_sampler,
        **_dataloader_kwargs(),
    )
    pool_loader = DataLoader(
        train_full,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        sampler=pool_sampler,
        **_dataloader_kwargs(),
    )

    return SharedLoaders(
        train_loader=train_loader,
        train_sampler=train_sampler,
        val_loader=val_loader,
        val_sampler=val_sampler,
        pool_loader=pool_loader,
        pool_sampler=pool_sampler,
    )


def make_test_loader(test_dataset):
    return DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        **_dataloader_kwargs(),
    )


# -------------------------
# Data split (same as reproduction)
# - init: 2 points per class
# - val: 100 points
# - pool: remainder
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
    val_idx = remaining[:VAL_SIZE]
    pool_idx = remaining[VAL_SIZE:]
    return init_idx, val_idx, pool_idx


# -------------------------
# Model: CNN with dropout (MC Dropout at acquisition/eval time)
#
# [CHANGE 3] forward(return_features=True) returns (logits, penultimate_features).
# Penultimate features are used as embeddings for Novel-A diversity selection.
# -------------------------
class SimpleBCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 11 * 11, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

        self.drop1 = nn.Dropout(p=0.25)
        self.drop2 = nn.Dropout(p=0.5)

    def forward(self, x, return_features: bool = False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        feats = x
        logits = self.fc2(feats)
        if return_features:
            return logits, feats
        return logits


# -------------------------
# Training loop (same structure as reproduction)
#
# Key detail:
# - We train for a fixed number of SGD updates per round (TRAIN_STEPS).
# - Sampling is with replacement to match the update budget exactly.
#
# Initialization detail (reproduction-style):
# - If init_state_dict is provided, we load it (fixed init across rounds).
# - Otherwise, init_seed controls deterministic weight initialization.
# -------------------------
def train_model(
    train_idx,
    weight_decay,
    steps,
    shared: SharedLoaders,
    init_state_dict=None,
    init_seed=None,
    shuffle_seed=None,
):
    if init_state_dict is None and init_seed is not None:
        set_torch_seed_only(init_seed)

    model = SimpleBCNN()
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    shared.train_sampler.set_indices(train_idx)
    shared.train_sampler.set_seed(shuffle_seed if shuffle_seed is not None else 0)
    shared.train_sampler.set_num_samples(steps * BATCH_SIZE_TRAIN)

    model.train()
    for batch in shared.train_loader:
        x, y, _ = unpack_batch(batch)
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    return model


# -------------------------
# Predictive helpers
#
# MC dropout:
# - We use dropout ON via model.train() and do T stochastic passes (MC_SAMPLES).
# - Embeddings for diversity are computed with dropout OFF via model.eval().
# -------------------------
def predictive_probs(model, x, T=MC_SAMPLES):
    model.train()
    B = x.size(0)
    x_rep = x.repeat(T, 1, 1, 1)
    logits = model(x_rep).view(T, B, NUM_CLASSES)
    probs_T = F.softmax(logits, dim=2)
    return probs_T.mean(dim=0), probs_T


def accuracy(model, loader, T=MC_SAMPLES):
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch in loader:
            x, y, _ = unpack_batch(batch)
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            mean_probs, _ = predictive_probs(model, x, T=T)
            preds = mean_probs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return correct / total


def mc_nll(model, loader, T=MC_SAMPLES):
    model.train()
    total_nll = 0.0
    total = 0
    with torch.inference_mode():
        for batch in loader:
            x, y, _ = unpack_batch(batch)
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            B = x.size(0)

            x_rep = x.repeat(T, 1, 1, 1)
            logits = model(x_rep).view(T, B, NUM_CLASSES)
            probs = F.softmax(logits, dim=2).mean(dim=0).clamp_min(EPS)

            total_nll += F.nll_loss(probs.log(), y, reduction="sum").item()
            total += B
    return total_nll / total


# -------------------------
# Novel Extension A: Diversity-aware acquisition
#
# Strategy naming:
#   - BASE       => standard uncertainty acquisition
#   - BASE_Div   => uncertainty + diversity penalty within each acquired batch
# -------------------------
def is_diverse_strategy(strategy: str) -> bool:
    return strategy.endswith("_Div")


def base_strategy_name(strategy: str) -> str:
    return strategy[:-4] if is_diverse_strategy(strategy) else strategy


# -------------------------
# Embeddings for diversity (dropout OFF)
# Computes L2-normalized penultimate features for a set of dataset indices.
# -------------------------
@torch.no_grad()
def compute_embeddings_for_indices(model, indices, shared: SharedLoaders):
    model.eval()
    shared.pool_sampler.set_indices(indices)

    all_embs = []
    all_idxs = []

    for batch in shared.pool_loader:
        x, _, idx = unpack_batch(batch)
        assert idx is not None, "IndexedDataset is required for diversity selection."

        x = x.to(DEVICE, non_blocking=True)
        _, feats = model(x, return_features=True)
        feats = feats.float()
        feats = feats / (feats.norm(dim=1, keepdim=True) + DIVERSE_EPS)

        all_embs.append(feats.cpu())
        all_idxs.append(idx.cpu())

    embs = torch.cat(all_embs, dim=0)
    idxs = torch.cat(all_idxs, dim=0)
    return idxs, embs


# -------------------------
# Greedy diversity-penalized selection
# Objective per step:
#   argmax_x  score(x) - lambda * max_{x' in selected} cos(e(x), e(x'))
# -------------------------
def select_batch_uncertainty_diversity(
    cand_scores: torch.Tensor,  # (M,)
    cand_embs: torch.Tensor,  # (M, E) L2-normalized
    cand_indices: torch.Tensor,  # (M,) dataset indices aligned to scores/embs
    batch_size: int = 10,
    lambda_div: float = 0.5,
):
    scores = cand_scores.clone().float()
    embs = cand_embs.clone().float()

    chosen = []

    first = torch.argmax(scores).item()
    chosen.append(first)

    max_sim = (embs @ embs[first].unsqueeze(1)).squeeze(1)

    for _ in range(1, batch_size):
        obj = scores - lambda_div * max_sim
        obj[chosen] = -1e9
        nxt = torch.argmax(obj).item()
        chosen.append(nxt)

        sim_to_new = (embs @ embs[nxt].unsqueeze(1)).squeeze(1)
        max_sim = torch.maximum(max_sim, sim_to_new)

    acquired_dataset_indices = cand_indices[torch.tensor(chosen)].tolist()
    return acquired_dataset_indices


# -------------------------
# Novel-A batch acquisition wrapper
# Inputs:
#   - pool_idx: current pool indices (global dataset indices)
#   - scores: uncertainty scores aligned with pool_idx order
# Steps:
#   1) shortlist top-M by uncertainty
#   2) compute embeddings for shortlist (dropout OFF)
#   3) greedy select k with diversity penalty
# -------------------------
def acquire_diverse_batch_from_pool(
    model,
    pool_idx,
    scores: np.ndarray,
    shared: SharedLoaders,
    k: int,
    top_m: int = DIVERSE_TOP_M,
    lambda_div: float = DIVERSE_LAMBDA,
):
    assert len(scores) == len(pool_idx), "Scores must align with pool_idx order."

    M = min(top_m, len(pool_idx))
    if M <= k:
        top_local = np.argpartition(scores, -k)[-k:]
        top_local = top_local[np.argsort(scores[top_local])[::-1]]
        return [pool_idx[i] for i in top_local]

    topM_local = np.argpartition(scores, -M)[-M:]
    topM_local = topM_local[np.argsort(scores[topM_local])[::-1]]

    cand_indices_list = [pool_idx[i] for i in topM_local]
    cand_scores_np = scores[topM_local]

    idxs, embs = compute_embeddings_for_indices(model, cand_indices_list, shared)

    score_map = {int(i): float(s) for i, s in zip(cand_indices_list, cand_scores_np)}
    aligned_scores = torch.tensor([score_map[int(i)] for i in idxs.tolist()], dtype=torch.float32)

    acquired = select_batch_uncertainty_diversity(
        cand_scores=aligned_scores,
        cand_embs=embs,
        cand_indices=idxs,
        batch_size=k,
        lambda_div=lambda_div,
    )
    return acquired


# -------------------------
# Pool scoring (uncertainty strategies)
# Returns scores aligned to pool_idx order.
#
# NOTE:
# - strategy must be a base name (no "_Div" suffix).
# - Uses MC dropout (dropout ON) with T samples to score uncertainty.
# -------------------------
def score_pool(model, pool_idx, strategy, shared: SharedLoaders, T=MC_SAMPLES):
    shared.pool_sampler.set_indices(pool_idx)

    all_scores = []
    with torch.inference_mode():
        for batch in shared.pool_loader:
            x, _, _ = unpack_batch(batch)
            x = x.to(DEVICE, non_blocking=True)
            mean_probs, probs_T = predictive_probs(model, x, T=T)

            if strategy == "MaxEntropy":
                s = -(mean_probs * torch.log(mean_probs + EPS)).sum(dim=1)

            elif strategy == "VarRatios":
                s = 1.0 - mean_probs.max(dim=1).values

            elif strategy == "MeanSTD":
                s = probs_T.std(dim=0, unbiased=False).mean(dim=1)

            elif strategy == "BALD":
                ent_mean = -(mean_probs * torch.log(mean_probs + EPS)).sum(dim=1)
                ent_each = -(probs_T * torch.log(probs_T + EPS)).sum(dim=2).mean(dim=0)
                s = ent_mean - ent_each
            else:
                raise ValueError(strategy)

            all_scores.append(s.cpu())

    return torch.cat(all_scores, dim=0).numpy()


# -------------------------
# Active learning loop (core experiment loop)
#
# Acquisition:
#   - BASE: top-k by uncertainty score
#   - BASE_Div: Novel-A (top-M shortlist + greedy diversity penalty)
# -------------------------
def run_active_learning_once(
    test_loader,
    shared: SharedLoaders,
    init_idx,
    pool_idx,
    weight_decay,
    strategy,
    seed,
    init_state_dict=None,
):
    set_seed(seed)
    train_idx = list(init_idx)
    pool_idx = list(pool_idx)

    acc_curve = []

    train_seed0 = 10_000 * seed + 0

    model = train_model(
        train_idx,
        weight_decay,
        TRAIN_STEPS,
        shared,
        init_seed=train_seed0,
        shuffle_seed=train_seed0,
        init_state_dict=init_state_dict,
    )
    acc_curve.append(accuracy(model, test_loader, T=MC_SAMPLES))

    for _round in range(1, TOTAL_ROUNDS + 1):
        base = base_strategy_name(strategy)
        scores = score_pool(model, pool_idx, base, shared, T=MC_SAMPLES)

        if is_diverse_strategy(strategy):
            acquired = acquire_diverse_batch_from_pool(
                model=model,
                pool_idx=pool_idx,
                scores=scores,
                shared=shared,
                k=ACQUISITION_SIZE,
                top_m=DIVERSE_TOP_M,
                lambda_div=DIVERSE_LAMBDA,
            )
        else:
            k = ACQUISITION_SIZE
            top_local = np.argpartition(scores, -k)[-k:]
            top_local = top_local[np.argsort(scores[top_local])[::-1]]
            acquired = [pool_idx[i] for i in top_local]

        acquired_set = set(acquired)
        train_idx.extend(acquired)
        pool_idx = [i for i in pool_idx if i not in acquired_set]

        train_seed = 10_000 * seed + _round

        model = train_model(
            train_idx,
            weight_decay,
            TRAIN_STEPS,
            shared,
            init_seed=train_seed0,
            shuffle_seed=train_seed,
            init_state_dict=init_state_dict,
        )
        acc_curve.append(accuracy(model, test_loader, T=MC_SAMPLES))

        if _round % LOG_EVERY_ROUNDS == 0 or _round == TOTAL_ROUNDS:
            print(
                f"[{strategy} | seed={seed}] "
                f"round {_round}/{TOTAL_ROUNDS} acc={acc_curve[-1]*100:.2f}%"
            )

    return acc_curve


# -------------------------
# Weight decay tuning (validation)
# - Trains on init_idx for TUNE_STEPS updates per candidate WD
# - Selects WD by validation MC-NLL (MC dropout ON)
# -------------------------
def tune_weight_decay(init_idx, val_idx, seed, shared: SharedLoaders, init_state_dict=None):
    shared.val_sampler.set_indices(val_idx)

    if init_state_dict is None:
        set_seed(seed)
        base_model = SimpleBCNN()
        init_state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}

    best_wd = None
    best_val = float("inf")
    tol = 1e-6

    for wd in WEIGHT_DECAY_GRID:
        model = train_model(
            init_idx,
            wd,
            TUNE_STEPS,
            shared,
            init_state_dict=init_state_dict,
            init_seed=12345 + seed,
            shuffle_seed=67890 + seed,
        )

        set_torch_seed_only(99999 + seed)
        val = mc_nll(model, shared.val_loader, T=MC_SAMPLES)
        print(f"WD={wd:.0e} | val=MC-NLL={val:.4f}")


        if (val < best_val - tol) or (abs(val - best_val) <= tol and (best_wd is None or wd < best_wd)):
            best_val = val
            best_wd = wd

    print(f"Selected best weight decay: {best_wd:.0e} (val={best_val:.4f})")
    return best_wd


# -------------------------
# Strategy runner (resume-capable)
#
# Saves per-strategy results as:
#   results_{strategy}.npz
#
# repeat_override:
#   - When set, runs exactly that repeat index and enforces sequential saving.
# -------------------------
def run_one_strategy(
    experiment_id,
    strategy,
    train_full,
    test_loader,
    shared: SharedLoaders,
    overwrite=False,
    fixed_weight_decay: float | None = None,
    repeat_override: int | None = None,
):
    save_path = results_path(experiment_id, strategy)

    if overwrite and os.path.exists(save_path):
        os.remove(save_path)
        print(f"[OVERWRITE] Deleted existing {save_path}")

    curves_done, wds_done, seeds_done = [], [], []

    if (not overwrite) and os.path.exists(save_path):
        with np.load(save_path, allow_pickle=True) as old:
            curves_done = [c.copy() for c in old["curves"]]
            wds_done = old["best_wd_per_repeat"].astype(np.float64).tolist()
            seeds_done = old["seeds"].astype(np.int64).tolist()
        print(f"[RESUME] Found {len(curves_done)}/{NUM_REPEATS} repeats in {save_path}")

    start_repeat = len(curves_done)

    if repeat_override is not None:
        r_target = int(repeat_override)

        if r_target < start_repeat:
            print(f"[SKIP] Repeat {r_target+1} already completed for {strategy}.")
            return
        if r_target != start_repeat:
            raise ValueError(
                f"Out-of-order repeat_override={r_target} but file has {start_repeat} repeats. "
                f"Run repeats in order (0,1,2,...) or delete the results file."
            )

        r_iter = range(r_target, r_target + 1)
    else:
        r_iter = range(start_repeat, NUM_REPEATS)

    for r in r_iter:
        repeat_seed = SEED + r
        print("\n" + "=" * 60)
        print(
            f"Experiment={experiment_id} | Strategy={strategy} | "
            f"Repeat {r+1}/{NUM_REPEATS} | seed={repeat_seed}"
        )

        print("=" * 60)

        init_idx, val_idx, pool_idx = split_indices(train_full, repeat_seed)

        # Fixed initialization per repeat (reproduction-style)
        train_seed0 = 10_000 * repeat_seed + 0
        set_torch_seed_only(train_seed0)
        base_model = SimpleBCNN()
        init_state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}

        if fixed_weight_decay is None:
            best_wd = tune_weight_decay(
                init_idx,
                val_idx,
                seed=repeat_seed,
                shared=shared,
                init_state_dict=init_state_dict,
            )
        else:
            best_wd = float(fixed_weight_decay)
            print(f"Using fixed weight decay: {best_wd:.0e}")

        acc_curve = run_active_learning_once(
            test_loader=test_loader,
            shared=shared,
            init_idx=init_idx,
            pool_idx=pool_idx,
            weight_decay=best_wd,
            strategy=strategy,
            seed=repeat_seed,
            init_state_dict=init_state_dict,
        )

        curves_done.append(np.array(acc_curve, dtype=np.float32))
        wds_done.append(best_wd)
        seeds_done.append(repeat_seed)

        atomic_savez(
            save_path,
            strategy=strategy,
            curves=np.stack(curves_done, axis=0).astype(np.float32),
            best_wd_per_repeat=np.array(wds_done, dtype=np.float64),
            seeds=np.array(seeds_done, dtype=np.int64),
        )

        print(f"[SAVED] {save_path} (repeats completed: {len(curves_done)}/{NUM_REPEATS})")

    print(f"\n[DONE] Experiment={experiment_id} Strategy={strategy} complete.")



# -------------------------
# Summaries + plots
# -------------------------
def acquired_axis():
    return np.arange(0, (TOTAL_ROUNDS + 1) * ACQUISITION_SIZE, ACQUISITION_SIZE)


# -------------------------
# Summary writer (5.1-style summary)
# Used by Novel-A to aggregate results_{strategy}.npz into one summary file.
# -------------------------
def summarize_experiment(experiment_id: str, strategies=None):
    if strategies is None:
        strategies = NOVEL_A_STRATEGIES
    x = acquired_axis()
    results = {}
    for strat in strategies:
        data = _load_npz_or_raise(results_path(experiment_id, strat))
        curves = data["curves"].astype(np.float32)
        results[strat] = {"curves": curves, "mean": curves.mean(axis=0), "std": curves.std(axis=0)}
    atomic_savez(summary_path(experiment_id), x_acquired=x, results=results)

# -------------------------
# Table helper: acquired images needed to reach target accuracies
# This is the table you call in the Novel-A runner block.
# -------------------------
def table_accuracy_thresholds(experiment_id: str, strategies=None, acc_targets=(0.85, 0.90)):
    if strategies is None:
        strategies = NOVEL_A_STRATEGIES

    data = _load_npz_or_raise(summary_path(experiment_id))
    x = data["x_acquired"]
    results = data["results"].item()

    out = {}
    for strat in strategies:
        mean_acc = results[strat]["mean"]
        out[strat] = {}
        for a in acc_targets:
            hit = np.where(mean_acc >= a)[0]
            out[strat][a] = int(x[hit[0]]) if len(hit) > 0 else None

    print("\nAcquired images to reach accuracy:")
    header = "strategy".ljust(14) + "".join([f"{int(100*a)}%".rjust(8) for a in acc_targets])
    print(header)
    print("-" * len(header))
    for strat in strategies:
        row = strat.ljust(14)
        for a in acc_targets:
            row += str(out[strat][a]).rjust(8)
        print(row)

    atomic_savez(
        os.path.join(get_results_dir(experiment_id), "table_accuracy_thresholds.npz"),
        table=out,
        x_acquired=x,
        acc_targets=np.array(acc_targets, dtype=np.float64),
    )
    return out


# -------------------------
# Figure: all strategies on one plot (mean curves)
# Used by Novel-A runner.
# -------------------------
def plot_figure_accuracy(experiment_id: str, strategies=None):
    if strategies is None:
        strategies = NOVEL_A_STRATEGIES

    data = _load_npz_or_raise(summary_path(experiment_id))
    x = data["x_acquired"]
    results = data["results"].item()

    label_map = {
        "BALD": "BALD",
        "BALD_Div": "BALD + Diversity",
        "VarRatios": "Var Ratios",
        "VarRatios_Div": "Var Ratios + Diversity",
        "MaxEntropy": "Max Entropy",
        "MaxEntropy_Div": "Max Entropy + Diversity",
        "MeanSTD": "Mean STD",
        "MeanSTD_Div": "Mean STD + Diversity",
    }

    plt.figure(figsize=(6.2, 4.8))
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=1.0, alpha=0.25)

    for strat in strategies:
        mean = 100.0 * results[strat]["mean"]
        ax.plot(x, mean, linewidth=2.2, label=label_map.get(strat, strat))

    ax.set_xlim(0, int(x[-1]))
    ax.set_ylim(80, 100)
    ax.set_yticks(np.arange(80, 101, 2))
    ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=1.0, fontsize=9)

    for spine in ax.spines.values():
        spine.set_visible(False)

    out = figure_path(experiment_id)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

# -------------------------
# Pair plots: BASE vs BASE_Div (mean curves only)
# One output figure per base strategy.
# -------------------------
def plot_diversity_pair_figures(
    experiment_id: str,
    base_strategies=None,
):
    if base_strategies is None:
        base_strategies = BASE_STRATEGIES


    data = _load_npz_or_raise(summary_path(experiment_id))
    x = data["x_acquired"]
    results = data["results"].item()

    out_dir = get_results_dir(experiment_id)
    os.makedirs(out_dir, exist_ok=True)

    pretty = {
        "BALD": "BALD",
        "VarRatios": "Var Ratios",
        "MaxEntropy": "Max Entropy",
        "MeanSTD": "Mean STD",
    }

    for base in base_strategies:
        div = f"{base}_Div"
        if base not in results or div not in results:
            print(f"[SKIP] Missing in summary: {base} or {div}")
            continue

        base_y = 100.0 * np.asarray(results[base]["mean"], dtype=np.float32)
        div_y = 100.0 * np.asarray(results[div]["mean"], dtype=np.float32)

        plt.figure(figsize=(6.2, 4.8))
        ax = plt.gca()
        ax.set_axisbelow(True)
        ax.grid(True, which="major", linestyle="--", linewidth=1.0, alpha=0.25)

        ax.plot(x, base_y, linewidth=2.5, label=pretty.get(base, base))
        ax.plot(x, div_y, linewidth=2.5, label=f"{pretty.get(base, base)} + Diversity")

        ax.set_xlim(0, int(x[-1]))
        ax.set_ylim(80, 100)
        ax.set_yticks(np.arange(80, 101, 2))
        ax.set_xlabel("Acquired images")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"{pretty.get(base, base)} vs {pretty.get(base, base)} + Diversity")

        ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=1.0, fontsize=9)

        for spine in ax.spines.values():
            spine.set_visible(False)

        out_path = os.path.join(out_dir, f"figure_{experiment_id}_{base}_pair.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")


# -------------------------
# Main (Novel Extension A only)
# -------------------------
if __name__ == "__main__":
    # MNIST datasets
    base_train = MNIST(root="./data", train=True, download=True, transform=TRANSFORM)
    base_test = MNIST(root="./data", train=False, download=True, transform=TRANSFORM)

    # Wrap train set so pool loader yields dataset indices (needed for diversity selection)
    train_full = IndexedDataset(base_train)

    # Test set does not need indices
    test_dataset = base_test

    shared = build_shared_loaders(train_full)
    test_loader = make_test_loader(test_dataset)

    # -------------------------
    # Novel Extension A runner
    # Protocol:
    #   1) For each repeat seed: tune weight decay once (shared across strategies)
    #   2) Run all strategies with that tuned WD fixed for the repeat
    #   3) Summarize + plots + threshold tables
    # -------------------------
    tuned_wd_per_repeat = {}  # seed -> wd

    # 1) Tune WD once per seed
    for r in range(NUM_REPEATS):
        repeat_seed = SEED + r
        init_idx, val_idx, _ = split_indices(train_full, repeat_seed)

        # Build the exact same init weights used by the AL runs for this repeat
        train_seed0 = 10_000 * repeat_seed + 0
        set_torch_seed_only(train_seed0)
        base_model = SimpleBCNN()
        init_state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}

        print("\n" + "=" * 60)
        print(f"[WD-TUNE-ONCE] Repeat {r+1}/{NUM_REPEATS} | seed={repeat_seed} | tuning once (shared across strategies)")
        print("=" * 60)

        wd = tune_weight_decay(
            init_idx,
            val_idx,
            seed=repeat_seed,
            shared=shared,
            init_state_dict=init_state_dict,
        )
        tuned_wd_per_repeat[repeat_seed] = wd
        print(f"[WD-FIXED] seed={repeat_seed} -> wd={wd:.0e}")

    # 2) Run each strategy with the fixed tuned WD per repeat
    for strategy in NOVEL_A_STRATEGIES:
        for r in range(NUM_REPEATS):
            repeat_seed = SEED + r
            wd = tuned_wd_per_repeat[repeat_seed]

            print("\n" + "=" * 60)
            print(
                f"[RUN-FIXED-WD] exp={NOVEL_A_EXPERIMENT_ID} | strat={strategy} | "
                f"repeat={r+1}/{NUM_REPEATS} | seed={repeat_seed} | wd={wd:.0e}"
            )
            print("=" * 60)

            run_one_strategy(
                NOVEL_A_EXPERIMENT_ID,
                strategy,
                train_full,
                test_loader,
                shared=shared,
                overwrite=(r == 0),      # overwrite only on first repeat for this strategy
                fixed_weight_decay=wd,
                repeat_override=r,
            )

    summarize_experiment(NOVEL_A_EXPERIMENT_ID, strategies=NOVEL_A_STRATEGIES)

    # All strategies in one plot
    plot_figure_accuracy(NOVEL_A_EXPERIMENT_ID, strategies=NOVEL_A_STRATEGIES)

    # BASE vs BASE_Div pair plots
    plot_diversity_pair_figures(
        NOVEL_A_EXPERIMENT_ID,
        base_strategies=BASE_STRATEGIES,
    )


    # Accuracy targets table
    table_accuracy_thresholds(
        NOVEL_A_EXPERIMENT_ID,
        strategies=NOVEL_A_STRATEGIES,
        acc_targets=(0.85, 0.90, 0.95, 0.96),
    )
