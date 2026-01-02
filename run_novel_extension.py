# ============================================================
# Active Learning with MC Dropout (MNIST) — Experiments 5.1 / 5.2
# Reproducing results from Y. Gal, R. Islam, Z. Ghahramani (2017). *Deep Bayesian Active Learning with Image Data*
#
# ----------------------------
# ✅ NOTES ON CHANGES (Novel Extension A: Batch Diversity / BatchBALD-style selection)
# ----------------------------
# [CHANGE 1] Added an IndexedDataset wrapper so DataLoaders can return (x, y, idx).
#            This enables diversity-aware batch selection without guessing indices from loader order.
#
# [CHANGE 2] Updated training/eval/scoring loops to accept batches of either (x, y) or (x, y, idx).
#            (We simply ignore idx when not needed.)
#
# [CHANGE 3] Updated SimpleBCNN.forward() to optionally return penultimate features for embeddings.
#            (logits, features) when return_features=True.
#
# [CHANGE 4] Added a new acquisition strategy suffix: "<BASE>_Div"
#            Example: "BALD_Div" uses BALD uncertainty + greedy diversity selection within each batch of 10.
#
# [CHANGE 5] Added NOVEL EXTENSION RUN FLAGS so you can run:
#            - pure reproduction (5_1, 5_2) exactly as before, OR
#            - a separate "novel_A" experiment comparing BALD vs BALD_Div (+ Random optional).
#
# [CHANGE 6] Added diversity hyperparameters:
#            DIVERSE_TOP_M (candidate set size), DIVERSE_LAMBDA (diversity strength).
#            Embeddings are computed with dropout OFF (model.eval()), while uncertainty uses dropout ON (model.train()).
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
import torch.optim as optim

from torch.utils.data import DataLoader, Sampler, Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# -------------------------
# Silence PyTorch DataLoader cleanup spam (keeps workers=4 + persistent=True)
# The silence_pytorch_dataloader_cleanup_error and silence_multiprocessing_connection_cleanup_error
# functions are AI-generated code to suppress benign errors during DataLoader cleanup.
# This does not affect the functionality of the active learning experiments.
# -------------------------

# Silence DataLoader multiprocessing cleanup AssertionError - AI-generated code
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
            # "can only test a child process"
            pass
    cls.__del__ = safe_del

# Silence multiprocessing connection cleanup OSError - AI-generated code
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
            if getattr(e, "errno", None) == 9:  # Bad file descriptor
                return
            raise
    base.__del__ = safe_del

# Apply the silencing functions
silence_pytorch_dataloader_cleanup_error()
silence_multiprocessing_connection_cleanup_error()


# -------------------------
# [CHANGE 1] Dataset wrapper to expose sample indices
# -------------------------
class IndexedDataset(Dataset):
    """
    Wrap any dataset so __getitem__(idx) returns (x, y, idx).
    This is needed for diversity-aware batch selection.
    """
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
# [CHANGE 2] Unified batch unpacking helper
# -------------------------
def unpack_batch(batch):
    """
    Supports datasets that return either (x, y) or (x, y, idx).
    Returns: x, y, idx_or_none
    """
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        x, y, idx = batch
        return x, y, idx
    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
        x, y = batch
        return x, y, None
    else:
        raise ValueError(f"Unexpected batch format of length {len(batch)}")


# -------------------------
# Experiment constants from the paper
# -------------------------
NUM_CLASSES = 10

INITIAL_POINTS_PER_CLASS = 2
VAL_SIZE = 100

NUM_REPEATS = 3

TOTAL_ROUNDS = 100
ACQUISITION_SIZE = 10

# The paper does not state the exact number of MC samples used, so this is my choice.
MC_SAMPLES = 20

# Training parameters - not explicitly given in the paper
TRAIN_STEPS = 1000
TUNE_STEPS  = 500

LR = 1e-3
WEIGHT_DECAY_GRID = [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

EPS = 1e-10
SEED = 33

BATCH_SIZE_TRAIN = 64
BATCH_SIZE_EVAL  = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = (DEVICE.type == "cuda")

TRANSFORM = transforms.ToTensor()
MNIST = datasets.MNIST

NUM_WORKERS = 4
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4

LOG_EVERY_ROUNDS = 10


# -------------------------
# Experiment routing
# -------------------------
EXP_5_1_STRATEGIES = ["BALD", "VarRatios", "MaxEntropy", "MeanSTD", "Random"]
EXP_5_2_STRATEGIES = ["BALD", "VarRatios", "MaxEntropy"]
MODES_5_2 = ["bayes", "det"]


# -------------------------
# [CHANGE 5] Novel extension run toggles
# -------------------------
RUN_REPRODUCTION_5_1 = False
RUN_REPRODUCTION_5_2 = False

# Novel Extension A: diversity-aware batch selection on MC-dropout classification
RUN_NOVEL_EXTENSION_A = True
NOVEL_A_EXPERIMENT_ID = "novel_A"

# Compare a small set for the novel extension (recommended)
# - BALD (baseline)
# - BALD_Div (NOVEL A)
# - Random (sanity)
NOVEL_A_STRATEGIES = ["BALD", "BALD_Div"]


# -------------------------
# [CHANGE 6] Diversity hyperparameters (Novel A)
# -------------------------
# We compute uncertainty scores over the full pool, then restrict to the top M candidates,
# then greedily select a batch of 10 that is both uncertain and diverse in embedding space.
DIVERSE_TOP_M = 500       # candidate shortlist size (try 200 if slow)
DIVERSE_LAMBDA = 0.5      # diversity strength (try 0.2 / 0.5 / 1.0)
DIVERSE_EPS = 1e-12       # numerical safety for normalization


# -------------------------
# Quick test
# -------------------------
QUICK_TEST = False

if QUICK_TEST:
    NUM_REPEATS = 2
    TOTAL_ROUNDS = 20
    ACQUISITION_SIZE = 10
    TRAIN_STEPS = 100
    TUNE_STEPS  = 75
    MC_SAMPLES  = 10
    WEIGHT_DECAY_GRID = [1e-4]
    LOG_EVERY_ROUNDS = 4

    EXP_5_1_STRATEGIES = ["BALD"]
    EXP_5_2_STRATEGIES = ["BALD"]
    RUN_NOVEL_EXTENSION_A = True
    NOVEL_A_STRATEGIES = ["BALD", "BALD_Div", "Random"]
    DIVERSE_TOP_M = 200


# -------------------------
# Paths
# -------------------------
def get_results_dir(experiment_id: str) -> str:
    return f"results_{experiment_id}_seed_{SEED}"

def is_bayesian_mode(mode: str) -> bool:
    return mode == "bayes"

def results_path(experiment_id: str, strategy: str, mode: str | None = None) -> str:
    base = get_results_dir(experiment_id)
    os.makedirs(base, exist_ok=True)
    if mode is None:
        return os.path.join(base, f"results_{strategy}.npz")
    return os.path.join(base, f"results_{strategy}_{mode}.npz")

def summary_path(experiment_id: str) -> str:
    base = get_results_dir(experiment_id)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"summary_{experiment_id}.npz")

def figure_path_5_1(experiment_id: str) -> str:
    base = get_results_dir(experiment_id)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"figure_{experiment_id}.png")

def figure_path_5_2_strategy(experiment_id: str, strategy: str) -> str:
    base = get_results_dir(experiment_id)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"figure_5_2_{strategy}.png")

def _load_npz_or_raise(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing results file: {path}")
    return np.load(path, allow_pickle=True)

def atomic_savez(path, **kwargs):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **kwargs)
    os.replace(tmp, path)

def reuse_5_1_bayes_as_5_2_bayes(src_exp="5_1", dst_exp="5_2", strategies=None, overwrite=False):
    if strategies is None:
        strategies = ["BALD", "VarRatios", "MaxEntropy"]

    for strat in strategies:
        src = results_path(src_exp, strat, mode=None)
        dst = results_path(dst_exp, strat, mode="bayes")

        if not os.path.exists(src):
            print(f"[REUSE-SKIP] Missing source (run 5.1 first): {src}")
            continue

        if os.path.exists(dst) and not overwrite:
            print(f"[REUSE-KEEP] Already exists: {dst}")
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[REUSE-COPY] {src} -> {dst}")


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
# Samplers
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
    val_sampler   = MutableSubsetSequentialSampler(indices=[])
    pool_sampler  = MutableSubsetSequentialSampler(indices=[])

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
        train_loader=train_loader, train_sampler=train_sampler,
        val_loader=val_loader,     val_sampler=val_sampler,
        pool_loader=pool_loader,   pool_sampler=pool_sampler,
    )


def make_test_loader(test_dataset):
    return DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        **_dataloader_kwargs(),
    )


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
# Model - Bayesian CNN with MC Dropout
# -------------------------
class SimpleBCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 11 * 11, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

        self.drop1 = nn.Dropout(p=0.25)
        self.drop2 = nn.Dropout(p=0.5)

    # [CHANGE 3] return_features option to get embeddings for Novel A diversity
    def forward(self, x, return_features: bool = False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        feats = x  # penultimate features (dim=128)
        logits = self.fc2(feats)
        if return_features:
            return logits, feats
        return logits


# -------------------------
# Training
# -------------------------
def train_model(train_idx, weight_decay, steps, shared: SharedLoaders,
               init_state_dict=None, init_seed=None, shuffle_seed=None):

    if init_seed is not None:
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
# -------------------------
def predictive_probs(model, x, T=MC_SAMPLES, bayesian=True):
    if bayesian:
        model.train()  # dropout ON
        B = x.size(0)
        x_rep = x.repeat(T, 1, 1, 1)
        logits = model(x_rep).view(T, B, NUM_CLASSES)
        probs_T = F.softmax(logits, dim=2)
        return probs_T.mean(dim=0), probs_T
    else:
        model.eval()   # dropout OFF
        logits = model(x)
        return F.softmax(logits, dim=1), None


def accuracy(model, loader, T=MC_SAMPLES, bayesian=True):
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch in loader:
            x, y, _ = unpack_batch(batch)
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            mean_probs, _ = predictive_probs(model, x, T=T, bayesian=bayesian)
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


def det_nll(model, loader):
    model.eval()
    total_nll = 0.0
    total = 0
    with torch.inference_mode():
        for batch in loader:
            x, y, _ = unpack_batch(batch)
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            logits = model(x)
            log_probs = F.log_softmax(logits, dim=1)
            total_nll += F.nll_loss(log_probs, y, reduction="sum").item()
            total += x.size(0)
    return total_nll / total


# -------------------------
# [CHANGE 4] Diversity-aware batch selection utilities (Novel Extension A)
# -------------------------
def is_diverse_strategy(strategy: str) -> bool:
    return strategy.endswith("_Div")

def base_strategy_name(strategy: str) -> str:
    return strategy[:-4] if is_diverse_strategy(strategy) else strategy


@torch.no_grad()
def compute_embeddings_for_indices(model, indices, shared: SharedLoaders):
    """
    Compute normalized embeddings for a list of dataset indices.
    IMPORTANT: embeddings are computed with dropout OFF (model.eval()) for stability.
    Returns:
      idxs: (M,) torch.int64
      embs: (M, E) torch.float32 (L2-normalized)
    """
    model.eval()  # dropout OFF
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


def select_batch_uncertainty_diversity(
    cand_scores: torch.Tensor,   # (M,) higher = more uncertain
    cand_embs: torch.Tensor,     # (M, E) L2-normalized
    cand_indices: torch.Tensor,  # (M,) dataset indices aligned to cand_scores/cand_embs
    batch_size: int = 10,
    lambda_div: float = 0.5,
):
    """
    Greedy selection maximizing: score - lambda * max_cosine_similarity_to_selected
    (penalizes redundancy).
    """
    scores = cand_scores.clone().float()
    embs = cand_embs.clone().float()

    M = scores.numel()
    chosen = []

    # Pick most uncertain first
    first = torch.argmax(scores).item()
    chosen.append(first)

    # Maintain max similarity to selected set for each candidate
    max_sim = (embs @ embs[first].unsqueeze(1)).squeeze(1)  # (M,)

    for _ in range(1, batch_size):
        obj = scores - lambda_div * max_sim
        obj[chosen] = -1e9  # prevent reselect
        nxt = torch.argmax(obj).item()
        chosen.append(nxt)

        sim_to_new = (embs @ embs[nxt].unsqueeze(1)).squeeze(1)
        max_sim = torch.maximum(max_sim, sim_to_new)

    acquired_dataset_indices = cand_indices[torch.tensor(chosen)].tolist()
    return acquired_dataset_indices


def acquire_diverse_batch_from_pool(
    model,
    pool_idx,
    scores: np.ndarray,
    shared: SharedLoaders,
    k: int,
    top_m: int = DIVERSE_TOP_M,
    lambda_div: float = DIVERSE_LAMBDA,
):
    """
    Implements Novel Extension A:
    1) shortlist top_m points by uncertainty score,
    2) compute embeddings for shortlist with dropout OFF,
    3) greedily select k diverse points.
    """
    assert len(scores) == len(pool_idx), "Scores must align with pool_idx order."

    M = min(top_m, len(pool_idx))
    if M <= k:
        # Not enough points left to diversify; fall back to top-k by uncertainty
        top_local = np.argpartition(scores, -k)[-k:]
        top_local = top_local[np.argsort(scores[top_local])[::-1]]
        return [pool_idx[i] for i in top_local]

    # shortlist top M by uncertainty
    topM_local = np.argpartition(scores, -M)[-M:]
    topM_local = topM_local[np.argsort(scores[topM_local])[::-1]]  # descending

    cand_indices_list = [pool_idx[i] for i in topM_local]
    cand_scores_np = scores[topM_local]

    # embeddings (dropout OFF)
    idxs, embs = compute_embeddings_for_indices(model, cand_indices_list, shared)

    # align scores to the returned idx order (robust in case loader order differs)
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
# Acquisition scoring (base strategies from paper)
# -------------------------
def score_pool(model, pool_idx, strategy, shared: SharedLoaders, T=MC_SAMPLES, bayesian=True):
    """
    Returns scores aligned to pool_idx order.
    NOTE: strategy here should be a BASE strategy name (no "_Div" suffix).
    """
    shared.pool_sampler.set_indices(pool_idx)

    all_scores = []
    with torch.inference_mode():
        for batch in shared.pool_loader:
            x, _, _ = unpack_batch(batch)
            x = x.to(DEVICE, non_blocking=True)
            mean_probs, probs_T = predictive_probs(model, x, T=T, bayesian=bayesian)

            if strategy == "MaxEntropy":
                s = -(mean_probs * torch.log(mean_probs + EPS)).sum(dim=1)

            elif strategy == "VarRatios":
                s = 1.0 - mean_probs.max(dim=1).values

            elif strategy == "MeanSTD":
                if not bayesian:
                    s = -(mean_probs * torch.log(mean_probs + EPS)).sum(dim=1)
                else:
                    s = probs_T.std(dim=0, unbiased=False).mean(dim=1)

            elif strategy == "BALD":
                if not bayesian:
                    # Paper doesn't define deterministic BALD; keep prior behavior (random == uniform selection)
                    s = torch.rand(mean_probs.size(0), device=mean_probs.device)
                else:
                    ent_mean = -(mean_probs * torch.log(mean_probs + EPS)).sum(dim=1)
                    ent_each = -(probs_T * torch.log(probs_T + EPS)).sum(dim=2).mean(dim=0)
                    s = ent_mean - ent_each

            else:
                raise ValueError(strategy)

            all_scores.append(s.cpu())

    return torch.cat(all_scores, dim=0).numpy()


# -------------------------
# Active learning loop
# -------------------------
def run_active_learning_once(test_loader, shared: SharedLoaders,
                            init_idx, pool_idx, weight_decay, strategy, seed, bayesian=True):

    set_seed(seed)
    train_idx = list(init_idx)
    pool_idx  = list(pool_idx)

    acc_curve = []

    train_seed0 = 10_000 * seed + 0
    model = train_model(train_idx, weight_decay, TRAIN_STEPS, shared,
                        init_seed=train_seed0, shuffle_seed=train_seed0)
    acc_curve.append(accuracy(model, test_loader, T=MC_SAMPLES, bayesian=bayesian))

    for _round in range(1, TOTAL_ROUNDS + 1):
        if strategy == "Random":
            acquired = random.sample(pool_idx, ACQUISITION_SIZE)

        else:
            # [CHANGE 4] Handle "<BASE>_Div" strategy here
            base = base_strategy_name(strategy)

            scores = score_pool(model, pool_idx, base, shared, T=MC_SAMPLES, bayesian=bayesian)

            if is_diverse_strategy(strategy):
                # NOVEL A: uncertainty + diversity
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
                # original behavior: top-k by score
                k = ACQUISITION_SIZE
                top_local = np.argpartition(scores, -k)[-k:]
                top_local = top_local[np.argsort(scores[top_local])[::-1]]
                acquired = [pool_idx[i] for i in top_local]

        acquired_set = set(acquired)
        train_idx.extend(acquired)
        pool_idx = [i for i in pool_idx if i not in acquired_set]

        train_seed = 10_000 * seed + _round
        model = train_model(train_idx, weight_decay, TRAIN_STEPS, shared,
                            init_seed=train_seed0, shuffle_seed=train_seed)
        acc_curve.append(accuracy(model, test_loader, T=MC_SAMPLES, bayesian=bayesian))

        if _round % LOG_EVERY_ROUNDS == 0 or _round == TOTAL_ROUNDS:
            print(f"[{strategy} | bayes={bayesian} | seed={seed}] round {_round}/{TOTAL_ROUNDS} acc={acc_curve[-1]*100:.2f}%")

    return acc_curve


def tune_weight_decay(init_idx, val_idx, seed, shared: SharedLoaders, bayesian=True):
    shared.val_sampler.set_indices(val_idx)

    set_seed(seed)
    base_model = SimpleBCNN()
    init_state = {k: v.clone() for k, v in base_model.state_dict().items()}

    best_wd = None
    best_val = float("inf")
    tol = 1e-6

    for wd in WEIGHT_DECAY_GRID:
        model = train_model(init_idx, wd, TUNE_STEPS, shared,
                            init_state_dict=init_state,
                            init_seed=12345 + seed,
                            shuffle_seed=67890 + seed)

        set_torch_seed_only(99999 + seed)
        val = mc_nll(model, shared.val_loader, T=MC_SAMPLES) if bayesian else det_nll(model, shared.val_loader)
        print(f"WD={wd:.0e} | val={'MC-NLL' if bayesian else 'NLL'}={val:.4f}")

        if (val < best_val - tol) or (abs(val - best_val) <= tol and (best_wd is None or wd < best_wd)):
            best_val = val
            best_wd = wd

    print(f"Selected best weight decay: {best_wd:.0e} (val={best_val:.4f})")
    return best_wd


# -------------------------
# Run one strategy (supports resuming)
# -------------------------
def run_one_strategy(experiment_id, strategy, train_full, test_loader, shared: SharedLoaders, overwrite=False, mode=None, fixed_weight_decay: float | None = None, repeat_override: int | None = None):
    bayesian = True if mode is None else is_bayesian_mode(mode)
    save_path = results_path(experiment_id, strategy, mode)

    if overwrite and os.path.exists(save_path):
        os.remove(save_path)
        print(f"[OVERWRITE] Deleted existing {save_path}")

    curves_done, wds_done, seeds_done = [], [], []

    if (not overwrite) and os.path.exists(save_path):
        with np.load(save_path, allow_pickle=True) as old:
            curves_done = [c.copy() for c in old["curves"]]
            wds_done    = old["best_wd_per_repeat"].astype(np.float64).tolist()
            seeds_done  = old["seeds"].astype(np.int64).tolist()
        print(f"[RESUME] Found {len(curves_done)}/{NUM_REPEATS} repeats in {save_path}")

    start_repeat = len(curves_done)

    # If repeat_override is set, we only run that single repeat index.
    if repeat_override is not None:
        r_target = int(repeat_override)

        # Enforce sequential saving (your main runs r=0,1,2 in order)
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
        print(f"Experiment={experiment_id} | Strategy={strategy} | Mode={mode or 'bayes'} | Repeat {r+1}/{NUM_REPEATS} | seed={repeat_seed}")
        print("=" * 60)

        init_idx, val_idx, pool_idx = split_indices(train_full, repeat_seed)
        # --- NEW: use fixed WD if provided, otherwise tune ---
        if fixed_weight_decay is None:
            best_wd = tune_weight_decay(init_idx, val_idx, seed=repeat_seed, shared=shared, bayesian=bayesian)
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
            bayesian=bayesian,
        )

        curves_done.append(np.array(acc_curve, dtype=np.float32))
        wds_done.append(best_wd)
        seeds_done.append(repeat_seed)

        atomic_savez(
            save_path,
            strategy=strategy,
            mode=(mode if mode is not None else "bayes"),
            curves=np.stack(curves_done, axis=0).astype(np.float32),
            best_wd_per_repeat=np.array(wds_done, dtype=np.float64),
            seeds=np.array(seeds_done, dtype=np.int64),
        )
        print(f"[SAVED] {save_path} (repeats completed: {len(curves_done)}/{NUM_REPEATS})")

    print(f"\n[DONE] Experiment={experiment_id} Strategy={strategy} Mode={mode or 'bayes'} complete.")


# -------------------------
# Summaries + plots
# -------------------------
def acquired_axis():
    return np.arange(0, (TOTAL_ROUNDS + 1) * ACQUISITION_SIZE, ACQUISITION_SIZE)


# [CHANGE 5] Let summarize/plot accept custom strategy lists (for novel_A experiment)
def summarize_5_1(experiment_id: str, strategies=None):
    if strategies is None:
        strategies = EXP_5_1_STRATEGIES
    x = acquired_axis()
    results = {}
    for strat in strategies:
        data = _load_npz_or_raise(results_path(experiment_id, strat, mode=None))
        curves = data["curves"].astype(np.float32)
        results[strat] = {"curves": curves, "mean": curves.mean(axis=0), "std": curves.std(axis=0)}
    atomic_savez(summary_path(experiment_id), x_acquired=x, results=results)


def table_5_1_thresholds(experiment_id: str, strategies=None, error_targets=(0.10, 0.05)):
    if strategies is None:
        strategies = EXP_5_1_STRATEGIES

    data = _load_npz_or_raise(summary_path(experiment_id))
    x = data["x_acquired"]
    results = data["results"].item()

    out = {}
    for strat in strategies:
        stats = results[strat]
        mean_err = 1.0 - stats["mean"]
        out[strat] = {}
        for e in error_targets:
            hit = np.where(mean_err <= e)[0]
            out[strat][e] = int(x[hit[0]]) if len(hit) > 0 else None

    print("\nTable 1 style (acquired images to reach error):")
    header = "strategy".ljust(14) + "10% err".rjust(10) + "5% err".rjust(10)
    print(header)
    print("-" * len(header))
    for strat in strategies:
        print(strat.ljust(14) + str(out[strat][0.10]).rjust(10) + str(out[strat][0.05]).rjust(10))

    atomic_savez(os.path.join(get_results_dir(experiment_id), "table1_thresholds.npz"),
                 table=out, x_acquired=x)
    return out

# New added function for accuracy thresholds
def table_5_1_accuracy_thresholds(experiment_id: str, strategies=None, acc_targets=(0.85, 0.90)):
    if strategies is None:
        strategies = EXP_5_1_STRATEGIES

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
        table=out, x_acquired=x, acc_targets=np.array(acc_targets, dtype=np.float64)
    )
    return out



def plot_figure_5_1(experiment_id: str, strategies=None):
    if strategies is None:
        strategies = EXP_5_1_STRATEGIES

    data = _load_npz_or_raise(summary_path(experiment_id))
    x = data["x_acquired"]
    results = data["results"].item()

    label_map = {
        "BALD": "BALD",
        "BALD_Div": "BALD + Diversity",
        "VarRatios": "Var Ratios",
        "MaxEntropy": "Max Entropy",
        "MeanSTD": "Mean STD",
        "Random": "Random",
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

    out = figure_path_5_1(experiment_id)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def summarize_5_2(experiment_id: str):
    x = acquired_axis()
    results = {}
    for strat in EXP_5_2_STRATEGIES:
        results[strat] = {}
        for mode in MODES_5_2:
            data = _load_npz_or_raise(results_path(experiment_id, strat, mode=mode))
            curves = data["curves"].astype(np.float32)
            results[strat][mode] = {"curves": curves, "mean": curves.mean(axis=0), "std": curves.std(axis=0)}
    atomic_savez(summary_path(experiment_id), x_acquired=x, results=results)


def plot_figure_5_2(experiment_id: str):
    data = _load_npz_or_raise(summary_path(experiment_id))
    x = data["x_acquired"]
    results = data["results"].item()

    BAYES_COLOR = "#d62728"
    DET_COLOR   = "#1f77b4"
    BAND_ALPHA  = 0.18

    for strat in EXP_5_2_STRATEGIES:
        plt.figure(figsize=(6.2, 4.8))
        ax = plt.gca()
        ax.set_axisbelow(True)
        ax.grid(True, which="major", linestyle="--", linewidth=1.0, alpha=0.25)

        if strat == "BALD":
            bayes_label = "BALD"
            det_label = "Deterministic BALD"
        elif strat == "VarRatios":
            bayes_label = "Var Ratios"
            det_label = "Deterministic Var Ratios"
        elif strat == "MaxEntropy":
            bayes_label = "Max Entropy"
            det_label = "Deterministic Max Entropy"

        bayes_mean = 100.0 * results[strat]["bayes"]["mean"]
        bayes_std  = 100.0 * results[strat]["bayes"]["std"]
        det_mean   = 100.0 * results[strat]["det"]["mean"]
        det_std    = 100.0 * results[strat]["det"]["std"]

        ax.plot(x, bayes_mean, color=BAYES_COLOR, linewidth=2.5, label=bayes_label)
        ax.fill_between(x, bayes_mean - bayes_std, bayes_mean + bayes_std, color=BAYES_COLOR, alpha=BAND_ALPHA, linewidth=0)

        ax.plot(x, det_mean, color=DET_COLOR, linewidth=2.5, label=det_label)
        ax.fill_between(x, det_mean - det_std, det_mean + det_std, color=DET_COLOR, alpha=BAND_ALPHA, linewidth=0)

        ax.set_ylim(80, 100)
        ax.set_yticks(np.arange(80, 101, 2))

        max_x = int(x[-1])
        ax.set_xlim(0, max_x)
        ax.set_xticks(np.arange(0, max_x + 1, 100))

        ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=1.0, fontsize=9)

        for spine in ax.spines.values():
            spine.set_visible(False)

        out = figure_path_5_2_strategy(experiment_id, strat)
        plt.tight_layout()
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Base MNIST datasets
    base_train = MNIST(root="./data", train=True, download=True, transform=TRANSFORM)
    base_test  = MNIST(root="./data", train=False, download=True, transform=TRANSFORM)

    # [CHANGE 1] Wrap train set to expose indices (needed for diversity selection)
    train_full = IndexedDataset(base_train)

    # Test set does not need indices (we never diversify on test); keep it simple
    test_dataset = base_test

    shared = build_shared_loaders(train_full)
    test_loader = make_test_loader(test_dataset)

    # QUICK_TEST path unchanged
    if QUICK_TEST:
        exp_5_2 = "5_2_smoke_new"
        # existing 5.2 smoke
        for strategy in EXP_5_2_STRATEGIES:
            for mode in MODES_5_2:
                run_one_strategy(exp_5_2, strategy, train_full, test_loader, shared=shared, overwrite=True, mode=mode)
        summarize_5_2(exp_5_2)
        plot_figure_5_2(exp_5_2)

        # NEW: novel A smoke
        exp_novel = "novel_A_smoke_new"
        for strategy in NOVEL_A_STRATEGIES:
            run_one_strategy(exp_novel, strategy, train_full, test_loader, shared=shared, overwrite=True, mode=None)
        summarize_5_1(exp_novel, strategies=NOVEL_A_STRATEGIES)
        plot_figure_5_1(exp_novel, strategies=NOVEL_A_STRATEGIES)


    else:
        # -------------------------
        # Reproduction: 5.1
        # -------------------------
        if RUN_REPRODUCTION_5_1:
            for strategy in EXP_5_1_STRATEGIES:
                run_one_strategy("5_1", strategy, train_full, test_loader, shared=shared, overwrite=True, mode=None)
            summarize_5_1("5_1", strategies=EXP_5_1_STRATEGIES)
            table_5_1_thresholds("5_1", strategies=EXP_5_1_STRATEGIES, error_targets=(0.10, 0.05))
            plot_figure_5_1("5_1", strategies=EXP_5_1_STRATEGIES)

        # -------------------------
        # Reproduction: 5.2
        # -------------------------
        if RUN_REPRODUCTION_5_2:
            # reuse bayes for 5.2 (copied from 5.1 results)
            reuse_5_1_bayes_as_5_2_bayes("5_1", "5_2", EXP_5_2_STRATEGIES, overwrite=False)

            # 5.2 det only
            for strategy in EXP_5_2_STRATEGIES:
                run_one_strategy("5_2", strategy, train_full, test_loader, shared=shared, overwrite=True, mode="det")
            summarize_5_2("5_2")
            plot_figure_5_2("5_2")

        # -------------------------
        # Novel Extension A: BALD + Diversity
        # -------------------------
        if RUN_NOVEL_EXTENSION_A:
            # Tune WD once per repeat/seed using BALD, then reuse for BALD_Div (and Random if you want).
            tuned_wd_per_repeat = {}  # seed -> wd

            # 1) Tune WD once per seed (repeat)
            for r in range(NUM_REPEATS):
                repeat_seed = SEED + r
                init_idx, val_idx, _ = split_indices(train_full, repeat_seed)

                print("\n" + "=" * 60)
                print(f"[WD-TUNE-ONCE] Repeat {r+1}/{NUM_REPEATS} | seed={repeat_seed} | tuning on BALD")
                print("=" * 60)

                wd = tune_weight_decay(init_idx, val_idx, seed=repeat_seed, shared=shared, bayesian=True)
                tuned_wd_per_repeat[repeat_seed] = wd
                print(f"[WD-FIXED] seed={repeat_seed} -> wd={wd:.0e}")

            # 2) Run each strategy, forcing the tuned WD for each repeat
            for strategy in NOVEL_A_STRATEGIES:
                for r in range(NUM_REPEATS):
                    repeat_seed = SEED + r
                    wd = tuned_wd_per_repeat[repeat_seed]

                    print("\n" + "=" * 60)
                    print(f"[RUN-FIXED-WD] exp={NOVEL_A_EXPERIMENT_ID} | strat={strategy} | repeat={r+1}/{NUM_REPEATS} | seed={repeat_seed} | wd={wd:.0e}")
                    print("=" * 60)

                    run_one_strategy(
                        NOVEL_A_EXPERIMENT_ID,
                        strategy,
                        train_full,
                        test_loader,
                        shared=shared,
                        overwrite=(r == 0),          # overwrite only on first repeat for this strategy
                        mode=None,
                        fixed_weight_decay=wd,       # <-- reuse tuned WD
                        repeat_override=r,           # <-- run exactly this repeat index
                    )

            summarize_5_1(NOVEL_A_EXPERIMENT_ID, strategies=NOVEL_A_STRATEGIES)
            plot_figure_5_1(NOVEL_A_EXPERIMENT_ID, strategies=NOVEL_A_STRATEGIES)

            # Use accuracy targets (85/90/95/96/97/98)
            table_5_1_accuracy_thresholds(
                NOVEL_A_EXPERIMENT_ID,
                strategies=NOVEL_A_STRATEGIES,
                acc_targets=(0.85, 0.90, 0.95, 0.96, 0.97, 0.98),
            )

