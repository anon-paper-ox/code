# ============================================================
# Active Learning with MC Dropout (MNIST) — Experiments 5.1 / 5.2
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
            # "can only test a child process"
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
            if getattr(e, "errno", None) == 9:  # Bad file descriptor
                return
            raise
    base.__del__ = safe_del


silence_pytorch_dataloader_cleanup_error()
silence_multiprocessing_connection_cleanup_error()



# -------------------------
# Experiment constants
# -------------------------
NUM_CLASSES = 10

INITIAL_POINTS_PER_CLASS = 2
VAL_SIZE = 100

NUM_REPEATS = 3

TOTAL_ROUNDS = 100
ACQUISITION_SIZE = 10

MC_SAMPLES = 20

TRAIN_STEPS = 1000
TUNE_STEPS  = 500

LR = 1e-3
WEIGHT_DECAY_GRID = [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

EPS = 1e-10
SEED = 48

BATCH_SIZE_TRAIN = 64
BATCH_SIZE_EVAL  = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = (DEVICE.type == "cuda")

TRANSFORM = transforms.ToTensor()
MNIST = datasets.MNIST

# ✅ requested
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
# Quick smoke test config
# -------------------------
QUICK_TEST = False

if QUICK_TEST:
    NUM_REPEATS = 2
    TOTAL_ROUNDS = 20
    ACQUISITION_SIZE = 10
    TRAIN_STEPS = 200
    TUNE_STEPS  = 100
    MC_SAMPLES  = 10
    WEIGHT_DECAY_GRID = [0.0, 1e-4, 1e-2]
    LOG_EVERY_ROUNDS = 1

    EXP_5_1_STRATEGIES = []
    EXP_5_2_STRATEGIES = ["BALD"]


# -------------------------
# Paths / IO
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
    """
    Samples WITH replacement from indices, returning exactly `num_samples`.
    With drop_last=True, this yields exactly num_samples / batch_size batches.
    """
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
# Model
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

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)


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
    for x, y in shared.train_loader:
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
        model.train()
        B = x.size(0)
        x_rep = x.repeat(T, 1, 1, 1)
        logits = model(x_rep).view(T, B, NUM_CLASSES)
        probs_T = F.softmax(logits, dim=2)
        return probs_T.mean(dim=0), probs_T
    else:
        model.eval()
        logits = model(x)
        return F.softmax(logits, dim=1), None


def accuracy(model, loader, T=MC_SAMPLES, bayesian=True):
    correct = 0
    total = 0
    with torch.inference_mode():
        for x, y in loader:
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
        for x, y in loader:
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
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            logits = model(x)
            log_probs = F.log_softmax(logits, dim=1)
            total_nll += F.nll_loss(log_probs, y, reduction="sum").item()
            total += x.size(0)
    return total_nll / total


# -------------------------
# Acquisition scoring
# -------------------------
def score_pool(model, pool_idx, strategy, shared: SharedLoaders, T=MC_SAMPLES, bayesian=True):
    shared.pool_sampler.set_indices(pool_idx)

    all_scores = []
    with torch.inference_mode():
        for x, _ in shared.pool_loader:
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
                    s = torch.zeros(mean_probs.size(0), device=mean_probs.device)
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
            scores = score_pool(model, pool_idx, strategy, shared, T=MC_SAMPLES, bayesian=bayesian)
            k = ACQUISITION_SIZE
            top_local = np.argpartition(scores, -k)[-k:]
            top_local = top_local[np.argsort(scores[top_local])[::-1]]
            acquired = [pool_idx[i] for i in top_local]

        acquired_set = set(acquired)
        train_idx.extend(acquired)
        pool_idx = [i for i in pool_idx if i not in acquired_set]

        train_seed = 10_000 * seed + _round
        model = train_model(train_idx, weight_decay, TRAIN_STEPS, shared,
                            init_seed=train_seed, shuffle_seed=train_seed)
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
# Run one strategy
# -------------------------
def run_one_strategy(experiment_id, strategy, train_full, test_loader, shared: SharedLoaders, overwrite=False, mode=None):
    bayesian = True if mode is None else is_bayesian_mode(mode)
    save_path = results_path(experiment_id, strategy, mode)

    if overwrite and os.path.exists(save_path):
        os.remove(save_path)
        print(f"[OVERWRITE] Deleted existing {save_path}")

    curves_done, wds_done, seeds_done = [], [], []

    if (not overwrite) and os.path.exists(save_path):
        old = np.load(save_path, allow_pickle=True)
        curves_done = list(old["curves"])
        wds_done    = old["best_wd_per_repeat"].tolist()
        seeds_done  = old["seeds"].tolist()
        print(f"[RESUME] Found {len(curves_done)}/{NUM_REPEATS} repeats in {save_path}")

    start_repeat = len(curves_done)

    for r in range(start_repeat, NUM_REPEATS):
        repeat_seed = SEED + r
        print("\n" + "=" * 60)
        print(f"Experiment={experiment_id} | Strategy={strategy} | Mode={mode or 'bayes'} | Repeat {r+1}/{NUM_REPEATS} | seed={repeat_seed}")
        print("=" * 60)

        init_idx, val_idx, pool_idx = split_indices(train_full, repeat_seed)
        best_wd = tune_weight_decay(init_idx, val_idx, seed=repeat_seed, shared=shared, bayesian=bayesian)

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

def summarize_5_1(experiment_id: str):
    x = acquired_axis()
    results = {}
    for strat in EXP_5_1_STRATEGIES:
        data = _load_npz_or_raise(results_path(experiment_id, strat, mode=None))
        curves = data["curves"].astype(np.float32)
        results[strat] = {"curves": curves, "mean": curves.mean(axis=0), "std": curves.std(axis=0)}
    atomic_savez(summary_path(experiment_id), x_acquired=x, results=results)

def table_5_1_thresholds(experiment_id: str, error_targets=(0.10, 0.05)):
    data = _load_npz_or_raise(summary_path(experiment_id))
    x = data["x_acquired"]
    results = data["results"].item()

    out = {}
    for strat, stats in results.items():
        mean_err = 1.0 - stats["mean"]
        out[strat] = {}
        for e in error_targets:
            hit = np.where(mean_err <= e)[0]
            out[strat][e] = int(x[hit[0]]) if len(hit) > 0 else None

    print("\nTable 1 style (acquired images to reach error):")
    header = "strategy".ljust(12) + "10% err".rjust(10) + "5% err".rjust(10)
    print(header)
    print("-" * len(header))
    for strat in EXP_5_1_STRATEGIES:
        if strat not in out:
            continue
        print(strat.ljust(12) + str(out[strat][0.10]).rjust(10) + str(out[strat][0.05]).rjust(10))

    atomic_savez(os.path.join(get_results_dir(experiment_id), "table1_thresholds.npz"),
                 table=out, x_acquired=x)
    return out

def plot_figure_5_1(experiment_id: str):
    data = _load_npz_or_raise(summary_path(experiment_id))
    x = data["x_acquired"]
    results = data["results"].item()

    label_map = {
        "BALD": "BALD",
        "VarRatios": "Var Ratios",
        "MaxEntropy": "Max Entropy",
        "MeanSTD": "Mean STD",
        "Random": "Random",
    }

    plt.figure(figsize=(6.2, 4.8))
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=1.0, alpha=0.25)

    for strat in EXP_5_1_STRATEGIES:
        mean = 100.0 * results[strat]["mean"]
        std  = 100.0 * results[strat]["std"]
        ax.plot(x, mean, linewidth=2.2, label=label_map.get(strat, strat))

    ax.set_xlabel("Acquired images")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlim(0, int(x[-1]))
    ax.set_ylim(80, 100)
    ax.set_yticks(np.arange(80, 101, 2))
    ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=1.0, fontsize=9)

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

    BAYES_LABEL = "Bayesian CNN"
    DET_LABEL   = "Deterministic CNN"

    titles = {"BALD": "(a) BALD", "VarRatios": "(b) Var Ratios", "MaxEntropy": "(c) Max Entropy"}

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

        ax.set_title(titles.get(strat, strat), fontsize=12)
        ax.set_xlabel("Acquired images")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_ylim(80, 100)
        ax.set_yticks(np.arange(80, 101, 2))

        max_x = int(x[-1])
        ax.set_xlim(0, max_x)
        ax.set_xticks(np.arange(0, max_x + 1, 100))

        ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=1.0, fontsize=9)

        out = figure_path_5_2_strategy(experiment_id, strat)
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

    shared = build_shared_loaders(train_full)
    test_loader = make_test_loader(test_dataset)

    if QUICK_TEST:
        exp_5_2 = "5_2_smoke"
        for strategy in EXP_5_2_STRATEGIES:
            for mode in MODES_5_2:
                run_one_strategy(exp_5_2, strategy, train_full, test_loader, shared=shared, overwrite=True, mode=mode)
        summarize_5_2(exp_5_2)
        plot_figure_5_2(exp_5_2)

    else:
        # 5.1
        for strategy in EXP_5_1_STRATEGIES:
            run_one_strategy("5_1", strategy, train_full, test_loader, shared=shared, overwrite=True, mode=None)
        summarize_5_1("5_1")
        table_5_1_thresholds("5_1", (0.10, 0.05))
        plot_figure_5_1("5_1")

        # reuse bayes for 5.2
        reuse_5_1_bayes_as_5_2_bayes("5_1", "5_2", EXP_5_2_STRATEGIES, overwrite=False)

        # 5.2 det only
        for strategy in EXP_5_2_STRATEGIES:
            run_one_strategy("5_2", strategy, train_full, test_loader, shared=shared, overwrite=True, mode="det")
        summarize_5_2("5_2")
        plot_figure_5_2("5_2")
