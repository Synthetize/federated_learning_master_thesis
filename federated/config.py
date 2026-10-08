"""Configuration of the federated sweep.

Every value in here is cited in the thesis (Implementation): the comments that justify it
are part of the documentation and must not be removed together with the code.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

# Repository root, computed from this file's location and not from os.getcwd(): the Ray
# actors of the simulation do not guarantee the working directory, and in the notebook
# every path was relative. This way the sweep can be launched from any folder.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.environ.get("FL_RESULTS_ROOT", os.path.join(REPO_ROOT, "results"))


@dataclass
class ExperimentConfig:
    # --- Federated setup ---
    num_clients: int = 6               # simulated clients (SuperNodes)
    # 100 and not 50: in diagnostics D1, D2 and D5 the maximum fell on the LAST round,
    # i.e. no run had converged within 50 rounds. At alpha=0.5 seed 43 needed ~180
    # rounds to reach 0.951 (but with the old optimizer).
    num_rounds: int = 100
    fraction_train: float = 1.0        # fraction of clients sampled for training
    fraction_evaluate: float = 1.0     # fraction of clients sampled for evaluation
    min_available_clients: int = 2

    # --- Local hyperparameters (client-side) ---
    local_epochs: int = 1
    batch_size: int = 128
    learning_rate: float = 0.03

    # momentum = 0.0, and this is the change that unlocks the low alphas.
    # With m=0.9 the combination (alpha=0.5, seed=42) was DEAD: loss stuck at ln2 for
    # 200 rounds, accuracy never above 0.5. The test in test_ottimizzatore.ipynb isolated
    # the cause: it is not the step size. Two configurations with the SAME effective step
    # lr/(1-m) = 0.03 give opposite outcomes -- lr=0.003 with m=0.9 stays at 0.500,
    # lr=0.03 with m=0.0 reaches 0.703 -- so the problem is the accumulation itself.
    # Reason: a nearly single-class client produces fully coherent gradients, momentum
    # amplifies them (up to 10x with m=0.9), the local model runs far from the global
    # weights, and the average of six models that ran off in different directions is a
    # constant predictor.
    # Cost of the change where things already worked: at alpha=10 it goes from 0.961 to
    # 0.952, one point. Gain at alpha=0.5 seed 42: from 0.500 (dead) to 0.703.
    momentum: float = 0.0

    # --- Aggregation: FedProx ---
    fedprox_mu: float = 0.01           # weight of the proximal term. 0.0 => FedAvg

    # --- Data heterogeneity (Dirichlet partitioning) ---
    # Folder names use f"alpha_{alpha:g}" (see paths.py), so 10.0 -> "10" and 1.0 -> "1":
    # a single convention for every script. In the notebook "alpha_10" (int) for the
    # federated runs coexisted with "alpha_10.0" (float) for the baselines, and since
    # dca.py normalizes both with float() the two folders collapsed onto the same key,
    # overwriting each other.
    alphas: list = field(default_factory=lambda: [10.0, 1.0, 0.5, 0.4, 0.3, 0.2])
    # alpha = 10 and not 100 as the anchor
    #   near-IID: at 10 the mean per-client class purity is 58.1% against the 50% of the
    #   pure IID case, so in the thesis it must be described as "mildly non-IID", not as
    #   an IID baseline. In exchange the grid is better spaced in realized heterogeneity:
    #   the purity jumps become 15.7 / 6.9 / 8.4 points instead of 21.2 / 6.9 / 8.4.
    # alpha = 0.2 and not 0.1: at 0.1 the probability that a single draw satisfies
    #   min_partition_size=500 is ~1.8%, and the partitions that survive the constraint
    #   look like those of 0.2 anyway (ratio 11.3x against 10.5x).
    min_partition_size: int = 500
    # Floor on the partition size: with the 80/20 split inside the client it guarantees
    # n_train >= 400, hence sample rate q = 128/400 = 0.32 < 1. It limits quantity skew
    # but does not remove it: at alpha=0.2 the median ratio between the largest and the
    # smallest client stays ~11:1.
    partition_max_retries: int = 50    # retry loop outside the DirichletPartitioner

    # --- Seeds: independent Dirichlet draws at the same alpha ---
    # Ten and not three. With three draws the only available statistic was the half-range,
    # an ad hoc measure to explain and defend; with ten we report mean +- 1.96*SE, which is
    # the standard. The standard error drops by 45%. Note that the simulation is not
    # bit-reproducible (the Ray actors and the Opacus generator are not reseeded), so the
    # means over seeds must be read as estimates with variance -- which is also what makes
    # the fixed-partition replicates free (see paths.set_replicate).
    seeds: list = field(default_factory=lambda: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51])

    # --- Differential Privacy: record-level DP-SGD on the client side (Opacus) ---
    dp_enabled: bool = True
    run_baseline: bool = True          # run without DP (epsilon = inf), reference
    # Grid revised after the first sweep. Dropped 16 as redundant: at alpha=10 epsilons
    # 8/16/32 gave 0.922/0.925/0.930, three runs for 0.8 points. Added 0.5 because the
    # steepest stretch of the curve was between 1 and 2 (0.751 -> 0.849), so the privacy
    # breaking point lies lower and had never been approached.
    target_epsilons: list = field(default_factory=lambda: [0.5, 1, 2, 4, 8])
    target_delta: float = 1e-5         # delta << 1 / n_client_samples

    # max_grad_norm = 5.0, and this is the other change that matters.
    # With C=1.0 clipping, not noise, was the limiting factor: at alpha=1 seed 42 the run
    # with epsilon=1000 (negligible noise) stayed at 0.623, while epsilon=32 with C=5
    # reached 0.908. Thirty times less privacy budget and 0.28 more accuracy.
    # C is NOT a privacy parameter: it is the sensitivity bound, and sigma scales with it,
    # so (epsilon, delta) remain exactly those declared.
    max_grad_norm: float = 5.0

    # --- Execution ---
    skip_completed: bool = True        # skip combinations already on disk (resume)
    max_hours: float = 0.0             # > 0: the sweep stops cleanly after N hours and
                                       # resumes on relaunch. The loop is ordered seed ->
                                       # alpha -> epsilon, so it stops leaving whole seeds.

    # GPU: 6 clients on a single card => 1/6 = 0.166 each, so they all run in parallel.
    # num_gpus in Ray is a LOGICAL share, not a memory limit: this is fine because the
    # model has 102k parameters and Opacus per-sample gradients on a batch of 128 take
    # ~50 MB per client, so there is plenty of room (the 9070 XT has 16 GB).
    # CPU only: {"num_cpus": 2, "num_gpus": 0.0}.
    client_resources: dict = field(
        default_factory=lambda: {"num_cpus": 1, "num_gpus": 0.166}
    )


CONFIG = ExperimentConfig()


def validate(cfg: ExperimentConfig = CONFIG) -> None:
    """Guard on the DP accounting.

    The batch must stay well below the TRAINING size of the smallest client
    (min_partition_size minus the internal 80/20 split), otherwise the sample rate
    q = batch / n_train approaches 1, the subsampling amplification vanishes and sigma
    explodes. The 0.5 factor keeps q <= 0.5 even in the worst case.
    """
    min_client_train = int(cfg.min_partition_size * 0.8)
    if cfg.batch_size > 0.5 * min_client_train:
        raise ValueError(
            f"batch_size={cfg.batch_size} troppo grande per min_partition_size="
            f"{cfg.min_partition_size}: il client piu' piccolo avrebbe "
            f"n_train={min_client_train} e q={cfg.batch_size / min_client_train:.2f}. "
            f"Abbassa il batch o alza la soglia."
        )


def num_sampled_clients(cfg: ExperimentConfig = CONFIG) -> int:
    """Clients sampled per round: used ONLY by the FedProx strategy.

    It does not enter the privacy accounting: with record-level DP-SGD the budget depends
    on the sample rate of the BATCHES, not on that of the clients.
    """
    return max(cfg.min_available_clients, round(cfg.fraction_train * cfg.num_clients))


def privacy_settings(cfg: ExperimentConfig = CONFIG) -> list:
    """The privacy configurations of one sweep row: None = baseline without DP."""
    return ([None] if cfg.run_baseline else []) + (
        list(cfg.target_epsilons) if cfg.dp_enabled else []
    )


def set_seeds(seed: int) -> None:
    """Reset the RNGs of the main process.

    Does NOT reseed the Ray actors nor the Opacus noise generator: runs are not
    bit-reproducible, and this must be declared among the limitations in Methodology.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device() -> "torch.device":
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def describe(cfg: ExperimentConfig = CONFIG) -> str:
    n_priv = len(privacy_settings(cfg))
    n_runs = len(cfg.seeds) * len(cfg.alphas) * n_priv
    min_client_train = int(cfg.min_partition_size * 0.8)
    return "\n".join([
        f"{asdict(cfg)}",
        f"client campionati per round (train): {num_sampled_clients(cfg)}",
        f"run pianificate: {len(cfg.seeds)} seed x {len(cfg.alphas)} alpha x {n_priv} "
        f"configurazioni di privacy = {n_runs}",
        f"sample rate DP nel caso peggiore: q = {cfg.batch_size}/{min_client_train} "
        f"= {cfg.batch_size / min_client_train:.3f}",
    ])
