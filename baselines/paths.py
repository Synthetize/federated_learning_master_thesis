"""Baselines output paths.

This is exactly what dca.py reads, so names and structure must not change without also
changing dca.py (`_local_files`, `baseline_nb`):

    results/baselines/alpha_<alpha>/probs_local_s<seed>_c<client>.npy   P(class 1), float32
    results/baselines/probs_central_s<seed>.npy                         P(class 1), float32
    results/baselines/central_weights_s<seed>.pt                        centralized weights
    results/baselines/test_labels_s<seed>.npy                           raw 0/1 labels
    results/baselines/local_meta.csv                                    one row per local model
    results/baselines/central_meta.csv                                  one row per seed

The local models live in one folder per alpha; the centralized model and the labels stay at
the top level because they do not depend on alpha. The labels depend only on the seed (the
test-set split).

NAMING CONVENTION: f"alpha_{alpha:g}", the same as federated/paths.py (10.0 -> "alpha_10",
1.0 -> "alpha_1"). With two conventions dca.py saw two folders for the same alpha and
collapsed them onto the same key, overwriting each other.
"""

from __future__ import annotations

import os

from federated.paths import BASELINES_DIR, alpha_tag

OUT_DIR = BASELINES_DIR
LOCAL_META_CSV = os.path.join(OUT_DIR, "local_meta.csv")
CENTRAL_META_CSV = os.path.join(OUT_DIR, "central_meta.csv")


def local_dir(alpha: float) -> str:
    return os.path.join(OUT_DIR, alpha_tag(alpha))


def local_path(alpha: float, seed: int, client: int) -> str:
    return os.path.join(local_dir(alpha), f"probs_local_s{seed}_c{client}.npy")


def central_path(seed: int) -> str:
    return os.path.join(OUT_DIR, f"probs_central_s{seed}.npy")


def central_weights_path(seed: int) -> str:
    return os.path.join(OUT_DIR, f"central_weights_s{seed}.pt")


def labels_path(seed: int) -> str:
    """At the top level and not inside alpha_*: the test-set labels depend only on the seed."""
    return os.path.join(OUT_DIR, f"test_labels_s{seed}.npy")
