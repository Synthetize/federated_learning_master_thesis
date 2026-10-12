"""Local models: Local_k, one model per (alpha, seed, client).

Each client trains a model on ONLY its own partition's data, with the same epoch budget it
consumes inside the federation, and the model of the best epoch on its own validation set
is evaluated on the common test set. It answers the question "was it worth federating, or
would it have been enough for everyone to train alone?".

Resumable: skips the models already on disk.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from . import paths
from .config import ALPHAS, EPOCHS, NUM_CLIENTS, SEEDS, set_seeds
from .data import class_counts, get_test_loader, local_loaders
from .training import evaluate, train_best_on_val


def local_done(alpha, seed, client) -> bool:
    return os.path.exists(paths.local_path(alpha, seed, client))


def _append_meta(row: dict) -> None:
    pd.DataFrame([row]).to_csv(paths.LOCAL_META_CSV, mode="a", header=not os.path.exists(
        paths.LOCAL_META_CSV), index=False)


def train_one(alpha, seed, client, test_loader, epochs) -> None:
    """Train, evaluate and save one local model."""
    # per-model seed, independent of the launch order: resuming an interrupted run does not
    # change the initialization of the models still to be trained
    set_seeds(seed * 100 + client)
    train_loader, val_loader = local_loaders(alpha, seed, client)
    classes = class_counts(train_loader)
    print(f"  client {client}: {len(train_loader.dataset)} campioni di training, "
          f"classi {classes}")

    t0 = time.time()
    net, val_acc, best_epoch = train_best_on_val(train_loader, val_loader, epochs,
                                                 tag=f"c{client}")
    _, test_acc, _, probs = evaluate(net, test_loader)

    os.makedirs(paths.local_dir(alpha), exist_ok=True)
    np.save(paths.local_path(alpha, seed, client), probs)
    minutes = (time.time() - t0) / 60
    # the meta row is written AFTER the .npy: an interruption between the two leaves a model
    # without a row, never a row without a model
    _append_meta({"alpha": alpha, "seed": seed, "client": client,
                  "n_train": len(train_loader.dataset), "classi": str(classes),
                  "val_acc": val_acc, "best_epoch": best_epoch,
                  "test_acc": test_acc, "durata_min": minutes})
    print(f"      -> test acc {test_acc:.4f} | epoca scelta {best_epoch}/{epochs} "
          f"| {minutes:.1f} min")


def run_local(alphas=None, seeds=None, epochs: int = EPOCHS) -> None:
    alphas = list(ALPHAS if alphas is None else alphas)
    seeds = list(SEEDS if seeds is None else seeds)
    os.makedirs(paths.OUT_DIR, exist_ok=True)

    total = len(alphas) * len(seeds) * NUM_CLIENTS
    todo = sum(not local_done(a, s, c) for a in alphas for s in seeds
               for c in range(NUM_CLIENTS))
    print(f"modelli locali da allenare: {todo} su {total} ({epochs} epoche ciascuno)\n")
    if not todo:
        return

    for alpha in alphas:
        for seed in seeds:
            if all(local_done(alpha, seed, c) for c in range(NUM_CLIENTS)):
                print(f"[alpha={alpha:g} seed={seed}] gia' completo, salto")
                continue
            print(f"\n=== alpha={alpha:g} seed={seed}")
            test_loader = get_test_loader(alpha, seed)
            if not os.path.exists(paths.labels_path(seed)):
                np.save(paths.labels_path(seed),
                        test_loader.dataset.labels.numpy().astype(np.int8))
            for client in range(NUM_CLIENTS):
                if not local_done(alpha, seed, client):
                    train_one(alpha, seed, client, test_loader, epochs)
