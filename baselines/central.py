"""Centralized model: ONE PER SEED.

Why per seed and not a single one. The centralized model does not depend on the Dirichlet
partition: pooling the clients' data rebuilds the same pool whatever the alpha, which is
why it is trained outside the loop over alphas. It does depend on the SEED, because the
seed also fixes the 80/20 split of the test set. The test set of seed 43 is a different
sample of images from that of seed 42, and a model trained on the seed-42 pool has already
seen most of those images: the saved probabilities do not match the right images and the
accuracy drops to chance level (0.4964 against the seed-43 labels, 0.9641 against its own).

Consequence for the thesis: Utility Cost and Federation Cost are the only two terms of the
decomposition that touch the centralized model. Heterogeneity, Privacy and Interaction are
FL-minus-FL differences within the same draw (the centralized term cancels algebraically
in the Interaction), and the Safe Zone never uses it. With one model per seed the
Federation Cost is computed within the same draw and also gets an error bar.

Resumable: skips the seeds already on disk.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch

from . import paths
from .config import EPOCHS, SEEDS, set_seeds
from .data import central_loaders
from .training import evaluate, train_best_on_val


def central_done(seed) -> bool:
    return os.path.exists(paths.central_path(seed))


def run_one(seed: int, epochs: int = EPOCHS) -> None:
    if central_done(seed):
        print(f"seed {seed}: gia' su disco, salto")
        return

    print(f"\n=== centralizzato, seed {seed}")
    set_seeds(seed)
    train_loader, val_loader, test_loader = central_loaders(seed)
    if not os.path.exists(paths.labels_path(seed)):
        np.save(paths.labels_path(seed),
                test_loader.dataset.labels.numpy().astype(np.int8))
    print(f"  training su {len(train_loader.dataset)} campioni, "
          f"validation su {len(val_loader.dataset)}")

    t0 = time.time()
    net, val_acc, best_epoch = train_best_on_val(train_loader, val_loader, epochs,
                                                 tag=f"central_s{seed}", log_every=10)
    _, test_acc, _, probs = evaluate(net, test_loader)

    os.makedirs(paths.OUT_DIR, exist_ok=True)
    torch.save(net.state_dict(), paths.central_weights_path(seed))
    np.save(paths.central_path(seed), probs)       # written last: it is the "done" marker
    minutes = (time.time() - t0) / 60
    pd.DataFrame([{"seed": seed, "n_train": len(train_loader.dataset),
                   "val_acc": val_acc, "best_epoch": best_epoch, "test_acc": test_acc,
                   "durata_min": minutes}]).to_csv(
        paths.CENTRAL_META_CSV, mode="a",
        header=not os.path.exists(paths.CENTRAL_META_CSV), index=False)
    print(f"  -> test acc {test_acc:.4f} | epoca scelta {best_epoch}/{epochs} "
          f"| val acc {val_acc:.4f} | {minutes:.1f} min")


def run_central(seeds=None, epochs: int = EPOCHS) -> None:
    seeds = list(SEEDS if seeds is None else seeds)
    todo = [s for s in seeds if not central_done(s)]
    print(f"modelli centralizzati da allenare: {len(todo)} su {len(seeds)} "
          f"({epochs} epoche ciascuno)")
    for seed in seeds:
        run_one(seed, epochs)
