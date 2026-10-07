"""Centralised reference, one run per seed.

Why this exists. The centralised model is invariant to the Dirichlet partition: pooling the
clients' data back together reconstructs the same training pool whatever alpha is, which is
why Chapter 3 trains it outside the alpha loop. It is NOT invariant to the SEED, because in
baselines.py the seed also fixes the 80/20 train/test split:

    sp = full.train_test_split(test_size=0.2, seed=seed, stratify_by_column="label")

so seed 43's test set is a different sample of images from seed 42's. A model trained on
seed 42's pool therefore cannot be scored on seed 43's test set: most of that test set was
in its training data, and in practice the stored probabilities line up with the wrong images
and score at chance (0.4964 against labels s43, against 0.9641 on its own s42).

Consequence for the thesis: the Utility Cost and the Federation Cost are the only two terms
of the decomposition that touch the centralised model, and with a single run they are
within-draw only for seed 42. Heterogeneity, Privacy and the Interaction are FL-minus-FL
differences taken inside one draw, and the centralised term cancels out of the Interaction
algebraically, so those three are unaffected either way. So is the whole Safe Zone, which
never uses the centralised model.

One run per seed fixes this and, incidentally, gives the Federation Cost the error bar it
currently does not have.

Writes results/baselines/probs_central_s<seed>.npy and central_weights_s<seed>.pt.
The existing probs_central.npy is left untouched; it is the seed-42 run and is reused.

    python central_seeds.py          # every missing seed
    python central_seeds.py 43 44    # only these
"""
import os
import sys
import time

import numpy as np
import torch

from baselines import (BATCH_SIZE, EPOCHS, OUT_DIR, SEEDS, evaluate, get_fds,
                       labels_path, loaders_from, set_seeds, train_best_on_val)

# Irrelevant to the result, since only the pooled train split is used. The mildest value is
# the one the partitioner's minimum-size retry loop accepts on the first attempt.
PARTITION_ALPHA = 10.0


def central_path(seed):
    return os.path.join(OUT_DIR, f"probs_central_s{seed}.npy")


def run_one(seed):
    if os.path.exists(central_path(seed)):
        print(f"seed {seed}: already on disk, skipping")
        return

    print(f"\n=== centralised, seed {seed}")
    fds = get_fds(PARTITION_ALPHA, seed)
    pool = fds.load_split("train")          # the test set is already excluded from this
    sp = pool.train_test_split(test_size=0.2, seed=seed)

    set_seeds(seed)
    train_loader, y_train = loaders_from(sp["train"], BATCH_SIZE, True)
    val_loader, _ = loaders_from(sp["test"], 256, False)
    test_loader, y_test = loaders_from(fds.load_split("test"), 256, False)

    if not os.path.exists(labels_path(seed)):
        np.save(labels_path(seed), y_test.numpy().astype(np.int8))

    print(f"  training on {len(y_train)} samples, validation on {len(val_loader.dataset)}")
    t0 = time.time()
    net, val_acc, best_ep = train_best_on_val(train_loader, val_loader, EPOCHS,
                                              tag=f"central_s{seed}", log_every=10)
    _, test_acc, _, p1 = evaluate(net, test_loader)

    np.save(central_path(seed), p1)
    torch.save(net.state_dict(), os.path.join(OUT_DIR, f"central_weights_s{seed}.pt"))
    print(f"  -> test acc {test_acc:.4f} | best epoch {best_ep}/{EPOCHS} "
          f"| val acc {val_acc:.4f} | {(time.time() - t0) / 60:.1f} min")


def main():
    seeds = [int(a) for a in sys.argv[1:]] or list(SEEDS)

    # Seed 42 is already trained and stored under the old name. Reuse it: retraining it
    # would only add a different random initialisation to a run the thesis already reports.
    legacy = os.path.join(OUT_DIR, "probs_central.npy")
    if 42 in seeds and not os.path.exists(central_path(42)) and os.path.exists(legacy):
        np.save(central_path(42), np.load(legacy))
        print("seed 42: reused the existing probs_central.npy")
        seeds = [s for s in seeds if s != 42]

    for s in seeds:
        run_one(s)

    print("\ndone. Re-run dca.py afterwards: it picks up probs_central_s*.npy automatically "
          "and falls back to probs_central.npy for any seed that is still missing.")


if __name__ == "__main__":
    main()
