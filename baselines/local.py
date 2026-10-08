"""Modelli locali: Local_k, un modello per (alpha, seed, client).

Ogni client allena un modello sui SOLI dati della propria partizione, con lo stesso budget
di epoche che consuma dentro la federazione, e il modello dell'epoca migliore sulla propria
validation viene valutato sul test set comune. E' la risposta alla domanda "e' valsa la
pena federarsi, o bastava che ognuno si allenasse da solo?".

Riprendibile: salta i modelli gia' su disco.
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
    """Allena, valuta e salva un modello locale."""
    # seme per modello, indipendente dall'ordine in cui si lanciano: riprendere una run
    # interrotta non cambia l'inizializzazione dei modelli ancora da fare
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
    # la riga di meta si scrive DOPO il .npy: un'interruzione fra i due lascia un modello
    # senza riga, mai una riga senza modello
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
