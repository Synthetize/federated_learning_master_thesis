"""Modello centralizzato: UNO PER SEME.

Perche' per seme e non uno solo. Il modello centralizzato non dipende dalla partizione
Dirichlet: unendo i dati dei client si ricostruisce lo stesso pool qualunque sia l'alpha, e
per questo si allena fuori dal ciclo sugli alpha. Dipende invece dal SEED, perche' il seed
fissa anche lo split 80/20 del test set. Il test set del seed 43 e' un campione di immagini
diverso da quello del seed 42, e un modello allenato sul pool del seed 42 ci ha gia' visto
dentro gran parte delle immagini: le probabilita' salvate non corrispondono alle immagini
giuste e l'accuracy cade a livello del caso (0.4964 contro le label del seed 43, 0.9641
contro le proprie).

Conseguenza per la tesi: Utility Cost e Federation Cost sono gli unici due termini della
decomposizione che toccano il modello centralizzato. Heterogeneity, Privacy e Interaction
sono differenze FL-meno-FL dentro uno stesso draw (il termine centralizzato si semplifica
algebricamente nell'Interaction), e la Safe Zone non lo usa mai. Con un modello per seme il
Federation Cost e' calcolato dentro lo stesso draw e acquista anche una barra d'errore.

Riprendibile: salta i semi gia' su disco.
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
    np.save(paths.central_path(seed), probs)       # per ultimo: e' il marcatore di "fatto"
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
