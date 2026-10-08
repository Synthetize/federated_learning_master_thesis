"""Percorsi di output dei baseline.

E' esattamente cio' che dca.py legge, quindi nomi e struttura non si cambiano senza cambiare
anche dca.py (`_local_files`, `baseline_nb`):

    results/baselines/alpha_<alpha>/probs_local_s<seed>_c<client>.npy   P(classe 1), float32
    results/baselines/probs_central_s<seed>.npy                         P(classe 1), float32
    results/baselines/central_weights_s<seed>.pt                        pesi del centralizzato
    results/baselines/test_labels_s<seed>.npy                           label grezze 0/1
    results/baselines/local_meta.csv                                    una riga per modello locale
    results/baselines/central_meta.csv                                  una riga per seme

I locali stanno in una cartella per alpha; centralizzato e label restano in cima perche'
non dipendono dall'alpha. Le label dipendono solo dal seed (lo split del test set).

CONVENZIONE SUI NOMI: f"alpha_{alpha:g}", la stessa di federated/paths.py (10.0 ->
"alpha_10", 1.0 -> "alpha_1"). Con due convenzioni dca.py vedeva due cartelle per lo stesso
alpha e le faceva collassare sulla stessa chiave, sovrascrivendole.
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
    """In cima e non dentro alpha_*: le label del test set dipendono solo dal seed."""
    return os.path.join(OUT_DIR, f"test_labels_s{seed}.npy")
