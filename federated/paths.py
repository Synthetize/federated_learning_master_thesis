"""Tutti i percorsi di output in un posto solo.

Nel notebook erano sparsi fra tre celle e relativi alla working directory. Qui sono
ancorati a RESULTS_ROOT (assoluto), cosi' lo sweep si puo' lanciare da qualunque cartella
e gli attori Ray scrivono dove ci si aspetta.

CONVENZIONE SUI NOMI: f"alpha_{alpha:g}", quindi 10.0 -> "alpha_10" e 1.0 -> "alpha_1".
Una sola convenzione per lo sweep federato e per i baseline. dca.py rilegge l'alpha con
float() sul nome, quindi la accetta; averne due, come nel notebook, creava cartelle
distinte che collassavano sulla stessa chiave in _local_files() sovrascrivendosi.
"""

from __future__ import annotations

import os

import numpy as np

from .config import RESULTS_ROOT

FL_DIR = os.path.join(RESULTS_ROOT, "malaria")
BASELINES_DIR = os.path.join(RESULTS_ROOT, "baselines")

# ---------------------------------------------------------------------------------
# Ripetizioni a partizione fissa
# ---------------------------------------------------------------------------------
# set_seeds non risemina gli attori Ray ne' il generatore di rumore di Opacus, quindi
# rilanciare la stessa (alpha, seed, epsilon) da' la STESSA partizione Dirichlet e lo
# STESSO test set, con rumore DP diverso: e' esattamente la ripetizione a partizione fissa
# che serve per separare la varianza del draw da quella di DP-SGD. Serve solo non
# sovrascrivere i risultati, ed e' quello che fa questo indice.
# 0 = run principale, cartelle senza suffisso.
_REPLICATE = 0


def set_replicate(n: int) -> None:
    global _REPLICATE
    _REPLICATE = int(n)


def get_replicate() -> int:
    return _REPLICATE


def alpha_tag(alpha: float) -> str:
    return f"alpha_{float(alpha):g}"


def results_dir(alpha: float, seed: int) -> str:
    base = os.path.join(FL_DIR, f"{alpha_tag(alpha)}_seed{int(seed)}")
    return base if _REPLICATE == 0 else f"{base}_rep{_REPLICATE}"


def _eps_tag(epsilon) -> str:
    return "inf" if not np.isfinite(epsilon) else f"{epsilon:g}"


def probs_path(alpha: float, seed: int, epsilon) -> str:
    """Un file binario per configurazione di privacy: float32 accodati, una riga per round.

    Formato grezzo e non CSV perche' sono ~5.500 valori per round: in binario sono 22 KB,
    in CSV sarebbero dieci volte tanto. Si rileggono con
        np.fromfile(path, dtype=np.float32).reshape(-1, n_test)
    e le righe incomplete di una run interrotta si riconoscono dal resto della divisione.
    """
    return os.path.join(results_dir(alpha, seed), f"probs_eps{_eps_tag(epsilon)}.f32")


def labels_path(alpha: float, seed: int) -> str:
    """Le label del test set dipendono solo dal seed, quindi si salvano una volta sola."""
    return os.path.join(results_dir(alpha, seed), "test_labels.npy")


def val_probs_path(alpha: float, seed: int, epsilon) -> str:
    """Come probs_path, ma sul validation pooled. Stesso formato, rileggibile con
    np.fromfile(path, dtype=np.float32).reshape(-1, n_val).
    """
    return os.path.join(results_dir(alpha, seed), f"val_probs_eps{_eps_tag(epsilon)}.f32")


def val_labels_path(alpha: float, seed: int) -> str:
    return os.path.join(results_dir(alpha, seed), "val_labels.npy")


def model_path(alpha: float, seed: int, epsilon) -> str:
    label = "no_dp" if epsilon is None or not np.isfinite(epsilon) else f"epsilon_{epsilon}"
    return os.path.join(results_dir(alpha, seed), f"{label}.pt")


def history_path(alpha: float, seed: int) -> str:
    return os.path.join(results_dir(alpha, seed), "training_history.csv")


def final_results_path(alpha: float, seed: int) -> str:
    return os.path.join(results_dir(alpha, seed), "final_results.csv")


def config_snapshot_path(alpha: float, seed: int) -> str:
    return os.path.join(results_dir(alpha, seed), "config.json")
