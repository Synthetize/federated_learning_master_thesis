"""Accounting DP: sigma per client, anteprima del rapporto rumore/segnale."""

from __future__ import annotations

import math

from opacus.accountants.utils import get_noise_multiplier

from .config import CONFIG, ExperimentConfig
from .dataset import get_fds
from .model import num_parameters


def client_train_sizes(alpha: float, seed: int,
                       cfg: ExperimentConfig = CONFIG) -> list[int]:
    """Campioni di TRAINING per ciascun client (dopo lo split locale 80/20)."""
    fds_ = get_fds(alpha, seed, cfg)
    return [int(len(fds_.load_partition(pid)) * 0.8) for pid in range(cfg.num_clients)]


def dp_plan(n_train: int, target_epsilon: float, cfg: ExperimentConfig = CONFIG):
    """(sample_rate, steps, sigma) per UN client con `n_train` campioni di training.

    Stesso schema di accounting che Opacus applichera' internamente in
    `make_private_with_epsilon`: unita' di privacy = singolo campione (record-level),
    Sampled Gaussian Mechanism con campionamento di Poisson sui batch.
    """
    sample_rate = cfg.batch_size / n_train
    steps = cfg.num_rounds * cfg.local_epochs * math.ceil(n_train / cfg.batch_size)
    sigma = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=cfg.target_delta,
        sample_rate=sample_rate,
        steps=steps,
        accountant="rdp",
    )
    return sample_rate, steps, sigma


def noise_to_signal(sigma: float, cfg: ExperimentConfig = CONFIG) -> float:
    """Rapporto rumore/segnale per step, come riportato in Implementation."""
    return sigma * (num_parameters() ** 0.5) / cfg.batch_size


def dp_preview(alpha: float, seed: int, cfg: ExperimentConfig = CONFIG) -> None:
    sizes = client_train_sizes(alpha, seed, cfg)
    n_params = num_parameters()
    print(f"alpha={alpha}, seed={seed} | parametri del modello: {n_params:,}")
    print(f"campioni di training per client: {sizes}")
    print(f"delta = {cfg.target_delta}, max_grad_norm (C) = {cfg.max_grad_norm}, "
          f"batch = {cfg.batch_size}, local_epochs = {cfg.local_epochs}\n")
    if min(sizes) <= cfg.batch_size:
        print(f"!! ATTENZIONE: il client piu' piccolo ha {min(sizes)} campioni, <= "
              f"batch_size ({cfg.batch_size}): sample_rate >= 1, l'accounting non e' "
              f"valido. Alza min_partition_size o abbassa batch_size.\n")
    for eps in cfg.target_epsilons:
        print(f"=== target epsilon = {eps}  (per client, sull'intero training) ===")
        for pid, n in enumerate(sizes):
            q, steps, sigma = dp_plan(n, eps, cfg)
            print(f"  client {pid}: n={n:6d}  q={q:.5f}  steps={steps:6d}  "
                  f"sigma={sigma:7.3f}  rumore/segnale per step ~ "
                  f"{noise_to_signal(sigma, cfg):7.1f}x")
        print()


def sigma_table(cfg: ExperimentConfig = CONFIG):
    """Tabella sigma per (alpha, epsilon), mediata sui client del primo seed.

    E' la tabella che in Implementation sta ancora come placeholder: e' l'unica prova che
    l'epsilon dichiarato corrisponda a una scala di rumore plausibile.
    """
    import pandas as pd

    rows = []
    seed = cfg.seeds[0]
    for alpha in cfg.alphas:
        sizes = client_train_sizes(alpha, seed, cfg)
        for eps in cfg.target_epsilons:
            plans = [dp_plan(n, eps, cfg) for n in sizes]
            sigmas = [p[2] for p in plans]
            rows.append({
                "alpha": alpha,
                "epsilon": eps,
                "n_train_min": min(sizes),
                "n_train_max": max(sizes),
                "q_min": min(p[0] for p in plans),
                "q_max": max(p[0] for p in plans),
                "sigma_min": min(sigmas),
                "sigma_max": max(sigmas),
                "sigma_mean": sum(sigmas) / len(sigmas),
                "noise_signal_mean": sum(noise_to_signal(s, cfg) for s in sigmas) / len(sigmas),
            })
    return pd.DataFrame(rows)
