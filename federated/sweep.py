"""Sweep federato: una run per (seed, alpha, epsilon), con ripresa.

    python -m federated.sweep                      # lo sweep completo
    python -m federated.sweep --preflight-only     # solo la verifica delle partizioni
    python -m federated.sweep --dp-preview         # sigma e rumore/segnale, niente training
    python -m federated.sweep --alphas 10 --seeds 42 --epsilons inf    # run di controllo
    python -m federated.sweep --max-hours 8        # si ferma pulito dopo 8 ore
    python -m federated.sweep --replicates 4 --alphas 0.3 --seeds 42 --epsilons 0.5 8

Il ciclo e' ordinato seed -> alpha -> epsilon, quindi un'interruzione lascia seed interi
invece che tutti a meta'. Con --skip-completed (default) si riprende da dove si era fermato.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import pandas as pd
import torch
from flwr.app import Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedProx
from flwr.simulation import run_simulation

from . import paths
from .app import (CURRENT_RUN_METRICS, build_train_config, client_app,
                  initial_arrays, make_global_evaluate)
from .config import (CONFIG, ExperimentConfig, describe, privacy_settings, set_seeds,
                     validate)
from .dataset import preflight_partitions
from .privacy import client_train_sizes, dp_plan, dp_preview, noise_to_signal

SUMMARY_LAST_N = 10   # su quanti round finali calcolare media e deviazione standard


# =================================================================== una run
def run_training(alpha: float, seed: int, target_epsilon: float | None,
                 cfg: ExperimentConfig = CONFIG):
    """Una run federata completa. target_epsilon=None => baseline senza DP.
    Ritorna (metriche per round, pesi finali del modello globale)."""
    set_seeds(seed)

    if target_epsilon is not None:
        sizes = client_train_sizes(alpha, seed, cfg)
        for pid, n in enumerate(sizes):
            q, steps, sigma = dp_plan(n, target_epsilon, cfg)
            print(f"    client {pid}: n={n} q={q:.5f} steps={steps} "
                  f"sigma={sigma:.3f} rumore/segnale~{noise_to_signal(sigma, cfg):.1f}x")

    server_app = ServerApp()
    holder = {}

    @server_app.main()
    def main(grid: Grid, context: Context) -> None:
        strategy = FedProx(
            proximal_mu=cfg.fedprox_mu,
            fraction_train=cfg.fraction_train,
            fraction_evaluate=cfg.fraction_evaluate,
            min_train_nodes=cfg.min_available_clients,
            min_evaluate_nodes=cfg.min_available_clients,
            min_available_nodes=cfg.min_available_clients,
        )
        # alpha/seed servono al client sia in training sia in evaluation, e i due
        # ConfigRecord sono distinti: vanno messi in entrambi.
        from flwr.app import ConfigRecord
        eval_config = ConfigRecord({"alpha": float(alpha), "seed": int(seed)})
        start_kwargs = dict(
            grid=grid,
            initial_arrays=initial_arrays(),
            num_rounds=cfg.num_rounds,
            train_config=build_train_config(alpha, seed, target_epsilon),
            evaluate_fn=make_global_evaluate(alpha, seed, target_epsilon),
        )
        try:
            res = strategy.start(**start_kwargs, evaluate_config=eval_config)
        except TypeError:
            # versioni di Flower senza `evaluate_config`: il client ricade sulla
            # combinazione memorizzata durante il training (vedi _LAST_RUN_KEY)
            print("    nota: questa versione di Flower non accetta evaluate_config, "
                  "il client usa il fallback")
            res = strategy.start(**start_kwargs)
        holder["result"] = res

    CURRENT_RUN_METRICS.clear()
    error = None
    try:
        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=cfg.num_clients,
            backend_config={
                "client_resources": cfg.client_resources,
                # Ray does not detect AMD GPUs under WSL (no amdsmi/rocm-smi), so the
                # card must be declared by hand or the actor pool stays empty.
                "init_args": {"num_gpus": 1 if cfg.client_resources.get("num_gpus") else 0},
            },
        )
    except Exception as exc:  # noqa: BLE001 - si prosegue con la combinazione successiva
        error = exc
        print(f"    !! run interrotta: {type(exc).__name__}: {exc}")
        print(f"       round completati: {max(len(CURRENT_RUN_METRICS) - 1, 0)}")

    if error is None and "result" not in holder:
        error = RuntimeError("run_simulation terminata senza produrre un risultato")
        print(f"    !! {error}")

    epsilon_value = target_epsilon if target_epsilon is not None else float("inf")

    if error is None:
        res = holder["result"]
        rounds_ = sorted(res.evaluate_metrics_serverapp.keys())

        def col(name, default=float("nan")):
            return [res.evaluate_metrics_serverapp[r].get(name, default) for r in rounds_]

        round_df = pd.DataFrame({
            "round": rounds_,
            "accuracy": col("accuracy"),
            "loss": col("loss"),
            "val_accuracy": col("val_accuracy"),
            "epsilon": epsilon_value,
            "alpha": alpha,
            "seed": seed,
            "completed": True,
        })
        return round_df, res.arrays

    round_df = pd.DataFrame(CURRENT_RUN_METRICS)
    if not round_df.empty:
        round_df["epsilon"] = epsilon_value
        round_df["alpha"] = alpha
        round_df["seed"] = seed
        round_df["completed"] = False
    return round_df, None


# =================================================================== salvataggio
def summarize_results(history: pd.DataFrame, last_n: int = SUMMARY_LAST_N) -> pd.DataFrame:
    """Una riga per (alpha, seed, epsilon).

    Oltre ai valori dell'ultimo round riporta media/std dell'accuracy sugli ultimi
    `last_n` round e il massimo raggiunto. Motivo: con epsilon nella zona di transizione il
    modello globale non converge ma oscilla (il rumore DP viene rigenerato a ogni step,
    quindi i pesi fanno una specie di random walk) e l'accuracy del solo ultimo round e'
    poco affidabile. La media sugli ultimi round e' molto piu' stabile, la std quantifica
    l'instabilita'.

    `val_round_best` e' il round con la migliore accuracy sul validation pooled: serve per
    la lettura simmetrica all'early stopping dei baseline locali, e NON e' usato come
    selezione di default.
    """
    group_cols = [c for c in ["alpha", "seed", "epsilon"] if c in history.columns]
    rows = []
    for keys, group in history.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        group = group.sort_values("round")
        tail = group["accuracy"].tail(last_n)
        last = group.iloc[-1]
        best = group.loc[group["accuracy"].idxmax()]
        row = dict(zip(group_cols, keys))
        row.update({
            "round": int(last["round"]),
            "accuracy": last["accuracy"],
            "loss": last["loss"],
            f"acc_media_ultimi{last_n}": tail.mean(),
            f"acc_std_ultimi{last_n}": tail.std(),
            "acc_max": best["accuracy"],
            "acc_max_round": int(best["round"]),
            "completed": (bool(group["completed"].iloc[-1])
                          if "completed" in group.columns else True),
        })
        if "val_accuracy" in group.columns and group["val_accuracy"].notna().any():
            vbest = group.loc[group["val_accuracy"].idxmax()]
            row["val_acc_max"] = float(vbest["val_accuracy"])
            row["val_round_best"] = int(vbest["round"])
            row["acc_at_val_best"] = float(vbest["accuracy"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def save_config_snapshot(alpha: float, seed: int, cfg: ExperimentConfig = CONFIG) -> None:
    out_dir = paths.results_dir(alpha, seed)
    os.makedirs(out_dir, exist_ok=True)
    snapshot = asdict(cfg)
    snapshot["_run"] = {"alpha": alpha, "seed": seed,
                        "replicate": paths.get_replicate()}
    with open(paths.config_snapshot_path(alpha, seed), "w") as f:
        json.dump(snapshot, f, indent=2)


def save_run(alpha: float, seed: int, epsilon, arrays, round_df: pd.DataFrame) -> None:
    """Pesi + storico della run appena conclusa, unendo a quanto gia' su disco."""
    out_dir = paths.results_dir(alpha, seed)
    os.makedirs(out_dir, exist_ok=True)

    if arrays is not None:
        mp = paths.model_path(alpha, seed, epsilon)
        torch.save(arrays.to_torch_state_dict(), mp)
        print(f"    salvato: {mp}")
    else:
        print("    nessun modello finale (run interrotta)")

    if round_df.empty:
        return
    path = paths.history_path(alpha, seed)
    if os.path.exists(path):
        previous = pd.read_csv(path)
        previous = previous[previous["epsilon"] != round_df["epsilon"].iloc[0]]
        history = pd.concat([previous, round_df], ignore_index=True)
    else:
        history = round_df
    history.to_csv(path, index=False)
    summarize_results(history).to_csv(paths.final_results_path(alpha, seed), index=False)


def already_done(alpha: float, seed: int, epsilon) -> bool:
    """True se questa combinazione e' gia' su disco e completata (per la ripresa)."""
    path = paths.history_path(alpha, seed)
    if not os.path.exists(path):
        return False
    df = pd.read_csv(path)
    value = float("inf") if epsilon is None else float(epsilon)
    sub = df[df["epsilon"] == value]
    if sub.empty:
        return False
    return bool(sub["completed"].iloc[-1]) if "completed" in sub.columns else True


# =================================================================== il ciclo
def run_sweep(cfg: ExperimentConfig = CONFIG) -> None:
    jobs = [(s, a, e) for s in cfg.seeds for a in cfg.alphas
            for e in privacy_settings(cfg)]
    print(f"run totali pianificate: {len(jobs)}")
    if cfg.max_hours > 0:
        print(f"budget di tempo: {cfg.max_hours} h (poi si ferma in modo pulito)")
    print()

    t0 = time.time()
    durations = []
    stopped_early = False
    for i, (seed, alpha, eps) in enumerate(jobs, start=1):
        if cfg.max_hours > 0 and (time.time() - t0) / 3600 >= cfg.max_hours:
            print(f"\nbudget di {cfg.max_hours} h esaurito: mi fermo a "
                  f"{i - 1}/{len(jobs)} run.")
            print("rilancia lo stesso comando per riprendere da dove si e' fermato.")
            stopped_early = True
            break

        label = (f"seed={seed} alpha={alpha} "
                 f"epsilon={eps if eps is not None else 'inf (no DP)'}")
        if cfg.skip_completed and already_done(alpha, seed, eps):
            print(f"[{i}/{len(jobs)}] {label} -- gia' fatto, salto")
            continue

        print(f"\n[{i}/{len(jobs)}] {label}")
        run_t0 = time.time()
        try:
            save_config_snapshot(alpha, seed, cfg)
            round_df, arrays = run_training(alpha, seed, eps, cfg)
            save_run(alpha, seed, eps, arrays, round_df)
        except Exception as exc:  # noqa: BLE001 - una combinazione rotta non ferma lo sweep
            print(f"    !! combinazione fallita: {type(exc).__name__}: {exc}")
            print("       si prosegue con la successiva")

        durations.append(time.time() - run_t0)
        mean = sum(durations) / len(durations)
        left = len(jobs) - i
        print(f"    durata {durations[-1] / 60:.1f} min | media {mean / 60:.1f} min | "
              f"ETA ~{left * mean / 3600:.1f} h ({left} run rimanenti)")

    if not stopped_early:
        print("\nsweep terminato: tutte le combinazioni pianificate sono state percorse.")


def run_replicates(n: int, cfg: ExperimentConfig = CONFIG) -> None:
    """Ripetizioni a partizione fissa sulle celle indicate da --alphas/--seeds/--epsilons.

    Separa la varianza del draw Dirichlet da quella di DP-SGD: la stessa (alpha, seed,
    epsilon) rilanciata da' la stessa partizione e lo stesso test set, con rumore diverso.
    Le ripetizioni finiscono in cartelle con suffisso _rep<N> e non toccano le principali.
    """
    cells = [(s, a, e) for s in cfg.seeds for a in cfg.alphas
             for e in privacy_settings(cfg)]
    print(f"{len(cells)} celle x {n} ripetizioni = {len(cells) * n} run\n")
    try:
        for rep in range(1, n + 1):
            paths.set_replicate(rep)
            for seed, alpha, eps in cells:
                if cfg.skip_completed and already_done(alpha, seed, eps):
                    print(f"rep {rep} seed={seed} alpha={alpha} eps={eps} -- gia' fatto")
                    continue
                print(f"\nrep {rep}/{n} seed={seed} alpha={alpha} eps={eps}")
                t0 = time.time()
                try:
                    save_config_snapshot(alpha, seed, cfg)
                    round_df, arrays = run_training(alpha, seed, eps, cfg)
                    save_run(alpha, seed, eps, arrays, round_df)
                except Exception as exc:  # noqa: BLE001
                    print(f"    !! fallita: {type(exc).__name__}: {exc}")
                print(f"    durata {(time.time() - t0) / 60:.1f} min -> "
                      f"{paths.results_dir(alpha, seed)}")
    finally:
        paths.set_replicate(0)
    print("\nripetizioni terminate")


# =================================================================== CLI
def _parse_epsilons(values):
    """'inf' -> None (baseline senza DP), il resto float."""
    out = []
    for v in values:
        out.append(None if str(v).lower() in ("inf", "none", "nodp") else float(v))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sweep federato: eterogeneita' (alpha) x rumore di privacy (epsilon).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--alphas", nargs="+", type=float, default=None,
                   help="livelli di eterogeneita' (default: quelli di config.py)")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="seed dei draw Dirichlet (default: quelli di config.py)")
    p.add_argument("--epsilons", nargs="+", default=None,
                   help="budget di privacy; 'inf' per la run senza DP")
    p.add_argument("--rounds", type=int, default=None, help="override di num_rounds")
    p.add_argument("--max-hours", type=float, default=None,
                   help="si ferma in modo pulito dopo N ore (0 = nessun limite)")
    p.add_argument("--no-skip-completed", action="store_true",
                   help="riesegui anche le combinazioni gia' su disco")
    p.add_argument("--cpu", action="store_true",
                   help="forza l'esecuzione su CPU (num_gpus = 0)")
    p.add_argument("--replicates", type=int, default=0, metavar="N",
                   help="ripetizioni a partizione fissa sulle celle selezionate")
    p.add_argument("--preflight-only", action="store_true",
                   help="verifica le partizioni di tutte le combinazioni e esce")
    p.add_argument("--dp-preview", action="store_true",
                   help="stampa sigma e rumore/segnale per la prima combinazione e esce")
    p.add_argument("--sigma-table", action="store_true",
                   help="stampa la tabella sigma per (alpha, epsilon) e esce")
    return p


def apply_args(args, cfg: ExperimentConfig = CONFIG) -> ExperimentConfig:
    if args.alphas is not None:
        cfg.alphas = list(args.alphas)
    if args.seeds is not None:
        cfg.seeds = list(args.seeds)
    if args.epsilons is not None:
        eps = _parse_epsilons(args.epsilons)
        cfg.run_baseline = any(e is None for e in eps)
        cfg.target_epsilons = [e for e in eps if e is not None]
        cfg.dp_enabled = bool(cfg.target_epsilons)
    if args.rounds is not None:
        cfg.num_rounds = args.rounds
    if args.max_hours is not None:
        cfg.max_hours = args.max_hours
    if args.no_skip_completed:
        cfg.skip_completed = False
    if args.cpu:
        cfg.client_resources = {"num_cpus": 2, "num_gpus": 0.0}
    return cfg


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = apply_args(args)
    validate(cfg)

    print(f"dispositivo: {'cuda' if torch.cuda.is_available() else 'cpu'} | "
          f"risultati in: {paths.FL_DIR}")
    print(describe(cfg))
    print()

    if args.sigma_table:
        from .privacy import sigma_table
        print(sigma_table(cfg).to_string(index=False))
        return 0

    if args.dp_preview:
        dp_preview(cfg.alphas[0], cfg.seeds[0], cfg)
        return 0

    print("Preflight partizioni (tutte le combinazioni alpha x seed):")
    preflight_partitions(cfg)
    print()
    if args.preflight_only:
        return 0

    if args.replicates > 0:
        run_replicates(args.replicates, cfg)
    else:
        run_sweep(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
