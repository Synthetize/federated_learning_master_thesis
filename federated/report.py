"""Riepilogo dello sweep: tabelle aggregate e due figure diagnostiche.

    python -m federated.report            # tabelle + CSV
    python -m federated.report --plots    # anche le figure

Rilegge tutto dal disco, quindi funziona senza rifare il training. Non calcola Net
Benefit: quello e' dca.py. Qui c'e' solo l'accuracy, che serve a vedere se lo sweep e'
andato a buon fine e a produrre la tabella accuracy-vs-Safe-Range del Capitolo 5.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

from . import paths
from .config import CONFIG
from .sweep import SUMMARY_LAST_N, summarize_results

ACC_COL = f"acc_media_ultimi{SUMMARY_LAST_N}"


def load_history() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(paths.FL_DIR, "alpha_*_seed*",
                                              "training_history.csv"))):
        frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(
            f"nessun risultato in {paths.FL_DIR} - lancia prima python -m federated.sweep")
    return pd.concat(frames, ignore_index=True)


def aggregate(summary_all: pd.DataFrame) -> pd.DataFrame:
    """Media sui seed, con errore standard e semiampiezza dell'intervallo al 95%.

    Con dieci draw si riporta media +- 1.96*SE, che e' lo standard. Con tre si poteva solo
    riportare lo half-range, una misura ad hoc: `ci95` qui sotto e' il sostituto, e
    `n_seed` dice su quanti draw e' calcolato (serve a non leggere un intervallo stimato
    su due run come se valesse qualcosa).
    """
    done = summary_all[summary_all["completed"]]
    agg = (done.groupby(["alpha", "epsilon"])[ACC_COL]
           .agg(acc_media="mean", acc_std_tra_seed="std", n_seed="size")
           .reset_index())
    agg["se"] = agg["acc_std_tra_seed"] / np.sqrt(agg["n_seed"].clip(lower=1))
    agg["ci95"] = 1.96 * agg["se"]
    return agg.sort_values(["alpha", "epsilon"])


def export(summary_all, summary_mean, history_all) -> None:
    os.makedirs(paths.FL_DIR, exist_ok=True)
    for name, df in (("summary_all.csv", summary_all),
                     ("summary_mean.csv", summary_mean),
                     ("history_all.csv", history_all)):
        path = os.path.join(paths.FL_DIR, name)
        df.to_csv(path, index=False)
        print(f"salvato: {path}  ({len(df)} righe)")


def plots(summary_mean: pd.DataFrame, history_all: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) accuracy vs epsilon, una curva per alpha, barre = 1.96*SE sui seed
    for alpha, g in summary_mean.groupby("alpha"):
        finite = g[np.isfinite(g["epsilon"])].sort_values("epsilon")
        if finite.empty:
            continue
        axes[0].errorbar(finite["epsilon"], finite["acc_media"], yerr=finite["ci95"],
                         marker="o", capsize=3, label=f"alpha={alpha:g}")
        base = g[~np.isfinite(g["epsilon"])]
        if not base.empty:
            axes[0].axhline(base["acc_media"].iloc[0], linestyle="--", alpha=0.35)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("epsilon (scala log)")
    axes[0].set_ylabel(f"accuracy (media ultimi {SUMMARY_LAST_N} round, media sui seed)")
    axes[0].set_title("Accuracy vs privacy, per livello di eterogeneita")
    axes[0].legend()

    # (b) convergenza del primo seed, all'alpha piu' eterogenea
    alpha_min = history_all["alpha"].min()
    seed0 = sorted(history_all["seed"].unique())[0]
    conv = history_all[(history_all["alpha"] == alpha_min)
                       & (history_all["seed"] == seed0)]
    for eps, g in conv.groupby("epsilon"):
        label = "no DP" if not np.isfinite(eps) else f"epsilon={eps:g}"
        axes[1].plot(*g.sort_values("round")[["round", "accuracy"]].values.T,
                     label=label, linewidth=1.2)
    axes[1].set_xlabel("round")
    axes[1].set_ylabel("accuracy (test set centralizzato)")
    axes[1].set_title(f"Convergenza (alpha={alpha_min:g}, seed={seed0})")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(paths.FL_DIR, "sweep_overview.png")
    plt.savefig(out, dpi=150)
    print(f"salvato: {out}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plots", action="store_true", help="salva anche le figure")
    args = p.parse_args(argv)

    history_all = load_history()
    summary_all = summarize_results(history_all)
    summary_mean = aggregate(summary_all)

    print(f"combinazioni trovate: {len(summary_all)}  "
          f"(interrotte: {int((~summary_all['completed']).sum())})")
    print(f"seed presenti: {sorted(summary_all['seed'].unique())}")
    print()
    print(summary_mean.to_string(index=False))

    incomplete = summary_all[~summary_all["completed"]]
    if not incomplete.empty:
        print(f"\n{len(incomplete)} run interrotte (metriche NON confrontabili):")
        print(incomplete[["alpha", "seed", "epsilon", "round"]].to_string(index=False))

    low = summary_mean[summary_mean["n_seed"] < len(CONFIG.seeds)]
    if not low.empty:
        print(f"\n{len(low)} celle hanno meno di {len(CONFIG.seeds)} seed: "
              f"l'intervallo al 95% su quelle e' da leggere con cautela.")

    print()
    export(summary_all, summary_mean, history_all)

    n_models = len(glob.glob(os.path.join(paths.FL_DIR, "alpha_*_seed*", "*.pt")))
    print(f"\nmodelli salvati su disco: {n_models}")

    if args.plots:
        plots(summary_mean, history_all)
    return 0


if __name__ == "__main__":
    sys.exit(main())
