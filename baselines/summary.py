"""Riepilogo dei baseline, riletto da local_meta.csv e central_meta.csv."""

from __future__ import annotations

import os

import pandas as pd

from . import paths


def _read(path: str, key: list):
    if not os.path.exists(path):
        return None
    # una run ripresa puo' aver riscritto la stessa riga: vale l'ultima
    return pd.read_csv(path).drop_duplicates(subset=key, keep="last")


def print_summary() -> None:
    local = _read(paths.LOCAL_META_CSV, ["alpha", "seed", "client"])
    central = _read(paths.CENTRAL_META_CSV, ["seed"])
    if local is None and central is None:
        print("nessun baseline su disco: nulla da riassumere")
        return

    hours = 0.0
    if local is not None:
        print("\n" + "=" * 78)
        print("ACCURACY DEI MODELLI LOCALI SUL TEST SET COMUNE (media sui client e sui seed)")
        print(local.pivot_table(index="alpha", values="test_acc",
                                aggfunc=["mean", "min", "max"]).round(4).to_string())
        print("\nper client, media sui seed:")
        print(local.pivot_table(index="alpha", columns="client", values="test_acc")
              .round(3).to_string())
        print("\ndimensione delle partizioni di training (media sui seed):")
        print(local.pivot_table(index="alpha", columns="client", values="n_train")
              .round(0).astype(int).to_string())
        print(f"\nmodelli locali: {len(local)} | seed: {[int(s) for s in sorted(local['seed'].unique())]}")
        hours += local["durata_min"].sum() / 60

    if central is not None:
        print("\n" + "=" * 78)
        print("MODELLI CENTRALIZZATI (uno per seed)")
        print(central.sort_values("seed")[["seed", "n_train", "val_acc", "best_epoch",
                                           "test_acc"]].round(4).to_string(index=False))
        print(f"\nmedia test acc: {central['test_acc'].mean():.4f} "
              f"su {len(central)} seed")
        hours += central["durata_min"].sum() / 60

    print(f"\ntempo totale di training: {hours:.1f} h")
    print("\nProssimo passo:  python dca.py")
