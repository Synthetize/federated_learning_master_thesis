"""CLI dei baseline:  python -m baselines [local|central|all|summary] [opzioni]"""

from __future__ import annotations

import argparse
import sys

from . import config


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m baselines", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", nargs="?", default="all",
                   choices=["all", "local", "central", "summary"],
                   help="cosa allenare (default: all = locali e centralizzati); "
                        "summary non allena niente")
    p.add_argument("--alphas", type=float, nargs="+", default=None,
                   help="solo questi alpha (default: quelli di federated.config)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="solo questi seed (default: quelli di federated.config)")
    p.add_argument("--epochs", type=int, default=None,
                   help=f"epoche per modello (default {config.EPOCHS}, come lo sweep). "
                        f"Solo per prove rapide: cambiarle invalida il confronto")
    args = p.parse_args(argv)

    # import tardivi: --help e summary non devono caricare il dataset
    from .summary import print_summary
    if args.what == "summary":
        print_summary()
        return 0

    epochs = args.epochs if args.epochs is not None else config.EPOCHS
    print(f"device: {config.DEVICE}")
    print(f"lr={config.LEARNING_RATE} momentum={config.MOMENTUM} epoche={epochs} "
          f"batch={config.BATCH_SIZE}")
    if epochs != config.EPOCHS:
        print(f"ATTENZIONE: {epochs} epoche invece di {config.EPOCHS}. Risultati NON "
              f"confrontabili con lo sweep: usare solo per verificare che giri.")

    from .data import CELL_IMAGES_DIR
    print(f"dataset: {CELL_IMAGES_DIR}")

    if args.what in ("all", "local"):
        from .local import run_local
        run_local(args.alphas, args.seeds, epochs)
    if args.what in ("all", "central"):
        from .central import run_central
        run_central(args.seeds, epochs)
    print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
