"""Non-federated baselines: local models and the centralized model, per seed.

They define what the federated model has to beat (Methodology): Local_k of each client,
trained only on that client's data, and the Centralized model, trained on all the data
pooled together. Neither is federated and neither is trained under DP-SGD, so Flower, Ray
and Opacus are not needed here: it is a plain PyTorch loop.

    python -m baselines                 # local + centralized, all alphas and seeds
    python -m baselines local           # local models only
    python -m baselines central         # centralized models only
    python -m baselines summary         # summary only, trains nothing
    python -m baselines --help

Modules, in dependency order:

    config     alpha, seed, epochs, learning rate: READ from federated.config, not copied
    paths      where the files go and what they are called (what dca.py reads back)
    data       loaders for the clients, the test set and the centralized pool
    training   Net, evaluate, train_best_on_val (early stopping on validation)
    local      one model per (alpha, seed, client)
    central    one model per seed
    summary    summary tables from local_meta.csv and central_meta.csv

Nothing heavy is imported here, as in `federated`.
"""

__all__ = ["config", "paths", "data", "training", "local", "central", "summary"]
