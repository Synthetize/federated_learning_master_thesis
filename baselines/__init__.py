"""Baseline non federati: modelli locali e modello centralizzato, per seme.

Servono a definire cosa il modello federato deve battere (Methodology): il Local_k di
ciascun client, che si allena sui soli dati del client, e il Centralized, che si allena sui
dati di tutti messi insieme. Nessuno dei due e' federato e nessuno e' addestrato sotto
DP-SGD, quindi qui non servono Flower, Ray ne' Opacus: e' un semplice ciclo PyTorch.

    python -m baselines                 # locali + centralizzati, tutti gli alpha e i seed
    python -m baselines local           # solo i locali
    python -m baselines central         # solo i centralizzati
    python -m baselines summary         # solo il riepilogo, senza allenare niente
    python -m baselines --help

Moduli, in ordine di dipendenza:

    config     alpha, seed, epoche, learning rate: LETTI da federated.config, non copiati
    paths      dove finiscono i file e come si chiamano (e' cio' che dca.py rilegge)
    data       loader dei client, del test set e del pool centralizzato
    training   Net, evaluate, train_best_on_val (early stopping su validation)
    local      un modello per (alpha, seed, client)
    central    un modello per seed
    summary    tabelle di riepilogo da local_meta.csv e central_meta.csv

Non si importa niente di pesante qui, come in `federated`.
"""

__all__ = ["config", "paths", "data", "training", "local", "central", "summary"]
