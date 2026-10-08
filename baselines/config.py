"""Configurazione dei baseline.

Tutto viene da `federated.config.CONFIG`, nessun valore e' riscritto qui. Nella versione
precedente (baselines.py) alpha, seed, learning rate, momentum, epoche e dimensione minima
della partizione erano ricopiati a mano, con il commento "DEVE combaciare con lo sweep":
bastava dimenticarne uno per confrontare il modello federato con baseline allenati in
condizioni diverse, e il failure point sarebbe stato misurato contro un riferimento sbagliato.
Leggerli dalla stessa fonte rende quel disallineamento impossibile.
"""

from __future__ import annotations

from federated.config import CONFIG, device, set_seeds   # noqa: F401  (set_seeds riesportata)

ALPHAS = list(CONFIG.alphas)
SEEDS = list(CONFIG.seeds)
NUM_CLIENTS = CONFIG.num_clients

# Budget di epoche identico a quello che un client consuma dentro la federazione:
# num_rounds round x local_epochs epoche locali.
EPOCHS = CONFIG.num_rounds * CONFIG.local_epochs
BATCH_SIZE = CONFIG.batch_size
LEARNING_RATE = CONFIG.learning_rate
MOMENTUM = CONFIG.momentum       # 0.0: con m=0.9 gli alpha bassi muoiono (vedi federated/config.py)

# Batch size della sola valutazione: non incide sul risultato, solo sulla velocita'.
EVAL_BATCH_SIZE = 256

# Il modello centralizzato non dipende dall'alpha: unendo le partizioni si ricostruisce lo
# stesso pool di training. Ma get_fds vuole un alpha per costruire il dataset, e serve il
# valore piu' mite, che il partitioner accetta al primo tentativo. Il risultato non cambia.
CENTRAL_PARTITION_ALPHA = 10.0

DEVICE = device()
