"""Configurazione dello sweep federato.

Ogni valore qui dentro e' citato in tesi (Implementation): i commenti che lo giustificano
fanno parte della documentazione e non vanno rimossi insieme al codice.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

# Radice del repository, calcolata dalla posizione di questo file e non da os.getcwd():
# gli attori Ray della simulazione non garantiscono la working directory, e nel notebook
# tutti i percorsi erano relativi. Cosi' lo sweep si puo' lanciare da qualunque cartella.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.environ.get("FL_RESULTS_ROOT", os.path.join(REPO_ROOT, "results"))


@dataclass
class ExperimentConfig:
    # --- Setup federato ---
    num_clients: int = 6               # client simulati (SuperNode)
    # 100 e non 50: nelle diagnostiche D1, D2 e D5 il massimo cadeva sull'ULTIMO round,
    # cioe' nessuna run aveva convergito entro 50 round. A alpha=0.5 la seed 43 ha avuto
    # bisogno di ~180 round per arrivare a 0.951 (ma con il vecchio ottimizzatore).
    num_rounds: int = 100
    fraction_train: float = 1.0        # frazione di client campionati per il training
    fraction_evaluate: float = 1.0     # frazione di client campionati per la evaluation
    min_available_clients: int = 2

    # --- Iperparametri locali (client-side) ---
    local_epochs: int = 1
    batch_size: int = 128
    learning_rate: float = 0.03

    # momentum = 0.0, ed e' la modifica che sblocca gli alpha bassi.
    # Con m=0.9 la combinazione (alpha=0.5, seed=42) era MORTA: loss bloccata su ln2 per
    # 200 round, accuracy mai sopra 0.5. Il test in test_ottimizzatore.ipynb ha isolato la
    # causa: non e' la dimensione del passo. Due configurazioni con lo STESSO passo
    # effettivo lr/(1-m) = 0.03 danno esiti opposti -- lr=0.003 con m=0.9 resta a 0.500,
    # lr=0.03 con m=0.0 arriva a 0.703 -- quindi il problema e' l'accumulo in se'.
    # Motivo: un client quasi mono-classe produce gradienti tutti coerenti, il momentum li
    # amplifica (fino a 10x con m=0.9), il modello locale scappa lontano dai pesi globali e
    # la media di sei modelli scappati in direzioni diverse e' un predittore costante.
    # Costo della modifica dove le cose gia' funzionavano: a alpha=10 si passa da 0.961 a
    # 0.952, un punto. Guadagno a alpha=0.5 seed 42: da 0.500 (morta) a 0.703.
    momentum: float = 0.0

    # --- Aggregazione: FedProx ---
    fedprox_mu: float = 0.01           # peso del termine prossimale. 0.0 => FedAvg

    # --- Eterogeneita dei dati (Dirichlet partitioning) ---
    # I nomi delle cartelle usano f"alpha_{alpha:g}" (vedi paths.py), quindi 10.0 -> "10"
    # e 1.0 -> "1": una sola convenzione per tutti gli script. Nel notebook convivevano
    # "alpha_10" (int) per le run federate e "alpha_10.0" (float) per i baseline, e dato
    # che dca.py normalizza entrambi con float() le due cartelle collassavano sulla stessa
    # chiave sovrascrivendosi a vicenda.
    alphas: list = field(default_factory=lambda: [10.0, 1.0, 0.5, 0.4, 0.3, 0.2])
    # alpha = 10 e non 100 come ancora
    #   near-IID: a 10 la purezza media di classe per client e' 58,1% contro il 50% del
    #   caso IID puro, quindi in tesi va descritto come "mildly non-IID", non come
    #   baseline IID. In compenso la griglia e' meglio spaziata in eterogeneita'
    #   realizzata: i salti di purezza diventano 15,7 / 6,9 / 8,4 punti invece di
    #   21,2 / 6,9 / 8,4.
    # alpha = 0.2 e non 0.1: a 0.1 la probabilita' che un singolo draw rispetti
    #   min_partition_size=500 e' ~1.8%, e le partizioni che sopravvivono al vincolo
    #   assomigliano comunque a quelle di 0.2 (rapporto 11,3x contro 10,5x).
    min_partition_size: int = 500
    # Pavimento sulla dimensione della partizione: con lo split 80/20 interno al client
    # garantisce n_train >= 400, quindi sample rate q = 128/400 = 0.32 < 1. Limita il
    # quantity skew ma non lo elimina: a alpha=0.2 il rapporto mediano fra client piu'
    # grande e piu' piccolo resta ~11:1.
    partition_max_retries: int = 50    # retry esterno al DirichletPartitioner

    # --- Seed: draw Dirichlet indipendenti allo stesso alpha ---
    # Dieci e non tre. Con tre draw l'unica statistica disponibile era lo half-range, una
    # misura ad hoc da spiegare e difendere; con dieci si riporta media +- 1.96*SE, che e'
    # lo standard. L'errore standard scende del 45%. Nota che la simulazione non e'
    # bit-riproducibile (gli attori Ray e il generatore di Opacus non vengono riseminati),
    # quindi le medie sui seed vanno lette come stime con varianza -- ed e' anche cio' che
    # rende gratuite le ripetizioni a partizione fissa (vedi paths.set_replicate).
    seeds: list = field(default_factory=lambda: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51])

    # --- Differential Privacy: DP-SGD record-level lato client (Opacus) ---
    dp_enabled: bool = True
    run_baseline: bool = True          # run senza DP (epsilon = inf), riferimento
    # Griglia rivista dopo il primo sweep. Tolto 16 perche' ridondante: a alpha=10 gli
    # epsilon 8/16/32 davano 0.922/0.925/0.930, tre run per 0.8 punti. Aggiunto 0.5 perche'
    # il tratto piu' ripido della curva era fra 1 e 2 (0.751 -> 0.849), quindi il punto di
    # rottura della privacy sta piu' in basso e non era mai stato sfiorato.
    target_epsilons: list = field(default_factory=lambda: [0.5, 1, 2, 4, 8])
    target_delta: float = 1e-5         # delta << 1 / n_campioni_del_client

    # max_grad_norm = 5.0, ed e' l'altra modifica che conta.
    # Con C=1.0 il clipping, non il rumore, era il fattore limitante: a alpha=1 seed 42 la
    # run con epsilon=1000 (rumore trascurabile) restava a 0.623, mentre epsilon=32 con C=5
    # arrivava a 0.908. Trenta volte meno budget di privacy e 0.28 di accuracy in piu'.
    # C NON e' un parametro di privacy: e' il bound di sensibilita', e sigma scala con lui,
    # quindi (epsilon, delta) restano esattamente quelli dichiarati.
    max_grad_norm: float = 5.0

    # --- Esecuzione ---
    skip_completed: bool = True        # salta le combinazioni gia' su disco (ripresa)
    max_hours: float = 0.0             # > 0: lo sweep si ferma pulito dopo N ore e riprende
                                       # al rilancio. Il ciclo e' ordinato seed -> alpha ->
                                       # epsilon, quindi si interrompe lasciando seed interi.

    # GPU: 6 client su una sola scheda => 1/6 = 0.166 ciascuno, cosi' girano tutti in
    # parallelo. num_gpus in Ray e' una quota LOGICA, non un limite di memoria: va bene
    # perche' il modello e' da 102k parametri e i gradienti per-campione di Opacus su un
    # batch da 128 occupano ~50 MB per client, quindi si sta larghi anche su 8 GB.
    # Su CPU sola: {"num_cpus": 2, "num_gpus": 0.0}.
    client_resources: dict = field(
        default_factory=lambda: {"num_cpus": 1, "num_gpus": 0.166}
    )


CONFIG = ExperimentConfig()


def validate(cfg: ExperimentConfig = CONFIG) -> None:
    """Guardia sull'accounting DP.

    Il batch deve restare ben sotto la dimensione di TRAINING del client piu' piccolo
    (min_partition_size meno lo split 80/20 interno), altrimenti il sample rate
    q = batch / n_train si avvicina a 1, l'amplificazione da subsampling sparisce e sigma
    esplode. Il fattore 0.5 mantiene q <= 0.5 anche nel caso peggiore.
    """
    min_client_train = int(cfg.min_partition_size * 0.8)
    if cfg.batch_size > 0.5 * min_client_train:
        raise ValueError(
            f"batch_size={cfg.batch_size} troppo grande per min_partition_size="
            f"{cfg.min_partition_size}: il client piu' piccolo avrebbe "
            f"n_train={min_client_train} e q={cfg.batch_size / min_client_train:.2f}. "
            f"Abbassa il batch o alza la soglia."
        )


def num_sampled_clients(cfg: ExperimentConfig = CONFIG) -> int:
    """Client campionati per round: usato SOLO dalla strategia FedProx.

    Non entra nell'accounting di privacy: con DP-SGD record-level il budget dipende dal
    sample rate dei BATCH, non da quello dei client.
    """
    return max(cfg.min_available_clients, round(cfg.fraction_train * cfg.num_clients))


def privacy_settings(cfg: ExperimentConfig = CONFIG) -> list:
    """Le configurazioni di privacy di una riga dello sweep: None = baseline senza DP."""
    return ([None] if cfg.run_baseline else []) + (
        list(cfg.target_epsilons) if cfg.dp_enabled else []
    )


def set_seeds(seed: int) -> None:
    """Reimposta gli RNG del processo principale.

    NON risemina gli attori Ray ne' il generatore di rumore di Opacus: le run non sono
    bit-riproducibili, e questo va dichiarato fra i limiti in Methodology.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device() -> "torch.device":
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def describe(cfg: ExperimentConfig = CONFIG) -> str:
    n_priv = len(privacy_settings(cfg))
    n_runs = len(cfg.seeds) * len(cfg.alphas) * n_priv
    min_client_train = int(cfg.min_partition_size * 0.8)
    return "\n".join([
        f"{asdict(cfg)}",
        f"client campionati per round (train): {num_sampled_clients(cfg)}",
        f"run pianificate: {len(cfg.seeds)} seed x {len(cfg.alphas)} alpha x {n_priv} "
        f"configurazioni di privacy = {n_runs}",
        f"sample rate DP nel caso peggiore: q = {cfg.batch_size}/{min_client_train} "
        f"= {cfg.batch_size / min_client_train:.3f}",
    ])
