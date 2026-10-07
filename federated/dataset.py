"""Dataset, partizionamento Dirichlet e loader.

Il percorso del dataset viene risolto UNA volta all'import, in assoluto rispetto alla
radice del repository: nel notebook era relativo alla working directory, e in simulazione
gli attori Ray non la garantiscono.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize

from .config import CONFIG, REPO_ROOT, ExperimentConfig
from .model import IMAGE_SIZE


def find_cell_images_dir(base_dir: str) -> str:
    """Cartella che contiene direttamente Parasitized/ e Uninfected/.

    Il dataset Kaggle a volte ha una struttura annidata (es. cell_images/cell_images/...):
    si cercano tutte le occorrenze e si prende quella con piu' immagini.
    """
    candidates = []
    for parasitized in glob.glob(os.path.join(base_dir, "**", "Parasitized"), recursive=True):
        candidate = os.path.dirname(parasitized)
        if os.path.isdir(os.path.join(candidate, "Uninfected")):
            candidates.append((len(os.listdir(parasitized)), candidate))
    if not candidates:
        raise FileNotFoundError(
            f"Non ho trovato sottocartelle 'Parasitized' e 'Uninfected' sotto '{base_dir}'."
        )
    candidates.sort(reverse=True)
    return candidates[0][1]


def _resolve_dataset_dir() -> str:
    """Di default la copia a 32x32, che evita di ridimensionare a ogni accesso (dataset
    ~9x piu' leggero da leggere). Se non c'e', si ricade sull'originale."""
    override = os.environ.get("FL_CELL_IMAGES_DIR")
    if override:
        return find_cell_images_dir(override)
    for name in ("cell_images_32", "cell_images"):
        base = os.path.join(REPO_ROOT, name)
        if os.path.isdir(base):
            return find_cell_images_dir(base)
    raise FileNotFoundError(
        f"ne' 'cell_images_32' ne' 'cell_images' sotto {REPO_ROOT}. "
        f"Imposta FL_CELL_IMAGES_DIR se il dataset e' altrove."
    )


CELL_IMAGES_DIR = _resolve_dataset_dir()

_resize = Resize((IMAGE_SIZE, IMAGE_SIZE))

# Un solo (alpha, seed) alla volta resta in memoria: i tensori decodificati di tutto il
# dataset pesano ~270 MB, quindi tenerne una copia per ogni combinazione dello sweep
# esaurirebbe la RAM. Cambiando combinazione le cache vengono svuotate.
_FDS_CACHE: dict = {}
_PARTITIONER_CACHE: dict = {}
_CLIENT_CACHE: dict = {}      # (alpha, seed, pid, batch) -> (trainloader, valloader)
_TESTSET_CACHE: dict = {}     # (alpha, seed, batch) -> testloader
_POOLEDVAL_CACHE: dict = {}   # (alpha, seed, batch) -> valloader globale
# (alpha, seed) -> indice (0-based) del tentativo di draw che ha soddisfatto il vincolo di
# dimensione minima. NON viene svuotata: e' metadato leggero e serve al preflight per
# riassumere quanto si e' lavorato vicino al limite di tentativi.
_PARTITION_ATTEMPTS: dict = {}
_ACTIVE_KEY = None


def _make_split_fn(seed: int):
    """imagefolder produce solo lo split 'train': ne ricaviamo un test set centralizzato
    (20%, stratificato per label), mai visto dai client. Lo split dipende dal seed, quindi
    seed diversi danno anche test set diversi -- ed e' il motivo per cui il modello
    centralizzato va allenato una volta per seed."""
    def split_train_test(dataset_dict: DatasetDict) -> DatasetDict:
        full = dataset_dict["train"].rename_column("image", "img")
        split = full.train_test_split(test_size=0.2, seed=seed, stratify_by_column="label")
        return DatasetDict({"train": split["train"], "test": split["test"]})
    return split_train_test


def get_fds(alpha: float, seed: int, cfg: ExperimentConfig = CONFIG):
    """FederatedDataset partizionato con Dirichlet(alpha), draw determinato da `seed`.

    Chiamabile sia dal ServerApp sia dai client: ogni processo Ray costruisce la propria
    copia una volta sola. alpha/seed sono espliciti (invece di globali) per non dipendere
    da come Ray propaga lo stato fra i processi.
    """
    global _ACTIVE_KEY
    key = (alpha, seed)
    if _ACTIVE_KEY is not None and _ACTIVE_KEY != key:
        _FDS_CACHE.clear()
        _PARTITIONER_CACHE.clear()
        _CLIENT_CACHE.clear()
        _TESTSET_CACHE.clear()
        _POOLEDVAL_CACHE.clear()
    _ACTIVE_KEY = key

    if key not in _FDS_CACHE:
        last_error = None
        for attempt in range(cfg.partition_max_retries):
            # Seed del draw derivato deterministicamente dal seed nominale. Il seed
            # NOMINALE (42..51) resta quello registrato nei risultati e nei nomi delle
            # cartelle: cambia solo quale draw Dirichlet viene estratto. La derivazione e'
            # deterministica, quindi rilanciando si riottengono le stesse partizioni.
            draw_seed = seed * 10_000 + attempt
            partitioner = DirichletPartitioner(
                num_partitions=cfg.num_clients,
                partition_by="label",
                alpha=alpha,
                min_partition_size=cfg.min_partition_size,
                self_balancing=True,
                seed=draw_seed,
            )
            fds_new = FederatedDataset(
                dataset="imagefolder",
                data_dir=CELL_IMAGES_DIR,
                partitioners={"train": partitioner},
                # Lo split del test set centralizzato dipende dal seed NOMINALE, non da
                # draw_seed: deve restare identico fra i tentativi, altrimenti ogni
                # ridisegno cambierebbe il test set e le run non sarebbero comparabili.
                preprocessor=_make_split_fn(seed),
            )
            try:
                # FederatedDataset e' lazy: il partizionamento vero (e l'eventuale
                # ValueError su min_partition_size) avviene alla prima load_partition,
                # non alla costruzione. Il try DEVE avvolgere questa chiamata.
                fds_new.load_partition(0)
            except ValueError as exc:
                last_error = exc
                continue
            _PARTITION_ATTEMPTS[key] = attempt
            _PARTITIONER_CACHE[key] = partitioner
            _FDS_CACHE[key] = fds_new
            break
        else:
            raise RuntimeError(
                f"nessun draw Dirichlet valido per alpha={alpha}, seed={seed} in "
                f"{cfg.partition_max_retries} tentativi con min_partition_size="
                f"{cfg.min_partition_size}. Ultimo errore: {last_error}. Rimedi: alzare "
                f"alpha, abbassare min_partition_size (attenzione al sample rate DP), "
                f"oppure alzare partition_max_retries."
            )
    return _FDS_CACHE[key]


def get_partitioner(alpha: float, seed: int, cfg: ExperimentConfig = CONFIG):
    """Partitioner col dataset gia' assegnato (serve a plot_label_distributions)."""
    fds_ = get_fds(alpha, seed, cfg)
    fds_.load_partition(0)
    return _PARTITIONER_CACHE[(alpha, seed)]


def decode_to_tensors(hf_dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Decodifica UNA VOLTA SOLA l'intero dataset in due tensori in RAM (immagini
    normalizzate + label), invece di decodificare i PNG a ogni accesso.

    Le operazioni sono identiche a `Compose([Resize, ToTensor, Normalize(0.5, 0.5)])`.
    """
    n = len(hf_dataset)
    images = torch.empty((n, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
    labels = torch.empty(n, dtype=torch.int64)
    for i, example in enumerate(hf_dataset):
        img = example["img"].convert("RGB")
        if img.size != (IMAGE_SIZE, IMAGE_SIZE):
            img = _resize(img)
        arr = torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1)
        images[i] = arr.float().div_(255.0).sub_(0.5).div_(0.5)
        labels[i] = example["label"]
    return images, labels


class DictTensorDataset(Dataset):
    """Dataset su tensori gia' in RAM che restituisce dizionari {"img", "label"}.

    Deve essere un vero `torch.utils.data.Dataset` perche' Opacus rimpiazza il DataLoader
    con un `DPDataLoader` a campionamento di Poisson (requisito dell'analisi di privacy del
    Sampled Gaussian Mechanism). L'ottimizzazione che conta resta: le immagini vengono
    decodificate una volta sola, quindi `__getitem__` e' solo uno slice di tensore.
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return {"img": self.images[idx], "label": self.labels[idx]}


def load_client_data(partition_id: int, batch_size: int, alpha: float, seed: int):
    """Partizione di un client, divisa 80/20 in train/validation locali.

    Lo split di validation serve alla evaluation locale per round e, per i baseline
    locali, alla selezione del modello migliore (best-on-validation).
    """
    key = (alpha, seed, partition_id, batch_size)
    if key not in _CLIENT_CACHE:
        partition = get_fds(alpha, seed).load_partition(partition_id)
        split = partition.train_test_split(test_size=0.2, seed=seed)
        train_x, train_y = decode_to_tensors(split["train"])
        val_x, val_y = decode_to_tensors(split["test"])
        _CLIENT_CACHE[key] = (
            DataLoader(DictTensorDataset(train_x, train_y), batch_size=batch_size,
                       shuffle=True),
            DataLoader(DictTensorDataset(val_x, val_y), batch_size=batch_size,
                       shuffle=False),
        )
    return _CLIENT_CACHE[key]


def load_centralized_testset(alpha: float, seed: int, batch_size: int = 128):
    """Test set centralizzato, mai visto dai client. Dipende dal seed."""
    key = (alpha, seed, batch_size)
    if key not in _TESTSET_CACHE:
        test_x, test_y = decode_to_tensors(get_fds(alpha, seed).load_split("test"))
        _TESTSET_CACHE[key] = DataLoader(
            DictTensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False
        )
    return _TESTSET_CACHE[key]


def load_pooled_valset(alpha: float, seed: int, batch_size: int = 128,
                       cfg: ExperimentConfig = CONFIG):
    """Unione dei validation locali dei sei client: un validation set GLOBALE.

    Serve a due cose che senza di lui restano impossibili:
      1. scegliere il round migliore del modello federato senza selezionare sul test set,
         rendendo l'early stopping simmetrico a quello dei baseline locali, che usano
         best-on-validation;
      2. stimare una ricalibrazione (Platt) su dati mai visti e mai sul test.

    Lo split e' lo STESSO di load_client_data (test_size=0.2, seed=seed), quindi sono
    esattamente i campioni che i client tengono fuori dal proprio training, e nessuno di
    loro e' nel test set centralizzato.
    """
    key = (alpha, seed, batch_size)
    if key not in _POOLEDVAL_CACHE:
        xs, ys = [], []
        for pid in range(cfg.num_clients):
            partition = get_fds(alpha, seed, cfg).load_partition(pid)
            split = partition.train_test_split(test_size=0.2, seed=seed)
            vx, vy = decode_to_tensors(split["test"])
            xs.append(vx)
            ys.append(vy)
        _POOLEDVAL_CACHE[key] = DataLoader(
            DictTensorDataset(torch.cat(xs), torch.cat(ys)),
            batch_size=batch_size, shuffle=False,
        )
    return _POOLEDVAL_CACHE[key]


def preflight_partitions(cfg: ExperimentConfig = CONFIG, verbose: bool = True):
    """Verifica TUTTE le combinazioni (alpha, seed) dello sweep prima di lanciarlo.

    Costruire una partizione costa secondi, addestrare costa ore: farlo qui significa
    scoprire subito se una combinazione e' irrecuperabile, invece di trovare lo sweep
    fermo a meta' notte. Per ogni combinazione riporta la dimensione del client piu'
    piccolo e piu' grande, il loro rapporto (quantity skew), il sample rate DP nel caso
    peggiore e a quale tentativo il draw si e' risolto.
    """
    rows = []
    for alpha in cfg.alphas:
        for seed in cfg.seeds:
            fds_ = get_fds(alpha, seed, cfg)
            sizes = [len(fds_.load_partition(pid)) for pid in range(cfg.num_clients)]
            n_train_min = max(int(min(sizes) * 0.8), 1)   # split 80/20 interno al client
            row = {
                "alpha": alpha,
                "seed": seed,
                "min": min(sizes),
                "max": max(sizes),
                "ratio": max(sizes) / max(min(sizes), 1),
                "q_worst": cfg.batch_size / n_train_min,
                "attempt": _PARTITION_ATTEMPTS.get((alpha, seed), 0) + 1,
            }
            rows.append(row)
            if verbose:
                flag = ("   <-- q >= 1: ACCOUNTING DP NON VALIDO"
                        if row["q_worst"] >= 1.0 else "")
                print(f"  alpha={alpha:<7} seed={seed}: min={row['min']:<6} "
                      f"max={row['max']:<6} ratio={row['ratio']:>6.1f}x  "
                      f"q_worst={row['q_worst']:.3f}  tentativo {row['attempt']}{flag}")

    df = pd.DataFrame(rows)
    worst = int(df["attempt"].max())
    print(f"\n{len(df)} combinazioni su {len(df)} valide. "
          f"Tentativo peggiore: {worst} su {cfg.partition_max_retries} disponibili.")
    if worst > 0.6 * cfg.partition_max_retries:
        print("ATTENZIONE: si sta lavorando vicino al limite di tentativi. Valuta di "
              "alzare alpha o abbassare min_partition_size.")
    if (df["q_worst"] >= 1.0).any():
        raise RuntimeError(
            "almeno una combinazione ha sample rate >= 1: DP-SGD non e' applicabile. "
            "Alza min_partition_size o abbassa batch_size."
        )
    return df


def class_map(alpha: float, seed: int, cfg: ExperimentConfig = CONFIG) -> dict:
    """Mappa indice -> nome classe e quota di ciascuna nel test set.

    Le probabilita' salvate dallo sweep sono P(indice = 1). Quale classe sia l'EVENTO
    clinico si scegle nell'analisi (POSITIVE_LABEL in dca.py), non qui: cambiarla non
    richiede di rifare run.
    """
    split = get_fds(alpha, seed, cfg).load_split("test")
    names = split.features["label"].names
    y = np.asarray(split["label"])
    return {
        "names": {i: n for i, n in enumerate(names)},
        "shares": {n: float((y == i).mean()) for i, n in enumerate(names)},
        "n": int(len(y)),
    }
