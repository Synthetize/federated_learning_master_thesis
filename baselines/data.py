"""Loader dei baseline.

Le partizioni, lo split 80/20 train/validation di ogni client e il test set centralizzato
vengono da `federated.dataset`, cioe' dallo stesso codice che usa lo sweep. Nella versione
precedente questa parte era una copia (get_fds, decode_to_tensors, DictTensorDataset)
tenuta allineata a mano: se le due copie divergevano, i baseline venivano valutati su un
test set diverso da quello del modello federato e il confronto non voleva dire niente.
Importarla e' l'unico modo di garantire che coincidano.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from federated.dataset import (CELL_IMAGES_DIR, DictTensorDataset,   # noqa: F401
                               decode_to_tensors, get_fds, load_centralized_testset,
                               load_client_data)

from .config import BATCH_SIZE, CENTRAL_PARTITION_ALPHA, EVAL_BATCH_SIZE


def get_test_loader(alpha: float, seed: int) -> DataLoader:
    """Test set centralizzato di (alpha, seed): lo stesso del modello federato."""
    return load_centralized_testset(alpha, seed, EVAL_BATCH_SIZE)


def local_loaders(alpha: float, seed: int, client: int):
    """(training, validation) di un client: la sua partizione divisa 80/20, come nel client
    federato. La validation serve solo a scegliere l'epoca migliore."""
    return load_client_data(client, BATCH_SIZE, alpha, seed)


def central_loaders(seed: int):
    """(training, validation, test) del modello centralizzato di un seed.

    Il pool e' l'unione di tutte le partizioni, cioe' lo split "train" del dataset (il test
    set e' gia' escluso), diviso 80/20 come per i client. Il test set e' quello del seed:
    per questo il centralizzato va allenato una volta PER SEED. Un modello allenato sul pool
    del seed 42 ha visto buona parte del test set del seed 43, e le sue probabilita' salvate
    non corrispondono alle immagini di quel test set (accuracy a livello del caso).
    """
    fds = get_fds(CENTRAL_PARTITION_ALPHA, seed)
    pool = fds.load_split("train")
    split = pool.train_test_split(test_size=0.2, seed=seed)
    tx, ty = decode_to_tensors(split["train"])
    vx, vy = decode_to_tensors(split["test"])
    train = DataLoader(DictTensorDataset(tx, ty), batch_size=BATCH_SIZE, shuffle=True)
    val = DataLoader(DictTensorDataset(vx, vy), batch_size=EVAL_BATCH_SIZE, shuffle=False)
    return train, val, get_test_loader(CENTRAL_PARTITION_ALPHA, seed)


def class_counts(loader: DataLoader) -> dict:
    """{classe: numero di campioni} del dataset di un loader, per il log."""
    labels = loader.dataset.labels
    classes, counts = labels.unique(return_counts=True)
    return dict(zip(classes.tolist(), counts.tolist()))
