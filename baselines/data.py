"""Baselines loaders.

The partitions, the 80/20 train/validation split of each client and the centralized test
set come from `federated.dataset`, i.e. from the same code the sweep uses. In the previous
version this part was a copy (get_fds, decode_to_tensors, DictTensorDataset) kept aligned
by hand: if the two copies diverged, the baselines were evaluated on a different test set
from the federated model's and the comparison meant nothing. Importing it is the only way
to guarantee they coincide.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from federated.dataset import (CELL_IMAGES_DIR, DictTensorDataset,   # noqa: F401
                               decode_to_tensors, get_fds, load_centralized_testset,
                               load_client_data)

from .config import BATCH_SIZE, CENTRAL_PARTITION_ALPHA, EVAL_BATCH_SIZE


def get_test_loader(alpha: float, seed: int) -> DataLoader:
    """Centralized test set of (alpha, seed): the same as the federated model's."""
    return load_centralized_testset(alpha, seed, EVAL_BATCH_SIZE)


def local_loaders(alpha: float, seed: int, client: int):
    """(training, validation) of one client: its partition split 80/20, as in the federated
    client. The validation set is only used to pick the best epoch."""
    return load_client_data(client, BATCH_SIZE, alpha, seed)


def central_loaders(seed: int):
    """(training, validation, test) of the centralized model of one seed.

    The pool is the union of all the partitions, i.e. the dataset's "train" split (the test
    set is already excluded), split 80/20 as for the clients. The test set is the seed's
    own: that is why the centralized model must be trained once PER SEED. A model trained on
    the seed-42 pool has seen most of the seed-43 test set, and its saved probabilities do
    not match the images of that test set (chance-level accuracy).
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
    """{class: number of samples} of a loader's dataset, for logging."""
    labels = loader.dataset.labels
    classes, counts = labels.unique(return_counts=True)
    return dict(zip(classes.tolist(), counts.tolist()))
