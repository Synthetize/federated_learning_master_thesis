"""
Baseline non federati: Local_k (un modello per client, sui soli dati del client) e
Centralized (un modello sui dati di tutti i client messi insieme).

Non usa Flower, Ray o Opacus: nessuno dei due baseline e' federato e nessuno dei due e'
addestrato sotto DP-SGD (Methodology). E' un semplice ciclo PyTorch.

    python baselines.py              # locali + centralizzato
    python baselines.py local        # solo i locali
    python baselines.py central      # solo il centralizzato

Riprendibile: salta quello che trova gia' su disco, quindi si puo' interrompere e rilanciare.

COSA SCRIVE (e' esattamente cio' che dca.py legge)
    results/baselines/alpha_<alpha>/probs_local_s<seed>_c<client>.npy   P(classe 1), float32
    results/baselines/probs_central.npy    P(classe 1) del centralizzato
    results/baselines/test_labels_s<seed>.npy   label grezze 0/1 (dipendono solo dal seed)
    results/baselines/local_meta.csv       n_train, val acc, epoca scelta, test acc, durata

I modelli locali sono raggruppati in una cartella per alpha; il centralizzato e le label
restano in cima perche' non dipendono da alpha. Un eventuale layout piatto prodotto da una
versione precedente viene riconosciuto e non riallenato.

Le probabilita' sono salvate GREZZE, come nello sweep: quale classe sia l'evento clinico e
con quali soglie calcolare il Net Benefit sono scelte che restano all'analisi.
"""

from __future__ import annotations

import copy
import glob
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ===================================================================== configurazione
# DEVE combaciare con quella dello sweep federato, altrimenti il confronto contro cui si
# definisce il failure point non e' pulito.
ALPHAS = [0.4]
SEEDS = [42, 43, 44]
NUM_CLIENTS = 6
MIN_PARTITION_SIZE = 500
PARTITION_MAX_RETRIES = 50

NUM_ROUNDS = 100          # lo stesso num_rounds dello sweep
LOCAL_EPOCHS = 1
EPOCHS = NUM_ROUNDS * LOCAL_EPOCHS   # budget di epoche identico a quello che un client
                                     # consuma dentro la federazione
BATCH_SIZE = 128
LEARNING_RATE = 0.03
MOMENTUM = 0.0            # da test_ottimizzatore.ipynb: con m=0.9 gli alpha bassi muoiono

# Il centralizzato si allena UNA volta sola (Methodology): unendo le partizioni l'alpha non
# conta piu', serve solo il seed che determina lo split del test set.
CENTRALIZED_SEED = SEEDS[0]

IMAGE_SIZE = 32
NUM_CLASSES = 2
OUT_DIR = os.path.join("results", "baselines")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ===================================================================== dati
from datasets import DatasetDict                       # noqa: E402
from flwr_datasets import FederatedDataset             # noqa: E402
from flwr_datasets.partitioner import DirichletPartitioner   # noqa: E402
from torchvision.transforms import Resize              # noqa: E402

_CELL_BASE = "cell_images_32" if os.path.isdir("cell_images_32") else "cell_images"


def find_cell_images_dir(base_dir: str) -> str:
    cands = []
    for p in glob.glob(os.path.join(base_dir, "**", "Parasitized"), recursive=True):
        d = os.path.dirname(p)
        if os.path.isdir(os.path.join(d, "Uninfected")):
            cands.append((len(os.listdir(p)), d))
    if not cands:
        raise FileNotFoundError(
            f"nessuna cartella con 'Parasitized' e 'Uninfected' sotto '{base_dir}'. "
            "Lancia questo script dalla stessa cartella del notebook dello sweep.")
    cands.sort(reverse=True)
    return cands[0][1]


CELL_IMAGES_DIR = find_cell_images_dir(_CELL_BASE)
_resize = Resize((IMAGE_SIZE, IMAGE_SIZE))
_FDS_CACHE: dict = {}


def get_fds(alpha: float, seed: int):
    """Stesse partizioni dello sweep: stesso partitioner, stessa soglia, stesso seed
    derivato per il retry, stesso split 80/20 del test set centralizzato.

    Se questi non combaciassero, i baseline verrebbero valutati su un test set diverso da
    quello del modello federato e il confronto non avrebbe senso.
    """
    key = (alpha, seed)
    if key in _FDS_CACHE:
        return _FDS_CACHE[key]

    def split_fn(dd: DatasetDict) -> DatasetDict:
        full = dd["train"].rename_column("image", "img")
        sp = full.train_test_split(test_size=0.2, seed=seed, stratify_by_column="label")
        return DatasetDict({"train": sp["train"], "test": sp["test"]})

    last = None
    for attempt in range(PARTITION_MAX_RETRIES):
        part = DirichletPartitioner(
            num_partitions=NUM_CLIENTS, partition_by="label", alpha=alpha,
            min_partition_size=MIN_PARTITION_SIZE, self_balancing=True,
            seed=seed * 10_000 + attempt,
        )
        fds = FederatedDataset(dataset="imagefolder", data_dir=CELL_IMAGES_DIR,
                               partitioners={"train": part}, preprocessor=split_fn)
        try:
            fds.load_partition(0)      # FederatedDataset e' lazy: l'errore arriva qui
        except ValueError as exc:
            last = exc
            continue
        _FDS_CACHE.clear()             # una combinazione alla volta in RAM (~340 MB)
        _FDS_CACHE[key] = fds
        return fds
    raise RuntimeError(f"nessun draw valido per alpha={alpha}, seed={seed}: {last}")


def decode_to_tensors(hf_dataset):
    """Equivalente a Compose([Resize, ToTensor, Normalize(0.5, 0.5)]), una volta sola."""
    n = len(hf_dataset)
    images = torch.empty((n, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
    labels = torch.empty(n, dtype=torch.int64)
    for i, ex in enumerate(hf_dataset):
        img = ex["img"].convert("RGB")
        if img.size != (IMAGE_SIZE, IMAGE_SIZE):
            img = _resize(img)
        arr = torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1)
        images[i] = arr.float().div_(255.0).sub_(0.5).div_(0.5)
        labels[i] = ex["label"]
    return images, labels


class DictTensorDataset(Dataset):
    def __init__(self, x, y):
        self.images, self.labels = x, y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"img": self.images[i], "label": self.labels[i]}


def loaders_from(hf_split, batch_size, shuffle):
    x, y = decode_to_tensors(hf_split)
    return DataLoader(DictTensorDataset(x, y), batch_size=batch_size, shuffle=shuffle), y


# ===================================================================== modello
class Net(nn.Module):
    """Identica a quella dello sweep: GroupNorm e non BatchNorm."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.norm3 = nn.GroupNorm(8, 128)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        x = self.pool(F.relu(self.norm3(self.conv3(x))))
        x = self.global_pool(x).flatten(1)
        return self.fc2(F.relu(self.fc1(x)))


@torch.inference_mode()
def evaluate(net, loader):
    """Restituisce loss, accuracy, label grezze e P(classe 1)."""
    net.to(DEVICE).eval()
    crit = nn.CrossEntropyLoss()
    loss, correct, nb = 0.0, 0, 0
    probs, labels = [], []
    for batch in loader:
        x = batch["img"].to(DEVICE)
        y = batch["label"].to(DEVICE)
        out = net(x)
        loss += crit(out, y).item()
        correct += (out.argmax(1) == y).sum().item()
        probs.append(torch.softmax(out, 1)[:, 1].cpu())
        labels.append(y.cpu())
        nb += 1
    return (loss / max(nb, 1), correct / len(loader.dataset),
            torch.cat(labels).numpy().astype(np.int8),
            torch.cat(probs).numpy().astype(np.float32))


def train_best_on_val(trainloader, valloader, epochs, tag="", log_every=25):
    """Allena e restituisce il modello dell'EPOCA MIGLIORE sulla validation.

    E' l'early stopping previsto dalla Methodology per i baseline, che non essendo
    addestrati sotto DP non hanno l'instabilita' round-per-round del modello federato e
    quindi non richiedono la media su finestra. La validation serve solo a scegliere il
    modello: il Net Benefit riportato si calcola sul test set comune.
    """
    net = Net().to(DEVICE)
    crit = nn.CrossEntropyLoss().to(DEVICE)
    opt = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    best_state, best_acc, best_ep = copy.deepcopy(net.state_dict()), -1.0, -1
    for ep in range(1, epochs + 1):
        net.train()
        for batch in trainloader:
            x = batch["img"].to(DEVICE)
            y = batch["label"].to(DEVICE)
            opt.zero_grad()
            crit(net(x), y).backward()
            opt.step()
        _, acc, _, _ = evaluate(net, valloader)
        if acc > best_acc:
            best_acc, best_ep = acc, ep
            best_state = copy.deepcopy(net.state_dict())
        if log_every and ep % log_every == 0:
            print(f"      {tag} epoca {ep}/{epochs}: val {acc:.4f} "
                  f"(migliore {best_acc:.4f} @ {best_ep})")
    net.load_state_dict(best_state)
    return net, best_acc, best_ep


# ===================================================================== baseline locali
def local_dir(alpha) -> str:
    return os.path.join(OUT_DIR, f"alpha_{alpha}")


def local_path(alpha, seed, client) -> str:
    """Un file per modello, raggruppati in una cartella per alpha."""
    return os.path.join(local_dir(alpha), f"probs_local_s{seed}_c{client}.npy")


def legacy_local_path(alpha, seed, client) -> str:
    """Layout piatto della versione precedente. Serve solo a non riallenare modelli che
    esistono gia' da una run fatta prima di questa modifica."""
    return os.path.join(OUT_DIR, f"probs_local_a{alpha}_s{seed}_c{client}.npy")


def local_done(alpha, seed, client) -> bool:
    return (os.path.exists(local_path(alpha, seed, client))
            or os.path.exists(legacy_local_path(alpha, seed, client)))


def labels_path(seed):
    """In cima e non dentro alpha_*: le label del test set dipendono solo dal seed."""
    return os.path.join(OUT_DIR, f"test_labels_s{seed}.npy")


META_CSV = os.path.join(OUT_DIR, "local_meta.csv")


def run_local():
    os.makedirs(OUT_DIR, exist_ok=True)
    todo = [(a, s, c) for a in ALPHAS for s in SEEDS for c in range(NUM_CLIENTS)
            if not local_done(a, s, c)]
    print(f"modelli locali da allenare: {len(todo)} su "
          f"{len(ALPHAS) * len(SEEDS) * NUM_CLIENTS} ({EPOCHS} epoche ciascuno)\n")
    if not todo:
        return

    for alpha in ALPHAS:
        for seed in SEEDS:
            if all(local_done(alpha, seed, c) for c in range(NUM_CLIENTS)):
                print(f"[alpha={alpha} seed={seed}] gia' completo, salto")
                continue
            print(f"\n=== alpha={alpha} seed={seed}")
            fds = get_fds(alpha, seed)
            test_loader, y_test = loaders_from(fds.load_split("test"), 256, False)
            if not os.path.exists(labels_path(seed)):
                np.save(labels_path(seed), y_test.numpy().astype(np.int8))

            for cid in range(NUM_CLIENTS):
                if local_done(alpha, seed, cid):
                    continue
                set_seeds(seed * 100 + cid)
                part = fds.load_partition(cid)
                sp = part.train_test_split(test_size=0.2, seed=seed)  # come nel client
                tl, ytr = loaders_from(sp["train"], BATCH_SIZE, True)
                vl, _ = loaders_from(sp["test"], 256, False)
                cls = dict(zip(*[v.tolist() for v in torch.unique(ytr, return_counts=True)]))
                print(f"  client {cid}: {len(ytr)} campioni di training, classi {cls}")

                t0 = time.time()
                net, val_acc, best_ep = train_best_on_val(tl, vl, EPOCHS, tag=f"c{cid}")
                _, test_acc, y_raw, p1 = evaluate(net, test_loader)
                os.makedirs(local_dir(alpha), exist_ok=True)
                np.save(local_path(alpha, seed, cid), p1)
                dur = (time.time() - t0) / 60
                pd.DataFrame([{"alpha": alpha, "seed": seed, "client": cid,
                               "n_train": len(ytr), "classi": str(cls),
                               "val_acc": val_acc, "best_epoch": best_ep,
                               "test_acc": test_acc, "durata_min": dur}]).to_csv(
                    META_CSV, mode="a", header=not os.path.exists(META_CSV), index=False)
                print(f"      -> test acc {test_acc:.4f} | epoca scelta {best_ep}/{EPOCHS} "
                      f"| {dur:.1f} min")


# ===================================================================== centralizzato
# CENTRAL_NPY = os.path.join(OUT_DIR, "probs_central.npy")


# def run_central():
#     os.makedirs(OUT_DIR, exist_ok=True)
#     if os.path.exists(CENTRAL_NPY):
#         print("centralizzato gia' su disco, salto")
#         return
#     print(f"\n=== centralizzato (seed {CENTRALIZED_SEED})")
#     fds = get_fds(ALPHAS[0], CENTRALIZED_SEED)
#     pool = fds.load_split("train")      # pool di training: il test set e' gia' escluso
#     sp = pool.train_test_split(test_size=0.2, seed=CENTRALIZED_SEED)
#     set_seeds(CENTRALIZED_SEED)
#     tl, ytr = loaders_from(sp["train"], BATCH_SIZE, True)
#     vl, _ = loaders_from(sp["test"], 256, False)
#     test_loader, y_test = loaders_from(fds.load_split("test"), 256, False)
#     if not os.path.exists(labels_path(CENTRALIZED_SEED)):
#         np.save(labels_path(CENTRALIZED_SEED), y_test.numpy().astype(np.int8))
#     print(f"  training su {len(ytr)} campioni, validation su {len(vl.dataset)}")

#     t0 = time.time()
#     net, val_acc, best_ep = train_best_on_val(tl, vl, EPOCHS, tag="central", log_every=10)
#     _, test_acc, y_raw, p1 = evaluate(net, test_loader)
#     np.save(CENTRAL_NPY, p1)
#     torch.save(net.state_dict(), os.path.join(OUT_DIR, "central_weights.pt"))
#     print(f"  -> test acc {test_acc:.4f} | epoca scelta {best_ep}/{EPOCHS} "
#           f"| {(time.time() - t0) / 60:.1f} min")


# ===================================================================== riepilogo
def summary():
    if not os.path.exists(META_CSV):
        return
    m = pd.read_csv(META_CSV).drop_duplicates(subset=["alpha", "seed", "client"], keep="last")
    print("\n" + "=" * 78)
    print("ACCURACY DEI MODELLI LOCALI SUL TEST SET COMUNE (media sui client e sui seed)")
    print(m.pivot_table(index="alpha", values="test_acc", aggfunc=["mean", "min", "max"])
          .round(4).to_string())
    print("\nper client, media sui seed:")
    print(m.pivot_table(index="alpha", columns="client", values="test_acc")
          .round(3).to_string())
    print("\ndimensione delle partizioni di training:")
    print(m.pivot_table(index="alpha", columns="client", values="n_train")
          .astype(int).to_string())
    print(f"\ntempo totale: {m.durata_min.sum() / 60:.1f} h")
    print("\nProssimo passo:  python dca.py")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"device: {DEVICE} | dataset: {CELL_IMAGES_DIR}")
    print(f"lr={LEARNING_RATE} momentum={MOMENTUM} epoche={EPOCHS} batch={BATCH_SIZE}")
    if what in ("all", "local"):
        run_local()
    if what in ("all", "central"):
        run_central()
    summary()