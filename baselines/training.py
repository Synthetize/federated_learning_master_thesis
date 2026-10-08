"""Valutazione e training dei baseline: un ciclo PyTorch, SGD semplice, niente DP."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from federated.model import Net

from .config import DEVICE, LEARNING_RATE, MOMENTUM


@torch.inference_mode()
def evaluate(net: nn.Module, loader):
    """Restituisce loss, accuracy, label grezze (int8) e P(classe 1) (float32)."""
    net.to(DEVICE).eval()
    criterion = nn.CrossEntropyLoss()
    loss, correct, n_batches = 0.0, 0, 0
    probs, labels = [], []
    for batch in loader:
        x = batch["img"].to(DEVICE)
        y = batch["label"].to(DEVICE)
        out = net(x)
        loss += criterion(out, y).item()
        correct += (out.argmax(1) == y).sum().item()
        probs.append(torch.softmax(out, 1)[:, 1].cpu())
        labels.append(y.cpu())
        n_batches += 1
    return (loss / max(n_batches, 1), correct / len(loader.dataset),
            torch.cat(labels).numpy().astype(np.int8),
            torch.cat(probs).numpy().astype(np.float32))


def train_best_on_val(trainloader, valloader, epochs: int, tag: str = "",
                      log_every: int = 25):
    """Allena e restituisce il modello dell'EPOCA MIGLIORE sulla validation.

    E' l'early stopping previsto dalla Methodology per i baseline: non essendo addestrati
    sotto DP non hanno l'instabilita' round per round del modello federato, e quindi non
    richiedono la media su finestra. La validation serve solo a scegliere il modello: il
    Net Benefit riportato si calcola sul test set comune.

    Restituisce (modello, accuracy di validation dell'epoca scelta, epoca scelta 1-based).
    """
    net = Net().to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    best_state, best_acc, best_epoch = copy.deepcopy(net.state_dict()), -1.0, -1
    for epoch in range(1, epochs + 1):
        net.train()
        for batch in trainloader:
            x = batch["img"].to(DEVICE)
            y = batch["label"].to(DEVICE)
            optimizer.zero_grad()
            criterion(net(x), y).backward()
            optimizer.step()
        _, acc, _, _ = evaluate(net, valloader)
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            best_state = copy.deepcopy(net.state_dict())
        if log_every and epoch % log_every == 0:
            print(f"      {tag} epoca {epoch}/{epochs}: val {acc:.4f} "
                  f"(migliore {best_acc:.4f} @ {best_epoch})")
    net.load_state_dict(best_state)
    return net, best_acc, best_epoch
