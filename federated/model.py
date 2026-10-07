"""Rete, training locale, valutazione."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 2   # Parasitized vs Uninfected
IMAGE_SIZE = 32   # lato dell'immagine in input al modello


class Net(nn.Module):
    """CNN per crop di cellule malaria ridimensionati (RGB 32x32).

    Usa GroupNorm invece di BatchNorm: indipendente dalle statistiche di batch (adatto a
    batch piccoli/non-IID lato client) ed e' anche un requisito di Opacus, che rifiuta
    BatchNorm perche' mescola informazione tra campioni dello stesso batch e rende
    impossibile definire un gradiente per-campione.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm3 = nn.GroupNorm(8, 128)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        x = self.pool(F.relu(self.norm3(self.conv3(x))))
        x = self.global_pool(x).flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train(net, trainloader, epochs, lr, device, proximal_mu: float = 0.0,
          global_params=None, dp: dict | None = None, momentum: float = 0.9):
    """Training locale di un client.

    - `dp=None`: SGD normale (baseline non privata).
    - `dp` valorizzato: **DP-SGD record-level** via Opacus. Il `PrivacyEngine` clippa il
      gradiente di OGNI CAMPIONE a `max_grad_norm` e aggiunge rumore gaussiano prima
      dell'update, quindi i pesi rimandati al server sono gia' perturbati e il server non
      vede mai un update pulito (modello di minaccia honest-but-curious soddisfatto senza
      secure aggregation).

    `dp` e' un dict con: target_epsilon, target_delta, max_grad_norm, total_local_epochs.
    `total_local_epochs = num_rounds * local_epochs` e' il numero di epoche locali che il
    client eseguira' sull'INTERO training federato: passandolo a `make_private_with_epsilon`
    si ottiene un sigma calibrato sul budget totale, non su quello del singolo round.

    Termine prossimale di FedProx
    -----------------------------
    L'objective di FedProx e' h_k(w; w^t) = F_k(w) + (mu/2)*||w - w^t||^2, il cui gradiente
    e' grad F_k(w) + mu*(w - w^t). Qui il secondo pezzo viene applicato come step separato
    DOPO `optimizer.step()`, invece che sommato alla loss. Due motivi:

    1. **Correttezza sotto DP**: il `DPOptimizer` di Opacus ricostruisce `p.grad` dai
       gradienti per-campione (clippati + rumore), sovrascrivendolo. Un contributo scritto
       direttamente in `p.grad` sommando il termine alla loss verrebbe semplicemente perso.
    2. **Costo di privacy nullo**: il termine dipende solo dai pesi (locali e globali), che
       sono informazione pubblica, e non dai dati. Non deve quindi passare per il meccanismo
       DP ne' consumare budget.

    Rispetto alla formulazione accoppiata i due update coincidono a meno di un termine
    O(lr^2 * mu * grad): con lr=0.01 e mu=0.01 lo scarto relativo su uno step e' ~4e-4.

    Lo stesso schema si usa anche senza DP, cosi' baseline e run private ottimizzano
    esattamente lo stesso objective e restano confrontabili.

    NOTA su una correzione rispetto a una versione precedente: prima il termine veniva
    sommato alla loss come `(mu/2) * sum_i ||w_i - w_i^t||_2` usando `.norm(2)`, cioe' la
    somma delle NORME e non la norma al QUADRATO. Il gradiente che ne risulta e' un versore
    (w - w^t)/||w - w^t|| invece di (w - w^t), che non e' il termine prossimale di FedProx
    (Li et al., 2020). Qui e' implementato correttamente.
    """
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    # Riferimento ai Parameter originali. Opacus avvolge `net` in un GradSampleModule ma
    # NON copia i tensori: aggiornarli tramite il DPOptimizer aggiorna anche `net`, che e'
    # l'oggetto di cui il client fara' `state_dict()` per rispondere al server.
    params = list(net.parameters())

    forward_module = net
    privacy_engine = None
    if dp is not None:
        from opacus import PrivacyEngine

        privacy_engine = PrivacyEngine(accountant="rdp")
        forward_module, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
            module=net,
            optimizer=optimizer,
            data_loader=trainloader,
            target_epsilon=dp["target_epsilon"],
            target_delta=dp["target_delta"],
            epochs=dp["total_local_epochs"],
            max_grad_norm=dp["max_grad_norm"],
        )

    forward_module.train()
    running_loss, n_batches = 0.0, 0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            if labels.numel() == 0:
                continue  # il campionamento di Poisson di Opacus puo' dare batch vuoti
            optimizer.zero_grad()
            loss = criterion(forward_module(images), labels)
            loss.backward()
            optimizer.step()

            if proximal_mu > 0.0 and global_params is not None:
                # step prossimale disaccoppiato: w <- w - lr * mu * (w - w_globali)
                with torch.no_grad():
                    for p, g in zip(params, global_params):
                        p.add_(p - g, alpha=-lr * proximal_mu)

            running_loss += loss.item()
            n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)

    epsilon_round = None
    if privacy_engine is not None:
        # epsilon speso in QUESTO round: solo diagnostica. Il budget totale del client e'
        # dp["target_epsilon"], perche' sigma e' stato calibrato su total_local_epochs.
        epsilon_round = privacy_engine.get_epsilon(dp["target_delta"])
        if hasattr(forward_module, "remove_hooks"):
            forward_module.remove_hooks()

    return avg_loss, epsilon_round


def test(net, testloader, device):
    """loss media e accuracy. Usata dalla evaluation locale per round."""
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.inference_mode():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss / len(testloader), accuracy


@torch.inference_mode()
def evaluate_with_probs(net, loader, device):
    """Un solo passaggio sul loader: loss, accuracy, label grezze e P(label = 1).

    Restituisce di proposito la probabilita' della CLASSE 1 e le label GREZZE. Quale delle
    due classi sia l'evento clinico e' una scelta che appartiene all'analisi, non al
    training: tenendola fuori da qui, cambiarla non richiede di rifare nessuna run.
    """
    net.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    loss, correct, n_batches = 0.0, 0, 0
    probs, labels = [], []
    for batch in loader:
        x = batch["img"].to(device)
        y = batch["label"].to(device)
        out = net(x)
        loss += criterion(out, y).item()
        correct += (out.argmax(1) == y).sum().item()
        probs.append(torch.softmax(out, dim=1)[:, 1].cpu())
        labels.append(y.cpu())
        n_batches += 1
    p1 = torch.cat(probs).numpy().astype(np.float32)
    y_raw = torch.cat(labels).numpy().astype(np.int8)
    return loss / max(n_batches, 1), correct / len(loader.dataset), y_raw, p1


def num_parameters() -> int:
    return sum(p.numel() for p in Net().parameters())
