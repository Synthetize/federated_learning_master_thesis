"""ClientApp e ServerApp di Flower.

Nessun mod di clipping lato Flower e nessun wrapper
DifferentialPrivacyServerSideFixedClipping / ClientSideFixedClipping: quelli implementano
DP *client-level* con rumore aggiunto lato server, mentre qui la privacy e' record-level e
interamente dentro il training locale del client (model.train con Opacus). Il server non
vede mai un update non perturbato, quindi il modello di minaccia honest-but-curious e'
soddisfatto senza secure aggregation.
"""

from __future__ import annotations

import os

import numpy as np
from flwr.app import (ArrayRecord, ConfigRecord, Context, Message, MetricRecord,
                      RecordDict)
from flwr.clientapp import ClientApp

from .config import CONFIG, device
from .dataset import load_centralized_testset, load_client_data, load_pooled_valset
from .model import Net, evaluate_with_probs, test, train
from .paths import (labels_path, probs_path, results_dir, val_labels_path,
                    val_probs_path)

client_app = ClientApp()

# Ultima combinazione (alpha, seed) vista da questo attore in un messaggio di training.
# Rete di sicurezza per la evaluation: `train_config` e `evaluate_config` sono due
# ConfigRecord distinti, e se il secondo non arrivasse popolato il client saprebbe comunque
# su quale partizione lavorare (train precede sempre evaluate nello stesso round).
_LAST_RUN_KEY: dict = {}

# Metriche della run in corso, accumulate round per round da global_evaluate. Le stesse
# metriche sono nel Result di strategy.start(), ma quello arriva solo se la run finisce:
# tenendone una copia qui, anche una run interrotta a meta' conserva i round fatti.
CURRENT_RUN_METRICS: list = []


def _cfg_get(cfg, key, default=None):
    """ConfigRecord non espone .get(): questo lo emula."""
    try:
        return cfg[key]
    except KeyError:
        return default


@client_app.train()
def client_train(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    dev = device()
    model.to(dev)

    # Snapshot dei pesi globali ricevuti, usato dal termine prossimale di FedProx.
    global_params = [p.detach().clone() for p in model.parameters()]

    cfg = msg.content["config"]
    # alpha e seed arrivano dal server nel ConfigRecord: il client costruisce (o riusa
    # dalla cache) la propria partizione per quella combinazione, senza dipendere da
    # globali condivise tra processi Ray.
    alpha = float(cfg["alpha"])
    seed = int(cfg["seed"])
    _LAST_RUN_KEY["alpha"], _LAST_RUN_KEY["seed"] = alpha, seed
    partition_id = context.node_config["partition-id"]
    trainloader, _ = load_client_data(partition_id, CONFIG.batch_size, alpha, seed)

    target_epsilon = float(cfg["target-epsilon"])
    dp = None
    if target_epsilon > 0:
        dp = {
            "target_epsilon": target_epsilon,
            "target_delta": float(cfg["target-delta"]),
            "max_grad_norm": float(cfg["max-grad-norm"]),
            "total_local_epochs": int(cfg["total-local-epochs"]),
        }

    train_loss, epsilon_round = train(
        model,
        trainloader,
        CONFIG.local_epochs,
        float(cfg["lr"]),
        dev,
        proximal_mu=float(cfg["proximal-mu"]),
        global_params=global_params,
        dp=dp,
        momentum=float(_cfg_get(cfg, "momentum", CONFIG.momentum)),
    )

    metrics = {"train_loss": train_loss, "num-examples": len(trainloader.dataset)}
    if epsilon_round is not None:
        metrics["epsilon_round"] = epsilon_round

    content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord(metrics),
    })
    return Message(content=content, reply_to=msg)


@client_app.evaluate()
def client_evaluate(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    dev = device()

    cfg = msg.content["config"]
    # I messaggi di evaluate portano `evaluate_config`, non `train_config`: se alpha/seed
    # non ci fossero, si ricade sull'ultima combinazione vista in training.
    alpha = _cfg_get(cfg, "alpha", _LAST_RUN_KEY.get("alpha"))
    seed = _cfg_get(cfg, "seed", _LAST_RUN_KEY.get("seed"))
    if alpha is None or seed is None:
        raise RuntimeError(
            "client_evaluate non sa su quale (alpha, seed) lavorare: ne' evaluate_config "
            "ne' un training precedente li hanno forniti."
        )
    alpha, seed = float(alpha), int(seed)
    partition_id = context.node_config["partition-id"]
    _, valloader = load_client_data(partition_id, CONFIG.batch_size, alpha, seed)

    eval_loss, eval_acc = test(model, valloader, dev)

    content = RecordDict({
        "metrics": MetricRecord({
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "num-examples": len(valloader.dataset),
        }),
    })
    return Message(content=content, reply_to=msg)


# ---------------------------------------------------------------------------------
# Valutazione centralizzata: probabilita' salvate round per round
# ---------------------------------------------------------------------------------
# Lo sweep NON calcola il Net Benefit: salva soltanto le probabilita' predette, e la DCA si
# calcola dopo con dca.py a partire da questi file. Cosi' la griglia di soglie, la classe
# considerata evento e la finestra di media restano modificabili senza rifare una run.
#
# La utility del modello federato e' definita in Methodology come media sugli ultimi W
# round: servono quindi le predizioni di OGNI round, perche' i pesi finali salvati a fine
# run non bastano -- i modelli intermedi non esistono piu'.
#
# Dal 7 ottobre si salva anche lo stream sul validation pooled. Costa un forward in piu'
# per round su ~4.400 immagini e abilita, offline e senza riallenare: (a) la selezione del
# round su validation, che rende l'early stopping simmetrico a quello dei baseline locali,
# e (b) la ricalibrazione (Platt). Non servono checkpoint: scelto il round con
# argmax(val_accuracy), la riga corrispondente dello stream di test e' gia' su disco.

def make_global_evaluate(alpha: float, seed: int, epsilon=None):
    """Valutazione sul test set globale della combinazione (alpha, seed), piu' lo stream
    di validation. Scrivere round per round invece che a fine run significa che anche una
    run interrotta lascia utilizzabili i round gia' completati.
    """
    eps_value = float("inf") if epsilon is None else float(epsilon)

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = Net()
        model.load_state_dict(arrays.to_torch_state_dict())
        dev = device()

        test_loader = load_centralized_testset(alpha, seed)
        test_loss, test_acc, y_raw, p1 = evaluate_with_probs(model, test_loader, dev)

        os.makedirs(results_dir(alpha, seed), exist_ok=True)
        lp = labels_path(alpha, seed)
        if not os.path.exists(lp):
            np.save(lp, y_raw)      # le label dipendono solo dal seed: una volta sola
        with open(probs_path(alpha, seed, eps_value), "ab") as fb:
            fb.write(np.ascontiguousarray(p1, dtype=np.float32).tobytes())

        val_loader = load_pooled_valset(alpha, seed)
        _, val_acc, y_val, p1_val = evaluate_with_probs(model, val_loader, dev)
        vlp = val_labels_path(alpha, seed)
        if not os.path.exists(vlp):
            np.save(vlp, y_val)
        with open(val_probs_path(alpha, seed, eps_value), "ab") as fb:
            fb.write(np.ascontiguousarray(p1_val, dtype=np.float32).tobytes())

        CURRENT_RUN_METRICS.append({
            "round": server_round, "accuracy": test_acc, "loss": test_loss,
            "val_accuracy": val_acc,
        })
        return MetricRecord({"accuracy": test_acc, "loss": test_loss,
                             "val_accuracy": val_acc})

    return global_evaluate


def build_train_config(alpha: float, seed: int, target_epsilon) -> ConfigRecord:
    return ConfigRecord({
        "lr": CONFIG.learning_rate,
        "momentum": float(CONFIG.momentum),
        # letta da CONFIG e non da una costante, cosi' un override la vede. La strategia
        # FedProx inietta gia' "proximal-mu", ma metterla esplicitamente rende il codice
        # indipendente dalla versione di Flower.
        "proximal-mu": float(CONFIG.fedprox_mu),
        "alpha": float(alpha),
        "seed": int(seed),
        "target-epsilon": float(target_epsilon) if target_epsilon is not None else -1.0,
        "target-delta": float(CONFIG.target_delta),
        "max-grad-norm": float(CONFIG.max_grad_norm),
        # epoche locali totali del client sull'intero training federato: serve a Opacus per
        # calibrare sigma sul budget TOTALE e non su quello di un round
        "total-local-epochs": int(CONFIG.num_rounds * CONFIG.local_epochs),
    })


def initial_arrays() -> ArrayRecord:
    return ArrayRecord(Net().state_dict())


__all__ = [
    "client_app", "make_global_evaluate", "build_train_config", "initial_arrays",
    "CURRENT_RUN_METRICS",
]
