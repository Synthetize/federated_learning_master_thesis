import warnings

import torch
from flwr.client import ClientApp
from flwr.common import Context, Message, ArrayRecord, MetricRecord, RecordDict, ConfigRecord
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent

from model import Net, train as model_train, test as model_test, train_dp as model_train_dp
from data_loader import load_data

warnings.filterwarnings("ignore", category=UserWarning)

app = ClientApp()


def log_cumulative_epsilon(context, trainloader, noise_multiplier, sample_rate, target_delta):
    """Calcola e stampa l'epsilon totale accumulato usando lo stato persistente del client."""
    try:
        # In Flower ServerApp, lo stato si gestisce tramite config_records (che è un RecordSet)
        if "comm_state" not in context.state.config_records:
            current_round = 1
        else:
            current_round = int(context.state.config_records["comm_state"]["round"]) + 1
        
        # Salviamo il round corrente nello stato per il prossimo utilizzo
        context.state.config_records["comm_state"] = ConfigRecord({"round": current_round})

        # Passi totali = (passi per epoca) * (epoche per round) * (numero di round)
        steps_per_epoch = len(trainloader)
        local_epochs = context.run_config["local-epochs"]
        total_steps = steps_per_epoch * local_epochs * current_round
        
        # Calcolo offline dell'epsilon cumulativo usando RDP
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        rdp = compute_rdp(
            q=sample_rate,
            noise_multiplier=noise_multiplier,
            steps=total_steps,
            orders=orders
        )
        cum_eps, _ = get_privacy_spent(
            orders=orders,
            rdp=rdp,
            delta=target_delta
        )
        
        print(f"\n>>> [TEST] CUMULATIVE EPSILON al Round {current_round}: {cum_eps:.4f}")
    except Exception as e:
        print(f">>> [DEBUG] log_cumulative_epsilon failed: {e}")
        pass


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _unwrap_state_dict(model: torch.nn.Module) -> dict:
    """Return plain state_dict even if Opacus wrapped the model."""
    return (
        model._module.state_dict() if hasattr(model, "_module") else model.state_dict()
    )


@app.train()
def train(msg: Message, context: Context):
    device = _device()
    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Read DP parameters from run_config
    max_grad_norm = float(context.run_config["max-grad-norm"])
    target_delta = float(context.run_config["target-delta"])
    target_epsilon = float(context.run_config["target-epsilon"])
    sample_rate = batch_size / len(trainloader.dataset)
    local_epochs = int(context.run_config["local-epochs"])
    num_rounds = int(context.run_config["num-server-rounds"])
    total_epochs = local_epochs * num_rounds



    noise_multiplier = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        epochs=total_epochs,
    )

    log_cumulative_epsilon(context, trainloader, noise_multiplier, sample_rate, target_delta)

    # Load the model and initialize it with the received weights
    model = Net().to(device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=msg.content["config"]["lr"],
    )
    # Attach Opacus PrivacyEngine — wraps model, optimizer, and dataloader
    privacy_engine = PrivacyEngine(secure_mode=False)
    private_model, optimizer, private_trainloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    # Train with DP-SGD
    train_loss, epsilon = model_train_dp(
        private_model,
        private_trainloader,
        privacy_engine,
        optimizer,
        target_delta,
        device=device,
        epochs=context.run_config["local-epochs"],
    )

    # print(
    #     f"[client {partition_id}] epsilon(delta={target_delta})={epsilon:.2f}, "
    #     f"noise={noise_multiplier}, max_grad_norm={max_grad_norm}, loss={train_loss:.4f}"
    # )

    print(f"[CLIENT {partition_id}] \n train loss={train_loss:.4f} \n epsilon(delta={target_delta})={epsilon:.2f} \n noise={noise_multiplier} \n max_grad_norm={max_grad_norm}")

    # Use _unwrap_state_dict to strip the Opacus wrapper before sending back
    out_arrays = ArrayRecord(_unwrap_state_dict(private_model))
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(private_trainloader.dataset),
        "epsilon": float(epsilon),
        "target_delta": float(target_delta),
        "noise_multiplier": float(noise_multiplier),
        "max_grad_norm": float(max_grad_norm),
    }
    out_metrics = MetricRecord(metrics)
    content = RecordDict({"arrays": out_arrays, "metrics": out_metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    device = _device()

    # Load the model and initialize it with the received weights
    model = Net().to(device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc = model_test(model, valloader, device)

    print(f"[client {partition_id}] eval loss={eval_loss:.4f}, acc={eval_acc:.4f}")

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)