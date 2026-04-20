import warnings

import torch
from flwr.client import ClientApp
from flwr.common import Context, Message, ArrayRecord, MetricRecord, RecordDict
from opacus import PrivacyEngine

from model import Net, train as model_train, test as model_test, train_dp as model_train_dp
from data_loader import load_data

warnings.filterwarnings("ignore", category=UserWarning)

app = ClientApp()


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
    noise_multiplier = float(context.run_config["noise-multiplier"])
    max_grad_norm = float(context.run_config["max-grad-norm"])
    target_delta = float(context.run_config["target-delta"])

    # Load the model and initialize it with the received weights
    model = Net().to(device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)

    # Create optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=msg.content["config"]["lr"],
        momentum=0.9,
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

    print(
        f"[client {partition_id}] epsilon(delta={target_delta})={epsilon:.2f}, "
        f"noise={noise_multiplier}, max_grad_norm={max_grad_norm}, loss={train_loss:.4f}"
    )

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