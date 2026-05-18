import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
from .model import Net
from .data_loader import load_data, load_centralized_dataset
from .model import test
from flwr.app import RecordDict
import pandas as pd
# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedProx(
        fraction_evaluate=fraction_evaluate,
        # evaluate_metrics_aggr_fn=custom_aggregation_fn
        # weighted_by_key="num-examples",
        # proximal_mu=0.1
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )


    save_results(result)

      # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")

def _rounded_metric(metrics: MetricRecord, key: str) -> float | None:
    value = metrics.get(key)
    return round(float(value), 3) if value is not None else None


def _metrics_rows(metrics_by_round: dict[int, MetricRecord], source: str) -> list[dict]:
    rows = []
    for round_num, metrics in sorted(metrics_by_round.items()):
        print(f"Collecting {source} evaluation metrics for round {round_num}...")
        rows.append(
            {
                "Round": round_num,
                "Log_loss": _rounded_metric(metrics, "log_loss"),
                "accuracy": _rounded_metric(metrics, "accuracy"),
            }
        )
    return rows


def save_results(result: RecordDict[MetricRecord]) -> None:
    """Save server/client evaluation metrics to disk."""
    outputs = [
        (
            result.evaluate_metrics_serverapp,
            "server",
            "server_evaluate_metrics.csv",
        ),
        (
            result.evaluate_metrics_clientapp,
            "client",
            "client_evaluate_metrics.csv",
        ),
    ]

    for metrics_by_round, source, filename in outputs:
        rows = _metrics_rows(metrics_by_round, source)
        if rows:
            pd.DataFrame(rows).to_csv(filename, index=False)

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset(batch_size=128)

    # Evaluate the global model on the test set
    test_log_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "log_loss": test_log_loss})
