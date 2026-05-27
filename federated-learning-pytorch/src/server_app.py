import torch
from pathlib import Path
from typing import Iterable
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, Message
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedProx
from .model import Net
from .data_loader import load_centralized_dataset
from .model import test
from flwr.app import RecordDict
import pandas as pd
# Create ServerApp
app = ServerApp()


class CustomFedProx(FedProx):
    def configure_train(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        # decrease learning rate by half every 10 rounds
        if server_round > 1 and server_round % 10 == 0:
            config["lr"] *= 0.5
            print(f"[Round {server_round}] LR → {config['lr']:.6f}")

        return super().configure_train(server_round, arrays, config, grid)

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    target_epsilon: float = context.run_config["target-epsilon"]
    dirichlet_alpha: float = context.run_config["dirichlet-alpha"]
    experiment_suffix = (
        f"eps{target_epsilon}_"
        f"alpha{dirichlet_alpha}"
    )
    results_dir = _get_results_dir_path()

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = CustomFedProx(
        fraction_evaluate=fraction_evaluate,
        # evaluate_metrics_aggr_fn=custom_aggregation_fn
        # weighted_by_key="num-examples",
        proximal_mu=0.1
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )


    save_results(result, experiment_suffix, results_dir)

    print(f"\nSaving final model to disk in {results_dir}...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, results_dir / f"final_model_{experiment_suffix}.pt")

def _get_results_dir_path() -> Path:
    path = Path("results")
    path.mkdir(parents=True, exist_ok=True)
    return path

def _metrics_rows(metrics_by_round: dict[int, MetricRecord], source: str) -> list[dict]:
    rows = []
    for round_num, metrics in sorted(metrics_by_round.items()):
        print(f"Collecting {source} evaluation metrics for round {round_num}...")
        rows.append(
            {
                "Round": round_num,
                "log_loss": round(float(metrics.get("log_loss")), 3),
                "accuracy": round(float(metrics.get("accuracy")), 3),
            }
        )
    return rows

def save_results(
    result: RecordDict[MetricRecord],
    experiment_suffix: str | None = None,
    output_dir: Path | None = None,
) -> None:
    """Save server/client evaluation metrics to disk."""
    output_dir = output_dir if output_dir is not None else _get_results_dir_path()
    server_filename = f"server_evaluate_metrics_{experiment_suffix}.csv"
    client_filename = f"client_evaluate_metrics_{experiment_suffix}.csv"

    outputs = [
        (
            result.evaluate_metrics_serverapp,
            "server",
            server_filename,
        ),
        (
            result.evaluate_metrics_clientapp,
            "client",
            client_filename,
        ),
    ]

    for metrics_by_round, source, filename in outputs:
        rows = _metrics_rows(metrics_by_round, source)
        if rows:
            pd.DataFrame(rows).to_csv(output_dir / filename, index=False)

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
