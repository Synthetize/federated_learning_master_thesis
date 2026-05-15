import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
from .model import Net
from .data_loader import load_data, load_centralized_dataset
from .model import test
from flwr.app import MetricRecord, RecordDict
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

    # save_results(result.evaluate_metrics_serverapp)

    

      # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def save_results(metrics: RecordDict[MetricRecord]) -> None:
    """Save evaluation metrics and final model to disk."""
    rows = []
    for round_num, metrics in metrics.items():
        print(f"Collecting evaluation metrics for round {round_num}...")
        rows.append(
            {
                "Round": round_num,
                "Log_loss": round(metrics.get("log_loss"), 3),
                "accuracy": round(metrics.get("accuracy"), 3),
            }
        )

    if rows:
        metrics_df = pd.DataFrame(rows)
        metrics_df.to_csv("server_evaluate_metrics.csv", index=False)

  


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
