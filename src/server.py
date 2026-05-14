import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
from model import Net
from data_loader import load_data, load_centralized_dataset
from model import test
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
        #evaluate_metrics_aggr_fn=custom_aggregation_fn
        weighted_by_key="num-examples",
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

    print("Saving server side evaluation metrics...")
    pd.json_normalize(result.evaluate_metrics_serverapp).to_csv("server_evaluate_metrics.csv", index=False)



    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def custom_aggregation_fn(
    records: list[RecordDict], weighting_metric_name: str
) -> MetricRecord:
    
    metrics = [record.metric_records['metrics'] for record in records]
    total_accuracy = sum(metric["eval_acc"] for metric in metrics)
    total_log_loss = sum(metric["eval_log_loss"] for metric in metrics)
    num_clients = len(metrics)

    avg_accuracy = total_accuracy / num_clients
    avg_log_loss = total_log_loss / num_clients

    return MetricRecord({
        "avg_accuracy": round(avg_accuracy, 3),
        "avg_log_loss": round(avg_log_loss, 3)
    })

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
