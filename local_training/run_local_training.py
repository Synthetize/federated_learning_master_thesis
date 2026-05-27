from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
import tomllib

from src.data_loader import load_data, load_centralized_dataset
from src.model import Net, test as model_test
import torch


def _get_root_folder() -> Path:
    return Path(__file__).resolve().parents[1]

def _load_config_file(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("rb") as handle:
        return tomllib.load(handle)
    
def _save_results(results: list[dict], output_dir: Path, configs: dict) -> None:
    summary_path = output_dir / f"results_{configs['num_partitions']}c_{configs['epochs']}e.csv"
    with summary_path.open("w", newline="") as f:
        # Config section
        f.write("# Configuration\n")
        f.write(f"num_partitions, batch_size, learning_rate, epochs, dirichlet_alpha, early_stopping_patience, seed\n")
        f.write(f"{configs['num_partitions']}, {configs['batch_size']}, {configs['learning_rate']}, {configs['epochs']}, {configs['dirichlet_alpha']}, {configs['early_stopping_patience']}, {configs['seed']}\n")
        
        # Results section
        f.write("\n# Results\n")
        f.write("partition_id,log_loss,accuracy\n")
        for row in results:
            f.write(f"{row['partition_id']},{row['log_loss']},{row['accuracy']}\n")

def train(net, train_loader, val_loader, epochs, lr, device, patience):
    """Train with validation monitoring and early stopping.
    
    Uses Adam optimizer with ReduceLROnPlateau scheduler.
    Returns: (avg_train_loss, best_val_loss, epochs_completed, best_model_state)
    """
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    
    net.to(device)
    net.train()
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)
    
    best_val_loss = float('inf')
    patience_counter = 0
    total_train_loss = 0.0
    epochs_completed = 0
    
    for epoch in range(epochs):
        # Training phase
        train_loss = 0.0
        for batch in train_loader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        total_train_loss += train_loss
        
        # Validation phase
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        net.train()
        
        # Learning rate scheduling and early stopping
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = net.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epochs_completed = epoch + 1
                break
        
        epochs_completed = epoch + 1
    
    avg_train_loss = total_train_loss / epochs_completed
    return avg_train_loss, best_val_loss, epochs_completed, best_model_state



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="local_training/config.toml",
    )
    args = parser.parse_args()
    configs = _load_config_file(Path(args.config))

    output_dir = _get_root_folder() / "local_training" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_split_dataset = load_centralized_dataset(batch_size=configs["batch_size"])["test"]

    # Collect per-partition results
    results: list[dict] = []

    for partition_id in range(configs["num_partitions"]):
        train_loader, val_loader = load_data(
            partition_id=partition_id,
            num_partitions=configs["num_partitions"],
            batch_size=configs["batch_size"],
            dirichlet_alpha=configs["dirichlet_alpha"],
        )
        model = Net()
        train_loss, best_val_loss, epochs_completed, best_model_state = train(
            model, train_loader, val_loader, configs["epochs"], configs["learning_rate"], device, configs["early_stopping_patience"]
        )

        log_loss, accuracy = model_test(model, test_split_dataset, device)

        results.append({
            "partition_id": partition_id,
            "log_loss": round(float(log_loss), 4),
            "accuracy": round(float(accuracy), 4),
        })

        torch.save(best_model_state, output_dir / f"best_model_partition_{partition_id}.pth")
        _save_results(results, output_dir, configs)



if __name__ == "__main__":
    main()
