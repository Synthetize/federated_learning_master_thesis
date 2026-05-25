from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
import tomllib

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_flwr_src_to_path() -> None:
    flwr_root = _repo_root() / "federated-learning-pytorch"
    sys.path.insert(0, str(flwr_root))


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _require(config: dict, key: str) -> None:
    if key not in config:
        raise KeyError(f"Missing required config key: {key}")


def _parse_config(config: dict) -> dict:
    for key in ["num_partitions", "batch_size", "learning_rate", "epochs", "dirichlet_alpha", "early_stopping_patience"]:
        _require(config, key)

    parsed = {
        "num_partitions": int(config["num_partitions"]),
        "batch_size": int(config["batch_size"]),
        "learning_rate": float(config["learning_rate"]),
        "epochs": int(config["epochs"]),
        "dirichlet_alpha": float(config["dirichlet_alpha"]),
        "early_stopping_patience": int(config["early_stopping_patience"]),
        "seed": config.get("seed"),
    }

    if parsed["num_partitions"] <= 0:
        raise ValueError("num_partitions must be > 0")
    if parsed["batch_size"] <= 0:
        raise ValueError("batch_size must be > 0")
    if parsed["epochs"] <= 0:
        raise ValueError("epochs must be > 0")
    if parsed["learning_rate"] <= 0:
        raise ValueError("learning_rate must be > 0")
    if parsed["dirichlet_alpha"] <= 0:
        raise ValueError("dirichlet_alpha must be > 0")
    if parsed["early_stopping_patience"] <= 0:
        raise ValueError("early_stopping_patience must be > 0")

    return parsed


def train_with_validation(net, train_loader, val_loader, epochs, lr, device, patience):
    """Train with validation monitoring and early stopping.
    
    Uses Adam optimizer with ReduceLROnPlateau scheduler.
    Returns: (avg_train_loss, best_val_loss, epochs_completed)
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
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epochs_completed = epoch + 1
                break
        
        epochs_completed = epoch + 1
    
    avg_train_loss = total_train_loss / epochs_completed
    return avg_train_loss, best_val_loss, epochs_completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Local training baseline per client partition")
    parser.add_argument(
        "--config",
        default="local_training/config.toml",
        help="Path to the local training config file",
    )
    args = parser.parse_args()

    _add_flwr_src_to_path()

    from src.data_loader import load_data, load_centralized_dataset
    from src.model import Net, test as model_test

    config_path = Path(args.config)
    parsed = _parse_config(_load_config(config_path))

    seed = parsed["seed"]
    if seed is not None:
        parsed["seed"] = int(seed)
    _set_seed(parsed["seed"])

    output_dir = _repo_root() / "local_training" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _device()
    num_partitions = parsed["num_partitions"]
    batch_size = parsed["batch_size"]
    lr = parsed["learning_rate"]
    epochs = parsed["epochs"]
    dirichlet_alpha = parsed["dirichlet_alpha"]
    early_stopping_patience = parsed["early_stopping_patience"]

    central_test_loader = load_centralized_dataset(batch_size=batch_size)

    # Collect per-partition results
    results: list[dict] = []

    for partition_id in range(num_partitions):
        train_loader, val_loader = load_data(
            partition_id=partition_id,
            num_partitions=num_partitions,
            batch_size=batch_size,
            dirichlet_alpha=dirichlet_alpha,
        )
        model = Net()
        train_loss, best_val_loss, epochs_completed = train_with_validation(
            model, train_loader, val_loader, epochs, lr, device, early_stopping_patience
        )

        central_log_loss, central_acc = model_test(model, central_test_loader, device)

        results.append({
            "partition_id": partition_id,
            "log_loss": round(float(central_log_loss), 4),
            "accuracy": round(float(central_acc), 4),
        })

    # Write CSV with config section (top) + results section (bottom)
    summary_path = output_dir / f"results_{num_partitions}c_{epochs}e.csv"
    with summary_path.open("w", newline="") as f:
        # Config section
        f.write("# Configuration\n")
        f.write(f"num_partitions,{num_partitions}\n")
        f.write(f"batch_size,{batch_size}\n")
        f.write(f"learning_rate,{lr}\n")
        f.write(f"epochs,{epochs}\n")
        f.write(f"dirichlet_alpha,{dirichlet_alpha}\n")
        f.write(f"early_stopping_patience,{early_stopping_patience}\n")
        f.write(f"seed,{parsed['seed']}\n")
        
        # Results section
        f.write("\n# Results\n")
        f.write("partition_id,log_loss,accuracy\n")
        for row in results:
            f.write(f"{row['partition_id']},{row['log_loss']},{row['accuracy']}\n")


if __name__ == "__main__":
    main()
