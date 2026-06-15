# Federated Learning Master Thesis

Thesis repository for Federated Learning experiments with PyTorch and Flower, plus a local-training baseline.

## Repository structure

```
📦 federated_learning_master_thesis
├── 📄 THESIS.md
├── 📄 flower_docs.md
├── 📄 local_training_plan.md
├── 📄 plan.md
├── 📄 requirements.txt
├── 📁 federated-learning-pytorch
│   ├── 📄 pyproject.toml
│   ├── 📄 experiments.json
│   ├── 📄 run_batch_examples.py
│   ├── 📁 src
│   │   ├── 📄 server_app.py
│   │   ├── 📄 client_app.py
│   │   ├── 📄 data_loader.py
│   │   └── 📄 model.py
│   └── 📁 results
└── 📁 local_training
    ├── 📄 run_local_training.py
    ├── 📄 config.toml
    └── 📁 results
```

**Key details**
- `federated-learning-pytorch/src/server_app.py`: Flower server orchestration (FedProx strategy, central evaluation, results saving).
- `federated-learning-pytorch/src/client_app.py`: Flower client (DP training with Opacus + local evaluation).
- `federated-learning-pytorch/src/data_loader.py`: CIFAR-10 dataset with Dirichlet partitioning.
- `federated-learning-pytorch/src/model.py`: CNN and training/evaluation helpers.
- `local_training/run_local_training.py`: local baseline (one model per partition, early stopping).

## Quick setup

```
python -m venv .venv
./.venv/bin/pip install -r requirements.txt
cd federated-learning-pytorch && ../.venv/bin/pip install -e .
```

## Runs

**Federated simulation (Flower, defaults from pyproject.toml)**
```
cd federated-learning-pytorch
../.venv/bin/flwr run . --stream
```

**Smoke test (1 round)**
```
cd federated-learning-pytorch
../.venv/bin/flwr run . --run-config "num-server-rounds=1" --stream
```

**Experiment batch (target_epsilon / dirichlet_alpha)**
```
cd federated-learning-pytorch
../.venv/bin/python run_batch_examples.py --experiments experiments.json
```

**Local training (client-only baseline)**
```
./.venv/bin/python local_training/run_local_training.py --config local_training/config.toml
```

Main outputs:
- Federated: `federated-learning-pytorch/results/` (`final_model_*.pt`, `*_metrics_*.csv`)
- Local training: `local_training/results/` (`best_model_partition_*.pth`, `results_*c_*e.csv`)

## Configuration

**Federated (Flower)**
- File: `federated-learning-pytorch/pyproject.toml` → `[tool.flwr.app.config]` section.
- Key parameters: `num-server-rounds`, `fraction-evaluate`, `local-epochs`, `learning-rate`, `batch-size`, `max-grad-norm`, `target-delta`, `target-epsilon`, `dirichlet-alpha`.
- Quick CLI override:
  ```
  cd federated-learning-pytorch
  ../.venv/bin/flwr run . --run-config "num-server-rounds=5 target-epsilon=4.0 dirichlet-alpha=0.3" --stream
  ```

**Experiment batch**
- File: `federated-learning-pytorch/experiments.json`.
- Structure: list of `target_epsilon` / `dirichlet_alpha` pairs.

**Local training**
- File: `local_training/config.toml`.
- Parameters: `num_partitions`, `batch_size`, `learning_rate`, `epochs`, `dirichlet_alpha`, `early_stopping_patience`, `seed`.
