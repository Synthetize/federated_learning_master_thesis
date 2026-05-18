# Copilot instructions for this repository

## Build, run, and test commands

Primary Flower app lives in `federated-learning-pytorch/`. Run commands from that directory.

- Install dependencies in the project virtualenv from repo root: `./.venv/bin/pip install -r requirements.txt`
- Install the Flower app package in editable mode: `cd federated-learning-pytorch && ../.venv/bin/pip install -e .`
- Run the federated simulation with current defaults from `pyproject.toml`: `cd federated-learning-pytorch && ../.venv/bin/flwr run . --stream`
- Smoke-test a single round (closest equivalent to a single test in this repo): `cd federated-learning-pytorch && ../.venv/bin/flwr run . --run-config "num-server-rounds=1" --stream`

There is currently no dedicated lint configuration and no `pytest` test suite in the repository.

## High-level architecture

- `federated-learning-pytorch/src/server_app.py` is the orchestration entrypoint (`@app.main`). It:
  - reads runtime config from `context.run_config`
  - creates the global `Net`
  - runs `FedProx.start(...)`
  - performs centralized evaluation via `global_evaluate`
  - persists artifacts: `final_model.pt`, `server_evaluate_metrics.csv`, and `client_evaluate_metrics.csv`
- `federated-learning-pytorch/src/client_app.py` defines both client endpoints:
  - `@app.train`: loads partitioned data, computes DP noise from target epsilon/delta, wraps model/optimizer/data loader with Opacus `PrivacyEngine`, returns updated arrays and metrics
  - `@app.evaluate`: evaluates local validation partition and returns `log_loss`, `accuracy`, and `num-examples`
- `federated-learning-pytorch/src/data_loader.py` centralizes data access:
  - builds a `FederatedDataset` with `DirichletPartitioner(alpha=0.1)` for non-IID client partitions
  - splits each partition into train/validation
  - provides centralized CIFAR-10 test loader for server-side evaluation
- `federated-learning-pytorch/src/model.py` contains the CNN (`Net`) plus shared train/eval helpers, including DP training helper used by `client_app.py`.

## Key repository conventions

- Use Flower message records consistently:
  - model weights under `"arrays"` (`ArrayRecord`)
  - metrics under `"metrics"` (`MetricRecord`)
  - weighted aggregation key is `"num-examples"`
- Runtime experiment knobs come from `[tool.flwr.app.config]` in `federated-learning-pytorch/pyproject.toml` and are consumed via `context.run_config` (for example `local-epochs`, `target-epsilon`, `max-grad-norm`).
- Client partition identity and federation topology are read from `context.node_config` (`partition-id`, `num-partitions`), not hardcoded.
- Differential privacy behavior is first-class in training:
  - compute `noise_multiplier` from target privacy budget
  - train through Opacus private wrappers
  - report privacy metrics (`epsilon`, `target_delta`, `noise_multiplier`, `max_grad_norm`) in each training reply
- Client state can persist across rounds using `context.state.config_records`; this is already used to track cumulative privacy accounting by round.

