# Copilot Instructions for this Repository

## Build, test, and lint commands

Run commands from `src/` (that is where `pyproject.toml` lives).

- Install project deps: `python -m pip install -e .`
- Install Flower simulation extras (needed for local simulation runs): `python -m pip install "flwr[simulation]"`
- Run federated simulation: `flwr run . --stream`

There is currently no configured lint target and no automated test suite/check target in this repository, so there is no single-test command to run yet.

## High-level architecture

This project is a Flower federated-learning simulation around CIFAR-10 with FedProx on the server and Opacus differential privacy on clients.

- `server.py`: Defines `ServerApp` entrypoint. It reads run config from Flower context, initializes `Net`, runs rounds with `FedProx`, performs centralized evaluation via `global_evaluate`, and writes `final_model.pt`.
- `client.py`: Defines `ClientApp` train/evaluate handlers. In train, it loads partitioned data, computes DP noise multiplier from target privacy budget, wraps model/optimizer/dataloader with Opacus `PrivacyEngine`, trains with DP-SGD, and returns updated weights + metrics.
- `data_loader.py`: Builds federated partitions from CIFAR-10 using `DirichletPartitioner` (`partition_by="label"`, `alpha=0.1`, `seed=42`), creates train/validation loaders per client, and provides centralized test loader.
- `model.py`: Contains CNN model (`Net`) plus shared local train/test utilities (including DP training helper used by `client.py`).

## Key conventions in this codebase

- Flower config keys use hyphenated names in `pyproject.toml` (for example `num-server-rounds`, `local-epochs`, `target-epsilon`) and are accessed through `context.run_config[...]`.
- Flower message payloads follow the default keys: incoming train message expects `content["arrays"]` and `content["config"]`; client replies include `RecordDict({"arrays": ..., "metrics": ...})`.
- Aggregation weighting relies on `metrics["num-examples"]`; keep this metric in client replies if changing train/evaluate outputs.
- Client statefulness is used: cumulative-round tracking for privacy logging is stored under `context.state.config_records["comm_state"]`.
- Data batches are dict-like records with keys `img` and `label` (from Hugging Face datasets + transform pipeline), not `(x, y)` tuples.
- `data_loader.py` intentionally caches `FederatedDataset` in a module-level `fd` singleton so partitions are initialized once per process.
