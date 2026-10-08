# Differentially Private Federated Learning on Malaria Cell Images, evaluated with Decision Curve Analysis

Master's thesis code. It trains a CNN on the NIH malaria cell-image dataset
(Parasitized / Uninfected) under **federated learning** (Flower, FedProx) with
**record-level DP-SGD** on each client (Opacus), sweeping two stress axes:

| Axis | Parameter | Values |
|---|---|---|
| Data heterogeneity | Dirichlet `alpha` | 10, 1.0, 0.5, 0.4, 0.3, 0.2 (mild → harsh) |
| Privacy budget | `epsilon` | `inf` (no DP), 8, 4, 2, 1, 0.5 (loose → tight) |

Each cell of that grid is repeated over 3 independent partition draws
(`seeds = 42, 43, 44`), giving 6 x 6 x 3 runs of 100 federated rounds.

The point of the project is **not** accuracy. Every run stores the raw predicted
probabilities of the global model at every round, and a separate analysis stage
scores them with **Decision Curve Analysis** (Net Benefit, Vickers & Elkin 2006) to
answer a clinical question: *at which privacy budget and which degree of
heterogeneity does the federated model stop being worth using?*

Three verdicts come out of the analysis:

- **Safe Range** — the longest contiguous band of decision thresholds where the
  federated model beats *both* trivial strategies (treat everyone / treat nobody)
  *and* the average single-hospital model trained alone.
- **Failure point** — a configuration whose Safe Range is empty.
- **Cost decomposition** — the total utility loss against a centralised model, split
  into *Federation*, *Heterogeneity*, *Privacy* and *Interaction* terms that sum
  exactly: `U = F + H + P + I`.

---

## Repository layout

```
federated/               [1] the federated sweep, as a package -> results/malaria/
  config.py                  ExperimentConfig: the single source of truth
  paths.py                   every output path + the fixed-partition replicate index
  model.py                   Net, DP-SGD training, FedProx proximal term, evaluation
  dataset.py                 Dirichlet partitioning, loaders, pooled validation set
  privacy.py                 sigma per client, noise/signal preview, sigma table
  app.py                     Flower ClientApp + per-round centralised evaluation
  sweep.py                   one run, saving, the sweep loop, the CLI
  report.py                  aggregate tables and diagnostic figures
baselines/               [2] non-federated baselines -> results/baselines/ (package)
  config.py                  read from federated.config, nothing copied
  paths.py                   output names (what dca.py reads)
  data.py                    loaders, reused from federated.dataset
  training.py                evaluate, train_best_on_val (early stopping)
  local.py                   one model per (alpha, seed, client)
  central.py                 one centralised model per seed
  summary.py                 tables from local_meta.csv / central_meta.csv
dca.py                   [3] the analysis: DCA, costs, Safe Range -> results/dca/
cell_images_32/          the dataset, pre-resized to 32x32
results/                 every artefact produced by the three stages
legacy/malaria_dpsgd.ipynb   the notebook this package replaced, kept for its diagnostics
requirements.txt         dependencies
```

The numbers in brackets are the execution order. Stages are **resumable**: each one
skips whatever it already finds on disk, so a run can be interrupted and relaunched.

---

## What each file does

### `federated/` — the federated sweep

The main experiment. Until 7 October 2026 this was a 33-cell notebook
(`legacy/malaria_dpsgd.ipynb`); it is now a package, module by module:

| Module | What it holds |
|---|---|
| `config.py` | `ExperimentConfig`, the single source of truth for every hyperparameter, plus a DP-accounting guard that refuses a batch size too close to the smallest client's training split. Also `REPO_ROOT` / `RESULTS_ROOT`, so paths no longer depend on the working directory |
| `paths.py` | Every output path, one naming convention (`alpha_{alpha:g}`), and `set_replicate()` for fixed-partition repeats |
| `model.py` | `Net` (3 conv blocks, **GroupNorm** not BatchNorm — BatchNorm is incompatible with per-sample gradients), `train` (DP-SGD + the decoupled FedProx proximal step), `test`, `evaluate_with_probs` |
| `dataset.py` | Dataset discovery, Dirichlet partitioning into 6 clients, the 80/20 split of the global test set, the per-client 80/20 split, the **pooled validation set**, and `preflight_partitions()` |
| `privacy.py` | Opacus' `get_noise_multiplier` asked for the sigma each client will need, *before* any training starts, plus `sigma_table()` |
| `app.py` | The Flower `ClientApp` (local DP-SGD + proximal term) and `make_global_evaluate` (centralised evaluation per round, test **and** validation streams) |
| `sweep.py` | One run, saving, the seed → alpha → epsilon loop with `skip_completed` resumption and a `max_hours` wall clock, the replicate sub-sweep, and the CLI |
| `report.py` | Reads every result back from disk, aggregates over seeds with mean ± 1.96·SE, exports the summary CSVs |

The diagnostics that used to live in notebook sections 7.1–7.3 were already fully
commented out and were **not** ported. They are the record of *why* the config looks the
way it does, and they stay in `legacy/` — see `legacy/README.md`.

| Section | What it holds |
|---|---|
| 1 | `pip install` cell (commented out by default) |
| 2 | `ExperimentConfig`, the single source of truth for every hyperparameter — plus a DP-accounting guard that asserts the batch size stays well below the smallest client's training split |
| 3 | Dataset discovery, Dirichlet partitioning into 6 clients, and the 80/20 train/test split of the global test set |
| 3.1 | Label-distribution figures, one per alpha: the visual evidence of the heterogeneity |
| 4 | `Net` (3 conv blocks, **GroupNorm** not BatchNorm — BatchNorm is incompatible with per-sample gradients), plus `train` / `evaluate` |
| 5 | Privacy accounting preview: Opacus' `get_noise_multiplier` is asked for the sigma each client will need, *before* any training starts |
| 6 | The Flower `ClientApp` (local DP-SGD training + the FedProx proximal term) and the `ServerApp` (FedProx aggregation, centralised evaluation per round) |
| 6.1 | Sanity check that `imagefolder` really maps Parasitized→0, Uninfected→1 |
| 7 | The sweep itself: loops seed → alpha → epsilon, with `skip_completed` resumption and a `max_hours` wall clock that stops cleanly |
| 7.1–7.3 | Diagnostics, all commented out. They are the record of *why* the config looks the way it does: why low alpha failed, whether FedProx's proximal term is actually applied, and whether the optimiser was making clients diverge |
| 8 | Reads every result back from disk and plots accuracy / loss against epsilon and alpha |
| 9 | Exports the summary CSVs for the thesis |

Two configuration choices carry the whole experiment and are documented at length in
section 2:

- **`momentum = 0.0`.** With `momentum=0.9` every low-alpha run was dead — loss stuck
  at ln2, accuracy never above 0.5. A near-single-class client produces gradients that
  all point the same way, momentum amplifies them, the local model runs far from the
  global weights, and the average of six models that each ran off in a different
  direction is a constant predictor. Cost where things already worked: one point of
  accuracy at alpha=10.
- **`max_grad_norm = 5.0`.** With `C=1.0` *clipping*, not noise, was the binding
  constraint: a run at epsilon=1000 (negligible noise) sat at 0.623 while epsilon=32
  with `C=5` reached 0.908. `C` is a sensitivity bound, not a privacy parameter —
  sigma scales with it, so `(epsilon, delta)` stay exactly as declared.

**Writes**, per `(alpha, seed)`, into `results/malaria/alpha_<a>_seed<s>/`:

- `probs_eps<e>.f32` — appended float32, **one row per round**, `P(label == 1)` on the
  shared test set. Written round by round, so an interrupted run still leaves usable
  rounds behind; a trailing incomplete row is detected and dropped downstream.
- `test_labels.npy` — the raw 0/1 test labels (they depend on the seed only).
- `epsilon_<e>.pt` / `no_dp.pt` — the final global weights.
- `config.json`, `training_history.csv`, `final_results.csv`.

Plus, at the top level: `summary_all.csv`, `summary_mean.csv`, `history_all.csv`.

### `baselines/` — the two non-federated references

A plain PyTorch loop: no Flower, no Ray, no Opacus, because neither baseline is
federated and neither is trained under DP.

- **`Local_k`** — one model per client, trained on that client's partition only. This
  is "what one hospital achieves alone", and the opponent the federation has to beat
  to justify its existence.
- **`Centralized`** — one model on all clients' data pooled back together, **one per
  seed**. The upper bound the cost decomposition measures against.

Nothing is copied from the sweep any more: alphas, seeds, learning rate, momentum, epoch
budget (`EPOCHS = num_rounds * local_epochs = 100`) and batch size are read from
`federated.config`, and partitions, the 80/20 client split and the test set are built by
`federated.dataset`. The previous `baselines.py` kept its own copy of all of it with a
comment saying it "MUST match" the sweep; with a single source it cannot drift. Baselines
use early stopping on a validation split, which the federated model does not: they have
no DP noise and so no round-to-round instability requiring a window average.

On the old code the refactor was checked to be bit-identical (same probabilities, same
test labels for the same seed), so nothing about the baselines' results changed.

**Why one centralised model per seed.** The centralised model is invariant to the
Dirichlet partition (pooling the clients back together reconstructs the same training
pool whatever alpha is), but not to the seed, because the seed also fixes the 80/20
train/test split. Seed 43's test set is a different sample of images, so a model trained
on seed 42's pool scored against seed 43's labels collapses to chance (0.4964, against
0.9641 on its own seed). Only the Utility Cost and the Federation Cost touch the
centralised model; Heterogeneity, Privacy, the Interaction and the entire Safe Zone are
FL-minus-FL differences taken inside one draw. One run per seed also gives the
Federation Cost an error bar.

**Writes** into `results/baselines/`:

```
alpha_<a>/probs_local_s<seed>_c<client>.npy   P(class 1), float32
probs_central_s<seed>.npy                     P(class 1) of the centralised model
central_weights_s<seed>.pt                    its weights
test_labels_s<seed>.npy                       raw 0/1 labels
local_meta.csv                                n_train, val acc, chosen epoch, test acc, duration
central_meta.csv                              same, one row per seed
```

The directory name uses `f"alpha_{alpha:g}"`, the same convention as
`federated/paths.py` (`alpha_10`, `alpha_1`): the old script wrote `alpha_10.0` while the
sweep wrote `alpha_10`, two directories that `dca.py` normalises with `float()` and that
collapsed onto the same key in `_local_files()`, overwriting each other.

> **History.** `baselines.py` and `central_seeds.py` (commit `24166df2`) are replaced by
> this package. The old `run_central()` trained a single centralised model on seed 42 and
> was commented out while still being called (`NameError`); it is gone, and
> `python -m baselines central` is the correct route. The old `probs_central.npy`
> fallback in `dca.py` is still there but no longer produced.

### `dca.py` — the analysis

The largest file (~1350 lines) and the one that produces every number in the results
chapter. Standalone: it depends on no notebook, only on the files on disk.

Reading order inside the file:

| Region | Contents |
|---|---|
| module docstring | the full specification: input formats, outputs, the cost decomposition algebra, grid orientation |
| configuration | `POSITIVE_LABEL`, `THRESHOLDS`, `WINDOW`, `CLINICAL_RANGES`, `COST_STAT`, `MIN_SAFE_RUN` |
| DCA primitives | `net_benefit` (vectorised: sort once, then binary-search TP/FP per threshold), `treat_all` |
| sweep reading | `load_runs`, `fl_nb` — averages the last `WINDOW` rounds |
| baselines | `baseline_nb` — local and centralised curves, both kept **per seed** |
| analysis | `add_costs`, `analyze`, `_safe_range`, `report` |
| figures | decision curves, cost heat maps, Safe Range maps, cost profiles, LaTeX tables |

Key decisions it makes, all of them load-bearing:

- **Probabilities are stored raw**, so the event class, the threshold grid and the
  averaging window are *post-processing* choices. Changing any of them requires no run
  to be repeated.
- **`POSITIVE_LABEL = 0`** — Parasitized is the clinical event (`imagefolder` assigns
  indices alphabetically).
- **The decomposition is computed inside each seed, then averaged.** Every cost is a
  linear difference, so the averages are identical either way — what this buys is a
  meaningful spread: with both cells of a difference taken from the same draw, the
  disagreement left between seeds is about the *cost*, not about how good that draw
  happened to be. Spread is reported as a half-range (`_hr` suffix), not a standard
  deviation: with three draws a standard deviation invites being read as a confidence
  interval it cannot support.
- **`CLINICAL_RANGES = [(0.01, 0.20)]`** — for malaria screening a false negative is
  far more serious than a false positive: at `p_t = 0.20` one FN is worth 4 FP, at
  `p_t = 0.01` it is worth 99. Above 0.20 one would be assuming a false alarm is nearly
  as serious as a missed infection. More than one window can be listed; every table,
  CSV and heat map is then produced once per window, and the full threshold grid is
  always reported as well.
- **`MIN_SAFE_RUN = 3`** — a "range" one threshold wide is not a range. Without this
  rule the map reports widths smaller than the noise between draws.
- **`CURVE_XMAX`** zooms the figures only; every number is still computed over the
  whole threshold grid.

**Writes** into `results/dca/` — where `<w>` is the window tag (empty for the full
grid, `_0p01-0p2` for the clinical window):

```
full_analysis.csv       one row per (alpha, epsilon, threshold)
safe_range.csv          every window, one row per configuration
costs<w>.csv            the cost decomposition
safe_range<w>.tex       table ready to paste into the thesis
costs<w>.tex            cost decomposition, ready to paste
figures/decision_curve_alpha_<a>.png
figures/safe_range<w>.png, beats_trivial<w>.png, beats_local<w>.png
figures/<cost><w>.png   one heat map per term of the decomposition
```

Every table and heat map uses the same orientation, so the reading direction matches
the decomposition: epsilon from `inf` (left) to the tightest budget (right), alpha from
mildest (top) to harshest (bottom). Left to right is increasing privacy, top to bottom
increasing heterogeneity, and **the worst cell is always bottom-right**.

### `cell_images_32/`

The NIH malaria dataset (27,558 images, perfectly balanced: 13,779 `Parasitized` and
13,779 `Uninfected`), pre-resized to 32x32. Resizing once up front instead of on every
access makes the dataset about 9x lighter to read during training. If this folder is
absent the code falls back to a full-resolution `cell_images/` folder.

> ⚠️ The legacy notebook mentions `python resize_dataset.py` as the way
> to generate this folder, but **that script is not in the repository**. The folder is
> already here, so this only matters if you need to rebuild it from the original Kaggle
> download. Likewise `download_dataset.ipynb`, referenced in an error message, is absent.

---

## Setup

Requires Python 3. A GPU is used if available (`cuda:0`), otherwise everything runs
on CPU — slowly. The full sweep is measured in days of compute, not hours.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins no versions. The environment this was developed and run in:
Python 3.14, `flwr` 1.36, `flwr-datasets` 0.6.1, `torch` 2.14, `opacus` 1.6,
`datasets` 4.8, `numpy` 2.5, `pandas` 3.0. The Flower API used here
(`flwr.app`, `flwr.clientapp`, `flwr.serverapp.strategy.FedProx`, `Message`-based
clients) is the modern one and will **not** work on Flower 1.x releases before ~1.13.

`medmnist` is listed in `requirements.txt` but is not imported anywhere in the current
code — a leftover from an earlier dataset.

Note that `.gitignore` covers only `.venv`, `__pycache__` and `.claude`: both
`cell_images_32/` (27,558 files) and `results/` are currently untracked but not
ignored, so `git add -A` would commit them.

**All commands must be run from the repository root**, because every path
(`cell_images_32/`, `results/`) is relative to the working directory.

---

## How to run

### 1. The federated sweep

```bash
source .venv/bin/activate
python -m federated.sweep --help          # every option
python -m federated.sweep --preflight-only # check all alpha x seed partitions, then exit
python -m federated.sweep                  # the full grid
```

The loop is ordered seed → alpha → epsilon, so an interruption leaves whole seeds
finished rather than all of them half-done. `--skip-completed` is the default, so a
relaunch picks up exactly where it stopped; `--max-hours 8` stops cleanly after a time
budget. `--dp-preview` prints the sigma each client will need *before* training, which is
the cheapest way to catch an impossible budget.

For a smoke test, narrow the grid on the command line — no need to edit the config:

```bash
python -m federated.sweep --alphas 10 --seeds 42 --epsilons inf --rounds 10
```

Set `FL_RESULTS_ROOT` to send a test run somewhere other than `results/`, which holds
the real results:

```bash
FL_RESULTS_ROOT=/tmp/fl_smoke python -m federated.sweep --alphas 10 --seeds 42 --rounds 5
```

Fixed-partition repeats, once the main sweep is done. They separate the Dirichlet draw's
variance from DP-SGD's: the same `(alpha, seed, epsilon)` relaunched gives the same
partition and the same test set with different noise, and the output goes to
`alpha_<a>_seed<s>_rep<N>/` without touching the main directories.

```bash
python -m federated.sweep --replicates 4 --alphas 0.3 --seeds 42 --epsilons 0.5 8
```

Then the summary tables:

```bash
python -m federated.report --plots
```

### 2. The baselines

```bash
source .venv/bin/activate
python -m baselines            # local models + one centralised model per seed
python -m baselines local      # only the 360 local models
python -m baselines central    # only the 10 centralised models
python -m baselines summary    # tables from the CSVs, trains nothing
```

Both stages skip whatever is already on disk, so it can be interrupted and relaunched.
`--alphas` and `--seeds` restrict the run (e.g. `python -m baselines --seeds 43 44`),
`--epochs` shortens it for a smoke test only (results are then NOT comparable with the
sweep, and the script says so). **Do not skip this stage for a new seed**: the seed fixes
the 80/20 test split, so every seed needs its own references or `dca.py` discards it.

Expect hours: 360 local models at 100 epochs each, plus 10 centralised models on the full
pooled training set. `local_meta.csv` and `central_meta.csv` record the duration of each.

### 3. The analysis

```bash
source .venv/bin/activate
python dca.py
```

Minutes, not hours — it only reads stored probabilities. It prints the summary tables
and the LaTeX blocks to stdout, and writes everything listed under `results/dca/`
above. If the local baselines are missing it says so and skips the Safe Zone, still
producing the cost decomposition.

From a notebook, for interactive work:

```python
import dca
dca.POSITIVE_LABEL = 0                       # 0 = Parasitized (the clinical event)
fl = dca.fl_nb(window=10)                    # federated NB curves only
A, rng = dca.analyze()                       # + baselines, costs, Safe Range
dca.plot_decision_curves(A)                  # every window, one figure
dca.plot_cost_maps(A, window=(0.01, 0.20))   # one window at a time
dca.plot_safe_range_map(rng, window=(0.01, 0.20))
```

A sensitivity check on the round budget, for which `dca.py` has a deliberate escape
hatch:

```python
A, rng = dca.analyze(reduce="best")   # OPTIMISTIC: picks the best round on the TEST set
```

The default `reduce="window"` averages the last `WINDOW` rounds and is pessimistic when
a run has not converged; `reduce="best"` is optimistic. Together they bracket the
effect. Never quote `"best"` as a headline result.

---

## Reproducibility

Runs are **not** bit-reproducible, and the thesis declares this among its limitations.
`set_seeds()` reseeds the main process only: it does not reseed the Ray actors that run
the simulated clients, nor Opacus' noise generator. Averages over seeds should be read
as estimates with variance.

With three draws the only available statistic was the half-range, which is what `dca.py`
still reports. With ten it becomes mean ± 1.96·SE, the standard quantity — and the
non-reproducibility above is also what makes fixed-partition repeats free: relaunching
the same cell resamples the DP noise while keeping the partition.
