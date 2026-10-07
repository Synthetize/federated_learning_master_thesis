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
malaria_dpsgd.ipynb      [1] the federated sweep  -> results/malaria/
baselines.py             [2] non-federated baselines -> results/baselines/
central_seeds.py         [2b] one centralised run per seed (fixes a seed bug)
dca.py                   [3] the analysis: DCA, costs, Safe Range -> results/dca/
cell_images_32/          the dataset, pre-resized to 32x32
results/                 every artefact produced by the three stages
requirements.txt         dependencies
```

The numbers in brackets are the execution order. Stages are **resumable**: each one
skips whatever it already finds on disk, so a run can be interrupted and relaunched.

---

## What each file does

### `malaria_dpsgd.ipynb` — the federated sweep

The main experiment. 33 cells, organised in numbered sections:

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

### `baselines.py` — the two non-federated references

A plain PyTorch loop: no Flower, no Ray, no Opacus, because neither baseline is
federated and neither is trained under DP.

- **`Local_k`** — one model per client, trained on that client's partition only. This
  is "what one hospital achieves alone", and the opponent the federation has to beat
  to justify its existence.
- **`Centralized`** — one model on all clients' data pooled back together. The upper
  bound the cost decomposition measures against.

It deliberately reuses the sweep's exact partitioner, seeds, retry logic, 80/20 split,
architecture and epoch budget (`EPOCHS = NUM_ROUNDS * LOCAL_EPOCHS = 100`, the same
number of epochs a client consumes inside the federation). If those did not match, the
baselines would be scored on a different test set and the comparison would be
meaningless. Baselines use early stopping on a validation split, which the federated
model does not: they have no DP noise and so no round-to-round instability requiring a
window average.

**Writes** into `results/baselines/`:

```
alpha_<a>/probs_local_s<seed>_c<client>.npy   P(class 1), float32
probs_central.npy                             P(class 1) of the centralised model
test_labels_s<seed>.npy                       raw 0/1 labels
local_meta.csv                                n_train, val acc, chosen epoch, test acc, duration
```

> ⚠️ **Known bug.** `run_central()` is commented out at
> [baselines.py:328](baselines.py#L328) but still called at
> [baselines.py:380](baselines.py#L380), so `python baselines.py` and
> `python baselines.py central` both die with `NameError: name 'run_central' is not
> defined`. Use `python baselines.py local` for the local models and
> `python central_seeds.py` for the centralised ones — which is the better route
> anyway, see below. Uncommenting the block restores the old single-run behaviour.

> ⚠️ `ALPHAS` in `baselines.py` is set to `[0.4]`, while `results/baselines/` already
> holds every alpha from a previous full run. Widen the list if you need to retrain.

### `central_seeds.py` — one centralised run per seed

A fix, and the docstring explains the reasoning. The centralised model is invariant to
the Dirichlet partition — pooling the clients back together reconstructs the same
training pool whatever alpha is — which is why it sits outside the alpha loop. But it
is **not** invariant to the seed, because the seed also fixes the 80/20 train/test
split. Seed 43's test set is a different sample of images, so a model trained on seed
42's pool scored against seed 43's labels collapses to chance (0.4964, against 0.9641
on its own seed).

Only the Utility Cost and the Federation Cost touch the centralised model, so only
those two were affected; Heterogeneity, Privacy, the Interaction and the entire Safe
Zone are FL-minus-FL differences taken inside one draw. One run per seed fixes it and
incidentally gives the Federation Cost an error bar.

Writes `results/baselines/probs_central_s<seed>.npy` and `central_weights_s<seed>.pt`.
The existing `probs_central.npy` is left untouched and reused as the seed-42 run.

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

> ⚠️ Both the notebook and `baselines.py` mention `python resize_dataset.py` as the way
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
jupyter lab malaria_dpsgd.ipynb      # or open it in VS Code
```

Run the cells in order. Sections 7.1–7.3 are diagnostics and are commented out —
leave them that way.

Before launching section 7, set your time budget in section 2:

```python
max_hours: float = 8.0   # the sweep stops cleanly after N hours and resumes on relaunch
```

The loop is ordered seed → alpha → epsilon, so an interruption leaves whole seeds
finished rather than all of them half-done. `skip_completed = True` makes a relaunch
pick up exactly where it stopped. Section 5 prints the sigma each client will need
*before* training, which is the cheapest way to catch an impossible budget.

To shrink the grid for a smoke test, narrow `alphas`, `seeds`, `target_epsilons` and
`num_rounds` in section 2 — **and change the output root** so a test run cannot write
into `results/malaria/`, which holds the real results.

### 2. The baselines

```bash
source .venv/bin/activate
python baselines.py local     # 6 clients x 3 seeds x alphas, 100 epochs each
python central_seeds.py       # one centralised model per seed
```

`python baselines.py` and `python baselines.py central` are currently broken (see the
bug note above) — use these two commands instead. Both skip whatever is already on
disk. `central_seeds.py` accepts an explicit seed list:

```bash
python central_seeds.py 43 44
```

Expect hours: 18 local models at 100 epochs each, plus one centralised model per seed
on the full pooled training set. `local_meta.csv` records the duration of each.

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
as estimates with variance, which is exactly why `dca.py` reports a half-range for
every quantity.
