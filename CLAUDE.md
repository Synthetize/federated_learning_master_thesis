# Master's thesis — Federated Learning stress test

Context for a fresh session in this repo. The reader has no memory of previous chats:
this is the minimum needed not to break anything. Working branch: **v2**.

## What this is about

Thesis: *"Resilience and Critical Failure Points of Federated Learning: A Stress-Test on
Data Heterogeneity and Privacy Noise"* (University of Camerino). It measures how much it
costs, in **decision value** and not in accuracy, to combine data heterogeneity
(Dirichlet alpha) and privacy noise (record-level DP-SGD, epsilon).

- **RQ1** — how much each of the two factors costs and how much they cost together,
  decomposed into Federation / Heterogeneity / Privacy / Interaction, with the identity
  `U = F + H + P + I` exact at every threshold.
- **RQ2** — in which (alpha, epsilon) combinations the federated model **stops being
  worth it**: a two-condition Safe Zone (it beats both the strategies that need no model
  and the average of the local models of the same partition) and the Safe Range as the
  longest contiguous run of safe thresholds.

The metric is the **Net Benefit** of Decision Curve Analysis (Vickers & Elkin 2006),
averaged over the clinical window `p_t in [0.01, 0.20]`. **It is not accuracy, and it
must not be replaced with accuracy**: the whole thesis rests on this choice. The case
that justifies it: alpha=10 with epsilon=8 has accuracy 0.946, the highest among the
noisy configurations, and Safe Range 0.00.

## Stack

FedProx (mu = 0.01) on Flower + Ray in simulation, 6 clients, 100 rounds. Record-level
DP-SGD via Opacus with the RDP accountant. Dataset: malaria cell images (27,558 crops in
`cell_images_32/`), binary, balanced. Network `Net`: 3 conv + GroupNorm, 102,082
parameters, 32x32 input. GroupNorm and not BatchNorm because Opacus rejects BatchNorm.

## Structure

| what | where |
|---|---|
| federated sweep | `federated/` (package: config, paths, model, dataset, privacy, app, sweep, report) |
| local and centralized baselines (one per seed) | `baselines/` (config, paths, data, training, local, central, summary) |
| DCA analysis, figures, LaTeX tables | `dca.py` |
| the replaced notebook, with the diagnostics cited in the thesis | `legacy/` |
| LaTeX chapters | other repo: `master_thesis_overleaf/tesi_unicam_template/chapters/` |

```bash
python -m federated.sweep --help
python -m federated.sweep --preflight-only
python -m federated.sweep
python -m baselines
python -m federated.report --plots
python dca.py
```

## Current status

**Chapter 5 of the thesis is closed** on the 3-seed data (8,630 words, three review
passes, section 5.4 with 15 citations). The ongoing retraining serves to close five
limitations that the chapter declares and cannot resolve, **without changing the
framing**: nothing is to be rewritten at the level of methodology or research questions,
only the numbers.

In progress: sweep with **10 seeds (42-51) x 6 alphas x 6 privacy configurations = 360
runs**, from scratch on a machine with **AMD Radeon RX 9070 XT + Ryzen 7 9700X** (WSL2,
PyTorch ROCm). The 3-seed CPU results were deleted on October 7 and remain in commit
`24166df2`. Under WSL Ray does not detect the AMD GPU: `sweep.py` declares it by hand
with `init_args={"num_gpus": 1}`, otherwise Flower exits with "ActorPool is empty".

## Invariants — breaking these invalidates the thesis

1. **Do not change the hyperparameters.** `max_grad_norm = 5.0`, `momentum = 0.0`,
   `learning_rate = 0.03`, `fedprox_mu = 0.01`, `batch_size = 128`,
   `min_partition_size = 500`, `num_rounds = 100`. Each one is justified in
   `federated/config.py` with a diagnostic, and they are cited in the thesis. Changing
   one makes the runs incomparable with anything that has been written.
2. **A single folder naming convention:** `f"alpha_{alpha:g}"`, so 10.0 -> `alpha_10`
   and 1.0 -> `alpha_1`. It applies to `federated/paths.py` and to `baselines/paths.py`
   (which reuses `alpha_tag`). Previously `alpha_10` and `alpha_10.0` coexisted, which
   `dca.py` both normalizes with `float()` and which therefore collapsed onto the same
   key, overwriting each other.
3. **Three families for every seed.** The 80/20 split of the test set depends on the
   seed, so every new seed needs the FL sweep **plus** `python -m baselines` (local and
   centralized). If one is missing, the DCA drops that seed.
4. **Never one hardware for half the sample.** All 10 seeds must run on the same machine:
   mixing CPU and GPU within the sample would introduce a systematic difference in what
   is being estimated.
5. **Platt only on the pooled validation set**, never on the test set. The pooled
   validation set is `federated.dataset.load_pooled_valset`: the six local 80/20 splits
   concatenated, never part of the centralized test set.

## Traps that have already bitten

- **The centralized model is one per seed, never a single one.** The old `run_central()`
  trained a single model on seed 42, but the test set depends on the seed: evaluated on
  seed 43 it gave chance-level accuracy. The code has been removed (it remains in the
  history, commit `24166df2`).
- **The baselines no longer have a copy of the configuration.** `baselines/config.py`
  reads everything from `federated.config`, and `baselines/data.py` reuses
  `federated.dataset`. Do not reintroduce constants copied by hand: that is the
  misalignment the refactor removes. The refactor was verified bit for bit against the
  old `baselines.py`.
- **Averages of counts are not reported as counts.** A "3.4 clients out of 6" was an
  average over 20 thresholds x 3 draws. If a number is a count, it must be reported as
  an integer or as a declared fraction of comparisons.
- **`set_seeds` does not reseed the Ray actors nor Opacus**, so runs are not
  bit-reproducible. It is not a bug to fix: it is what makes fixed-partition replicates
  (`--replicates`) free, and it is declared among the limitations.
- Test probabilities are saved **round by round** in `probs_eps<eps>.f32` (appended
  float32, `np.fromfile(...).reshape(-1, n_test)`). Choosing a round requires no
  checkpoint.
- **Line endings:** the repo had CRLF files on disk and LF in git, so every change
  showed up as a whole file rewritten. Normalized to LF on October 7, with
  `.gitattributes` (`* text=auto eol=lf`). Do not reintroduce CRLF.

## What remains to do after the sweep

1. Recalibration script: Platt fitted on the pooled validation set, applied to the test
   probabilities, DCA rerun. **Does not require retraining.**
2. `dca.py`: from **half-range** to **mean +- 1.96*SE**, and drop the hand-built 0.02
   readability margin. With 10 seeds they become unnecessary.
3. Early-stopping symmetry: `argmax(val_accuracy)` from the validation stream (column in
   `training_history.csv`), and the corresponding row from the test stream.
4. Fixed-partition replicates: `python -m federated.sweep --replicates 4 --alphas 0.3
   --seeds 42 --epsilons 0.5 8`.
5. Rewriting the numbers in Chapter 5.

## Wider context

The Claude project "Master Thesis" has six handoff documents and 41 papers, but **Claude
Code does not see claude.ai projects**. If the full context is needed, it must be copied
here by hand.
