"""
Decision Curve Analysis from the probabilities saved by the federated sweep.

Standalone module: it does not depend on the notebooks. Use it from a notebook or a script:

    import dca
    dca.POSITIVE_LABEL = 0                  # 0 = Parasitized (the clinical event)
    fl = dca.fl_nb(window=10)               # federated NB curves
    A, rng = dca.analyze()                  # + baselines, cost decomposition, Safe Range
    dca.plot_decision_curves(A)                             # every window, one figure
    dca.plot_cost_maps(A, window=(0.01, 0.10))              # one window at a time
    dca.plot_safe_range_map(rng, window=(0.01, 0.10))

or directly:  python dca.py

INPUT FILE FORMAT
    results/malaria/alpha_<a>_seed<s>/probs_eps<e>.f32
        appended float32, ONE ROW PER ROUND, holding P(label == 1) on the shared test set.
        An incomplete row (a run interrupted mid-write) is discarded.
    results/malaria/alpha_<a>_seed<s>/test_labels.npy
        raw 0/1 labels of the test set. They depend on the seed only.
    results/baselines/alpha_<a>/probs_local_s<s>_c<k>.npy   (optional)
    results/baselines/probs_central.npy                (optional)
    results/baselines/test_labels_s<s>.npy             (optional)

OUTPUT (when run as a script)
    results/dca/full_analysis.csv       one row per (alpha, epsilon, threshold)
    results/dca/safe_range.csv          every window, one row per configuration
    results/dca/costs<w>.csv            the cost decomposition, one file per window
    results/dca/safe_range<w>.tex       table ready to paste into the thesis
    results/dca/costs<w>.tex            cost decomposition, ready to paste
    results/dca/figures/decision_curve_alpha_<a>.png   one figure per alpha
    results/dca/figures/safe_range<w>.png
    results/dca/figures/<cost><w>.png   one heat map per term of the decomposition

    <w> is the reading window: empty for the full threshold grid, "_0p01-0p10" and so on
    for each entry of CLINICAL_RANGES.

COST DECOMPOSITION
    The quantity this module used to call Privacy Cost -- the gap between the centralised
    model and a federated cell -- is not a cost of privacy: it also contains the cost of
    having split the data at all, and the cost of the split being uneven. It is non-zero
    even in the column where DP is switched off. It is therefore reported as Utility Cost,
    and split into terms that isolate one effect each. Writing f(a, e) for the federated Net
    Benefit, c for the centralised one, a0 for the mildest alpha in the grid and e0 =
    infinity for the run without DP:

        Utility Cost       U(a,e)  = c         - f(a,e)     everything at once
        Federation Cost    F       = c         - f(a0,e0)   splitting the data, nothing else
        Heterogeneity Cost H(a)    = f(a0,e0)  - f(a,e0)    the skew on its own, DP off
        Privacy Cost       P(e)    = f(a0,e0)  - f(a0,e)    the budget on its own, at a0
        Interaction        I(a,e)  = U - F - H(a) - P(e)    what the two do together

    and U = F + H + P + I holds exactly, at every threshold. Nothing is discarded: the
    decomposition only says where the total loss comes from.

    Each axis is also read at the operating point actually in use, which is the number one
    quotes in prose rather than in the table:

        Privacy Cost at a        f(a,e0) - f(a,e)     what the budget costs at THAT alpha
        Heterogeneity Cost at e  f(a0,e) - f(a,e)     what the skew costs under THAT budget

    Each of these equals its main effect plus the interaction, so they cannot be added to
    the main effects without counting the interaction twice. Charging the interaction to
    one axis by measuring that axis at the operating point and the other at the reference
    would be a choice with nothing to justify it, which is why the interaction gets a term
    of its own instead.

    The interaction is symmetric: the extra privacy cost caused by heterogeneity and the
    extra heterogeneity cost caused by privacy are the same number. It is zero on the
    reference row and in the reference column by construction. If it is flat, the two axes
    are independent and the stress test has found nothing; if it grows towards the
    bottom-right corner, they amplify each other.

GRID ORIENTATION
    Every table and heat map is laid out the same way, so that the reading direction always
    matches the decomposition: epsilon runs from infinity (no DP, leftmost) to the tightest
    budget on the right, alpha from the mildest value at the top to the harshest at the
    bottom. Left to right is therefore increasing privacy, top to bottom increasing
    heterogeneity, and the worst cell is always the bottom-right one.

Probabilities are stored RAW, so the threshold grid, the event class and the averaging
window are all post-processing choices: changing them requires no run to be repeated.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- configuration
FL_DIR = os.path.join("results", "malaria")      # input: sweep probabilities
BASE_DIR = os.path.join("results", "baselines")  # input: local and centralised baselines
OUT_DIR = os.path.join("results", "dca")         # output: everything the analysis produces
FIG_DIR = os.path.join(OUT_DIR, "figures")

# Which class index is the EVENT one decides to act upon.
# imagefolder assigns indices alphabetically: Parasitized=0, Uninfected=1.
POSITIVE_LABEL = 0

# Threshold grid. Change it freely: nothing has to be recomputed upstream.
# No threshold may equal 1.0 (p_t/(1-p_t) would diverge).
THRESHOLDS = np.round(np.arange(0.01, 0.96, 0.01), 2)

WINDOW = 10        # W: final rounds to average over, as in the Methodology

# Clinically plausible threshold windows, over which Safe Range, failure point and the cost
# decomposition are REPORTED. The computation still runs over the whole of THRESHOLDS: these
# are only reading windows.
#
# Vickers and Elkin prescribe no universal range: they state it has to be derived from the
# clinical context ("we need to consider the likely range of p_t in the population"), and
# their own examples use narrow windows that differ from one another (30-60% for hormone
# therapy in prostate cancer, 1-10% for seminal vesicle dissection).
#
# For malaria screening a false negative is far more serious than a false positive: at
# p_t = 0.20 one FN is worth 4 FP, at p_t = 0.01 it is worth 99. Above 0.20 one would be
# assuming that a false alarm is almost as serious as a missed infection, which does not
# hold.
# More than one window may be listed. Every table, every CSV and every heat map is then
# produced once per window, and the decision curves carry one shaded band per window in the
# same figure rather than being duplicated. Leave a single pair in the list to go back to
# one window; the readings over the full threshold grid are always produced as well.
#
# Two windows are listed by default because the upper bound is the arguable part. 0.01-0.10
# is the order of magnitude of the seminal vesicle example, which is likewise a case where
# the false negative dominates; 0.01-0.20 is more permissive. If a conclusion holds under
# both, the choice of bound stops being an objection.
CLINICAL_RANGES = [(0.01, 0.20)]

# How a curve is reduced to one number over a window of thresholds, for the cost tables and
# heat maps. The mean is the default because the three terms of the decomposition then add
# up exactly in the reported table, as they do at every individual threshold: the mean of a
# sum is the sum of the means, the median is not. Switch to "median" for a robustness check.
COST_STAT = "mean"

# Minimum number of CONSECUTIVE safe thresholds for a configuration to count as having a
# Safe Range. A "range" that is one isolated threshold is not a range: with a window of 20
# thresholds and a seed half-range of the order of 0.01 in Net Benefit, a curve that pokes
# above a reference at a single point and sits below it everywhere else has not shown
# anything. Without this rule the map reports widths of 0.05 and 0.10 that are smaller than
# the noise between draws, and orders configurations that are in fact all at zero.
MIN_SAFE_RUN = 3

# Upper end of the x axis on the decision curves. None draws the whole threshold grid.
# Setting it ZOOMS THE FIGURE ONLY: every number is still computed over the whole of
# THRESHOLDS, so no reading changes. Shortening THRESHOLDS itself would be a different
# thing entirely -- it would redefine the "all thresholds" reading, and a cut placed where
# the curves look worst is the first thing a reader challenges.
CURVE_XMAX = 0.6



# --------------------------------------------------------------------------- windows
def windows():
    """Every reading window, the full threshold grid first."""
    return [None] + list(CLINICAL_RANGES)


def win_label(window):
    """How a window is named in printed output and in a table caption."""
    return "all thresholds" if window is None else f"{window[0]:g}-{window[1]:g}"


def win_tag(window):
    """How a window is spelled in a file name: empty for the full grid."""
    return "" if window is None else f"_{window[0]:g}-{window[1]:g}".replace(".", "p")


def win_slice(A, window):
    return A if window is None else A[(A["p_t"] >= window[0]) & (A["p_t"] <= window[1])]


# --------------------------------------------------------------------------- DCA primitives
def to_event(y_raw, p1, positive_label=None):
    """From (raw labels, P(class 1)) to (boolean event, P(event))."""
    pl = POSITIVE_LABEL if positive_label is None else positive_label
    y = np.asarray(y_raw) == pl
    p = np.asarray(p1, dtype=float)
    return y, (p if pl == 1 else 1.0 - p)


def net_benefit(y_event, p_event, thresholds=None):
    """NB(p_t) = TP/n - (FP/n) * (p_t / (1 - p_t)).

    A case is predicted "treat" when p_event >= p_t. Vectorised implementation: the
    probabilities are sorted once, and TP and FP at every threshold are recovered with a
    binary search, instead of rescanning the array for each threshold.
    """
    th = THRESHOLDS if thresholds is None else np.asarray(thresholds, dtype=float)
    y = np.asarray(y_event, dtype=bool)
    p = np.asarray(p_event, dtype=float)
    n = len(y)
    order = np.argsort(p, kind="stable")
    ps = p[order]
    ys = y[order]
    # how many samples have p >= t  ->  n - searchsorted(ps, t, "left")
    idx = np.searchsorted(ps, th, side="left")
    cum_pos = np.concatenate([[0], np.cumsum(ys)])          # positives among the first k
    tp = ys.sum() - cum_pos[idx]
    fp = (n - idx) - tp
    return tp / n - (fp / n) * (th / (1.0 - th))


def nb_from_probs(y_raw, p1, thresholds=None, positive_label=None):
    y, p = to_event(y_raw, p1, positive_label)
    return net_benefit(y, p, thresholds)


def treat_all(y_raw, thresholds=None, positive_label=None):
    """The trivial 'treat all' strategy. 'Treat none' is 0 at every threshold."""
    th = THRESHOLDS if thresholds is None else np.asarray(thresholds, dtype=float)
    y, _ = to_event(y_raw, np.zeros(len(y_raw)), positive_label)
    n, npos, nneg = len(y), int(y.sum()), int((~y).sum())
    return npos / n - (nneg / n) * (th / (1.0 - th))


# --------------------------------------------------------------------------- sweep reading
_DIR_RE = re.compile(r"alpha_([0-9.]+)_seed(\d+)$")


def _read_probs(path, n_test):
    """Matrix (rounds, n_test). Discards a trailing incomplete row, if any."""
    raw = np.fromfile(path, dtype=np.float32)
    n_rounds, remainder = divmod(len(raw), n_test)
    if remainder:
        raw = raw[: n_rounds * n_test]
    return raw.reshape(n_rounds, n_test) if n_rounds else np.empty((0, n_test), np.float32)


def load_runs(fl_dir=None):
    """Every run found: a list of dicts with alpha, seed, epsilon, labels, probs."""
    fl_dir = FL_DIR if fl_dir is None else fl_dir
    runs = []
    for d in sorted(glob.glob(os.path.join(fl_dir, "alpha_*_seed*"))):
        m = _DIR_RE.search(os.path.basename(d))
        if not m:
            continue
        alpha, seed = float(m.group(1)), int(m.group(2))
        lab = os.path.join(d, "test_labels.npy")
        if not os.path.exists(lab):
            print(f"  skipping {os.path.basename(d)}: test_labels.npy is missing")
            continue
        y_raw = np.load(lab)
        for f in sorted(glob.glob(os.path.join(d, "probs_eps*.f32"))):
            tag = os.path.basename(f)[len("probs_eps"):-len(".f32")]
            eps = np.inf if tag == "inf" else float(tag)
            probs = _read_probs(f, len(y_raw))
            if len(probs) == 0:
                continue
            runs.append({"alpha": alpha, "seed": seed, "epsilon": eps,
                         "labels": y_raw, "probs": probs, "n_rounds": len(probs)})
    if not runs:
        raise FileNotFoundError(f"no run found under {fl_dir}")
    return runs


def fl_nb(window=None, reduce="window", thresholds=None, fl_dir=None, verbose=True,
          by_seed=False):
    """NB curves of the federated model, averaged across draws.

    `by_seed=True` skips the average and returns one row per (alpha, epsilon, seed, p_t).
    The decomposition is then computed separately within each seed, which is what makes a
    spread across seeds meaningful: every cost is a difference between two cells, and taking
    both cells from the same draw cancels the part of the variation the two share.

    reduce="window"  averages over the last `window` rounds (Methodology, Eq. fl-window-avg).
    reduce="best"    takes the round with the highest mean NB. WARNING: it picks the round by
                     looking at the TEST SET, so it is an optimistic estimate -- use it only
                     as a sensitivity check on the round budget, never as the headline
                     result. The "window" reading is pessimistic when a run has not
                     converged, this one is optimistic: together they bracket the effect.
    """
    window = WINDOW if window is None else window
    th = THRESHOLDS if thresholds is None else np.asarray(thresholds, dtype=float)
    rows = []
    for r in load_runs(fl_dir):
        curves = np.array([nb_from_probs(r["labels"], p, th) for p in r["probs"]])
        if reduce == "window":
            if r["n_rounds"] < window and verbose:
                print(f"  alpha={r['alpha']} seed={r['seed']} eps={r['epsilon']:g}: "
                      f"only {r['n_rounds']} rounds (window {window})")
            nb = curves[-window:].mean(axis=0)
            extra = {"selected_round": np.nan}
        elif reduce == "best":
            k = int(np.argmax(curves.mean(axis=1)))
            nb = curves[k]
            extra = {"selected_round": k + 1}
        else:
            raise ValueError("reduce must be 'window' or 'best'")
        rows += [{"alpha": r["alpha"], "epsilon": r["epsilon"], "seed": r["seed"],
                  "p_t": t, "nb": v, "n_rounds": r["n_rounds"], **extra}
                 for t, v in zip(th, nb)]
    d = pd.DataFrame(rows)
    rounds = (d.groupby(["alpha", "epsilon"])["n_rounds"].min()
              .reset_index(name="min_rounds"))
    if by_seed:
        return (d.rename(columns={"nb": "nb_fl"})
                [["alpha", "epsilon", "seed", "p_t", "nb_fl"]]
                .merge(rounds, on=["alpha", "epsilon"]))
    out = d.groupby(["alpha", "epsilon", "p_t"])["nb"].mean().reset_index(name="nb_fl")
    return out.merge(rounds, on=["alpha", "epsilon"])


# --------------------------------------------------------------------------- baselines
def _local_files(base_dir):
    """(alpha, seed, client, path) of the local models, under either of two layouts.

    New:    results/baselines/alpha_<a>/probs_local_s<seed>_c<client>.npy
    Legacy: results/baselines/probs_local_a<a>_s<seed>_c<client>.npy   (flat layout)

    If the same combination exists under both, the new one wins, so that a half-finished
    migration does not produce duplicates in the average.
    """
    found = {}
    for f in sorted(glob.glob(os.path.join(base_dir, "alpha_*", "probs_local_s*_c*.npy"))):
        alpha = float(os.path.basename(os.path.dirname(f)).split("_", 1)[1])
        tag = os.path.basename(f)[len("probs_local_s"):-len(".npy")]
        seed, client = tag.split("_c")
        found[(alpha, int(seed), int(client))] = f
    for f in sorted(glob.glob(os.path.join(base_dir, "probs_local_a*_s*_c*.npy"))):
        tag = os.path.basename(f)[len("probs_local_a"):-len(".npy")]
        a_s, client = tag.rsplit("_c", 1)
        alpha, seed = a_s.split("_s")
        found.setdefault((float(alpha), int(seed), int(client)), f)
    return [(a, s, c, p) for (a, s, c), p in sorted(found.items())]


def baseline_nb(thresholds=None, base_dir=None):
    """Baseline curves. Returns (local_per_alpha, centralised), or (None, None) if the files
    are not there yet, so that the federated analysis can be run straight away."""
    base_dir = BASE_DIR if base_dir is None else base_dir
    th = THRESHOLDS if thresholds is None else np.asarray(thresholds, dtype=float)

    rows = []
    for a, s, c, f in _local_files(base_dir):
        lab = os.path.join(base_dir, f"test_labels_s{s}.npy")
        if not os.path.exists(lab):
            continue
        nb = nb_from_probs(np.load(lab), np.load(f), th)
        rows += [{"alpha": a, "seed": s, "client": c, "p_t": t, "nb": v}
                 for t, v in zip(th, nb)]
    local = None
    if rows:
        d = pd.DataFrame(rows)
        # The seed is KEPT here. Averaging it away at this point would hand every draw the
        # same local baseline, which does two things wrong: it reports a half-range of zero
        # for a quantity that varies by up to 0.08 between draws at low alpha, and it pairs
        # the federated model of one partition against the local models of all of them. The
        # grand mean is unchanged either way -- every (alpha, seed) cell holds all clients --
        # so this adds the spread without moving a single reported value.
        local = (d.groupby(["alpha", "seed", "p_t"])["nb"].mean()
                 .reset_index(name="nb_local_avg"))

    # "treat all" needs the labels only, so it is built whether or not a centralised model
    # has been trained: the comparison against the strategies that require no model at all is
    # the first test a model has to pass, and it must not depend on an optional file.
    # Both references are built PER SEED. The centralised model is invariant to the Dirichlet
    # partition, which is why it sits outside the alpha loop, but it is NOT invariant to the
    # seed: the seed also fixes the 80/20 train/test split, so each draw has its own test set
    # and a model trained on one draw's pool cannot be scored on another's. Reusing a single
    # run for every seed would make the Utility Cost and the Federation Cost differences
    # between quantities measured on different samples of images, so the fallback below
    # announces itself instead of passing silently.
    central = None
    labs = sorted(glob.glob(os.path.join(base_dir, "test_labels_s*.npy")))
    if labs:
        legacy_path = os.path.join(base_dir, "probs_central.npy")
        legacy = np.load(legacy_path) if os.path.exists(legacy_path) else None

        # Which seed does the un-suffixed file belong to? Ask the data rather than assuming:
        # scored against the wrong draw's labels it collapses to chance, so the best accuracy
        # identifies its own seed unambiguously.
        legacy_seed = None
        if legacy is not None:
            best = -1.0
            for f in labs:
                y = np.load(f)
                if len(y) != len(legacy):
                    continue
                acc = float(((legacy > 0.5).astype(int) == y).mean())
                if acc > best:
                    best, legacy_seed = acc, int(re.search(r"_s(\d+)\.npy$", f).group(1))

        frames, borrowed = [], []
        for f in labs:
            seed = int(re.search(r"_s(\d+)\.npy$", f).group(1))
            y_raw = np.load(f)
            row = pd.DataFrame({"seed": seed, "p_t": th,
                                "nb_treat_all": treat_all(y_raw, th)})
            pc = os.path.join(base_dir, f"probs_central_s{seed}.npy")
            if os.path.exists(pc):
                row["nb_central"] = nb_from_probs(y_raw, np.load(pc), th)
            elif legacy is not None and seed == legacy_seed:
                row["nb_central"] = nb_from_probs(y_raw, legacy, th)
            elif legacy is not None:
                row["nb_central"] = nb_from_probs(np.load(labs[0]), legacy, th)
                borrowed.append(seed)
            frames.append(row)
        central = pd.concat(frames, ignore_index=True)

        if borrowed:
            print(f"  WARNING: no centralised run for seed(s) {borrowed}; reusing the one "
                  f"trained on seed {legacy_seed}.\n           The Utility Cost and the "
                  "Federation Cost are then differences across test sets for those seeds "
                  "(every other term is\n           a within-seed difference and is "
                  "unaffected). Run python -m baselines central to remove this.")
    return local, central


# --------------------------------------------------------------------------- analysis
def add_costs(A, keys=(), verbose=True):
    """Adds the cost decomposition described in the module docstring.

    Every term is a difference between two cells of the same grid, chosen so that exactly
    one thing changes between them: only the budget for the privacy cost, only alpha for the
    heterogeneity cost. The reference cell is (a0, infinity): the mildest alpha in the grid
    with DP switched off, i.e. the least damaged federated run available.

    Each axis is read twice, and the two readings are kept apart because they answer
    different questions:

        `_main`   the effect measured against the reference cell, with the other axis held
                  at its reference value. This is the effect of that axis on its own, and
                  these are the terms that enter the identity.
        (plain)   the effect measured at the operating point actually in use: the privacy
                  cost at that alpha, the heterogeneity cost at that budget. This is the
                  number one quotes in prose, and it contains the interaction.

    Measuring one axis at the operating point and the other at the reference would charge
    the whole interaction to the first axis, which is a choice with nothing to justify it.
    Keeping both readings, and giving the interaction a term of its own, avoids having to
    make it.

    `keys` adds further columns to every reference join, so that the whole decomposition can
    be computed within each seed separately rather than on the seed average.

    Requires the column with epsilon = infinity to be present; without it a privacy cost
    cannot be isolated from anything, and only the Utility Cost is added.
    """
    keys = list(keys)
    a0 = A["alpha"].max()
    no_dp = ~np.isfinite(A["epsilon"])

    if "nb_central" in A:
        A["utility_cost"] = A["nb_central"] - A["nb_fl"]

    if not no_dp.any():
        if verbose:
            print("\nno run with epsilon = infinity: the cost decomposition cannot be "
                  "computed.\nRun the sweep with DP disabled at least once per alpha.")
        return A

    # f(alpha, infinity): same row, DP off -> isolates the budget
    on_e = ["alpha", "p_t"] + keys
    ref_e = A[no_dp][on_e + ["nb_fl"]].rename(columns={"nb_fl": "nb_fl_nodp"})
    # f(a0, epsilon): same column, mildest alpha -> isolates the skew
    on_a = ["epsilon", "p_t"] + keys
    ref_a = A[A["alpha"] == a0][on_a + ["nb_fl"]].rename(columns={"nb_fl": "nb_fl_a0"})
    # f(a0, infinity): the reference cell itself
    on_0 = ["p_t"] + keys
    ref_0 = (A[no_dp & (A["alpha"] == a0)][on_0 + ["nb_fl"]]
             .rename(columns={"nb_fl": "nb_fl_ref"}))

    A = A.merge(ref_e, on=on_e, how="left")
    A = A.merge(ref_a, on=on_a, how="left")
    A = A.merge(ref_0, on=on_0, how="left")

    # main effects: one axis moves, the other stays at its reference value
    A["privacy_cost_main"] = A["nb_fl_ref"] - A["nb_fl_a0"]        # depends on epsilon only
    A["heterogeneity_cost_main"] = A["nb_fl_ref"] - A["nb_fl_nodp"]  # depends on alpha only
    # readings at the operating point actually in use: these carry the interaction
    A["privacy_cost"] = A["nb_fl_nodp"] - A["nb_fl"]
    A["heterogeneity_cost"] = A["nb_fl_a0"] - A["nb_fl"]
    A["interaction_cost"] = A["privacy_cost"] - A["privacy_cost_main"]

    # The interaction has to come out the same whether it is read as the extra privacy cost
    # caused by heterogeneity or as the extra heterogeneity cost caused by privacy. It does,
    # algebraically; checking it here catches a mis-joined reference rather than bad data.
    alt = A["heterogeneity_cost"] - A["heterogeneity_cost_main"]
    gap = float(np.nanmax(np.abs(A["interaction_cost"] - alt))) if len(A) else 0.0
    if gap > 1e-9 and verbose:
        print(f"\nWARNING: the interaction is not symmetric (gap {gap:.2e}). One of the "
              "reference\nrows or columns did not join correctly.")

    if "nb_central" in A:
        A["federation_cost"] = A["nb_central"] - A["nb_fl_ref"]
        # U = F + H_main + P_main + I must hold exactly; a mismatch means the grid has holes
        resid = (A["utility_cost"] - A["federation_cost"]
                 - A["heterogeneity_cost_main"] - A["privacy_cost_main"]
                 - A["interaction_cost"])
        worst = float(np.nanmax(np.abs(resid))) if len(resid) else 0.0
        if worst > 1e-9 and verbose:
            print(f"\nWARNING: the decomposition does not close (residual {worst:.2e}). "
                  "Some cell of\nthe grid is missing; check that every alpha was run at "
                  "every epsilon.")
    A["reference_alpha"] = a0
    return A


def analyze(window=None, reduce="window", thresholds=None, verbose=True):
    """Joins federated and baseline curves and computes the costs, Safe Zone, Safe Range.

    Returns (curves, safe_range). `curves` has one row per (alpha, epsilon, p_t);
    `safe_range` one row per configuration, with a boolean failure_point.
    """
    S = fl_nb(window=window, reduce=reduce, thresholds=thresholds, verbose=verbose,
              by_seed=True)
    local, central = baseline_nb(thresholds=thresholds)
    if local is not None:
        on = ["alpha", "seed", "p_t"] if "seed" in local and "seed" in S else ["alpha", "p_t"]
        S = S.merge(local, on=on, how="left")
    if central is not None:
        on = ["seed", "p_t"] if "seed" in central and "seed" in S else ["p_t"]
        S = S.merge(central, on=on, how="left")

    # The decomposition is computed inside each seed and only then averaged. Every cost is a
    # linear difference of Net Benefits, so the average of the per-seed costs is exactly the
    # cost of the averaged curves -- the numbers do not change. What this buys is the spread:
    # with both cells of a difference taken from the same draw, the disagreement left between
    # seeds is the disagreement about the COST, not about how good that draw happened to be.
    # The two MARGINS are computed per seed as well, for the same reason as the costs: a
    # margin is a linear difference, so averaging it afterwards gives the same number, but
    # taking both sides from the same draw makes the spread a disagreement about the MARGIN
    # rather than about how good that draw happened to be. This matters most for the local
    # comparison, whose reference varies by up to 0.08 between draws at low alpha. The two
    # VERDICTS stay on the averaged curves below: a boolean averaged over seeds would silently
    # turn into "the fraction of draws that passed", which is a different quantity.
    if "nb_treat_all" in S:
        S["trivial_margin"] = S["nb_fl"] - np.maximum(S["nb_treat_all"], 0.0)
    if "nb_local_avg" in S:
        S["collaboration_gain"] = S["nb_fl"] - S["nb_local_avg"]

    S = add_costs(S, keys=["seed"], verbose=verbose)
    A, spread = _collapse_seeds(S)

    # Two conditions, in the order in which they have to be passed.
    #
    # The first asks whether the model is worth using at all: it has to beat both strategies
    # that need no model, treating everyone and treating no one. This is the comparison DCA
    # supplies by construction, and it is the one that catches a model that is "accurate but
    # useless".
    #
    # The second asks whether the FEDERATION was worth building: the shared model has to beat
    # what an average client already achieves alone.
    #
    # They are kept as separate columns and combined into `safe`, because either one alone is
    # misleading. A configuration can beat the local baseline while being worse than treating
    # everyone -- which happens exactly where heterogeneity has ruined the local models too,
    # so the federated model wins a comparison against an opponent that is already on the
    # floor. And a configuration can beat the trivial strategies while losing to the local
    # baseline, which means the model works but collaborating added nothing.
    # Each condition also gets its MAGNITUDE, not only its verdict: a test passed by 0.0005
    # and one passed by 0.05 are both "safe", and the difference is worth seeing.
    if "nb_treat_all" in A:
        A["beats_trivial"] = A["nb_fl"] > np.maximum(A["nb_treat_all"], 0.0)
        if "trivial_margin" not in A:
            A["trivial_margin"] = A["nb_fl"] - np.maximum(A["nb_treat_all"], 0.0)
    if "nb_local_avg" in A:
        A["beats_local"] = A["nb_fl"] > A["nb_local_avg"]
        # positive = collaborating pays. This is the benefit side of the ledger, and the
        # mirror image of the Federation Cost: that one measures what is lost against
        # pooling everything, this one what is gained against not collaborating at all.
        if "collaboration_gain" not in A:
            A["collaboration_gain"] = A["nb_fl"] - A["nb_local_avg"]

    if "beats_trivial" in A and "beats_local" in A:
        A["safe"] = A["beats_trivial"] & A["beats_local"]
    elif verbose:
        missing = ("the test labels" if "beats_trivial" not in A
                   else "the local baselines")
        print(f"\n{missing} are missing: the Safe Zone and the failure point cannot be "
              "computed.\nRun first:  python -m baselines")

    if "safe" not in A:
        return A, None

    rng = pd.concat([_safe_range(A, w).assign(window=win_label(w)) for w in windows()],
                    ignore_index=True)
    return A, rng


SPREAD_SUFFIX = "_hr"     # half-range across seeds, appended to the column it belongs to


def _collapse_seeds(S):
    """Averages the per-seed frame and attaches a half-range for every quantity.

    The half-range is (max - min) / 2 across seeds, which with two seeds is simply how far
    each one sits from their average. It is reported rather than a standard deviation
    because with two draws a standard deviation is a half-range wearing a lab coat: it
    carries no extra information and invites a reader to treat it as a confidence interval,
    which two runs cannot support.

    Returns (averaged frame, number of seeds).
    """
    keep = [c for c in S.columns if c not in ("alpha", "epsilon", "seed", "p_t")]
    num = [c for c in keep if pd.api.types.is_numeric_dtype(S[c])]
    g = S.groupby(["alpha", "epsilon", "p_t"], as_index=False)
    A = g[num].mean()
    n_seeds = int(S["seed"].nunique())
    if n_seeds > 1:
        hr = g[num].agg(lambda v: (v.max() - v.min()) / 2.0)
        hr = hr.rename(columns={c: c + SPREAD_SUFFIX for c in num})
        A = A.merge(hr, on=["alpha", "epsilon", "p_t"])
    A["n_seeds"] = n_seeds
    return A, n_seeds


def seed_spread(A, cost="privacy_cost", window=None):
    """Typical disagreement between seeds about one cost, over a window of thresholds.

    Returns (per-cell map, typical value). The typical value is the median across cells,
    and is the resolution of the grid: two cells that differ by less than about twice it
    are not telling you they are different, whatever the heat map looks like.
    """
    col = cost + SPREAD_SUFFIX
    if col not in A:
        return None, np.nan
    piv = cost_map(A, col, window)
    return piv, float(np.nanmedian(piv.values))


def _longest_run(flags):
    """Length and index bounds of the longest run of consecutive True values."""
    best = cur = 0
    best_end = -1
    for k, f in enumerate(flags):
        cur = cur + 1 if f else 0
        if cur > best:
            best, best_end = cur, k
    return (0, -1, -1) if best == 0 else (best, best_end - best + 1, best_end)


def _safe_range(A, window):
    """Safe Range and failure point, computed over a window of thresholds.

    The Safe Range is the LONGEST CONTIGUOUS run of thresholds at which the federated model
    beats both references, and a run shorter than MIN_SAFE_RUN is discarded as a failure
    point. Requiring contiguity is what makes the quantity a range rather than a count: a
    model that wins at scattered single thresholds is not usable at any operating point a
    clinician could commit to in advance.

    `window=None` uses every available threshold; a (lo, hi) tuple restricts the reading to
    a clinical window. Reporting every window guards against the most obvious objection in a
    viva -- "did you choose the range after seeing the results?" -- and makes the interesting
    case explicit: a configuration that beats the local baseline only at thresholds outside
    the clinically plausible range is, in practice, a failure point.
    """
    sub = win_slice(A, window)
    ths = np.sort(sub["p_t"].unique())
    n_th = max(len(ths), 1)
    cells = A[["alpha", "epsilon"]].drop_duplicates()

    rows = []
    for (alpha, eps), g in sub.groupby(["alpha", "epsilon"], dropna=False):
        flags = (g.set_index("p_t").reindex(ths)["safe"]
                  .fillna(False).to_numpy(dtype=bool))
        n, i0, i1 = _longest_run(flags)
        kept = n >= MIN_SAFE_RUN
        rows.append({"alpha": alpha, "epsilon": eps,
                     "n_thresholds": n if kept else 0,
                     "p_t_min": ths[i0] if kept else np.nan,
                     "p_t_max": ths[i1] if kept else np.nan,
                     "n_safe_total": int(flags.sum()),
                     "longest_run": n})

    r = cells.merge(pd.DataFrame(rows), on=["alpha", "epsilon"], how="left")
    r["n_thresholds"] = r["n_thresholds"].fillna(0).astype(int)
    r["longest_run"] = r["longest_run"].fillna(0).astype(int)
    r["n_safe_total"] = r["n_safe_total"].fillna(0).astype(int)
    r["width"] = r["n_thresholds"] / n_th
    r["failure_point"] = r["n_thresholds"] == 0

    # the two conditions on their own, so the results chapter can say WHICH one failed.
    # These stay plain fractions of the window, not runs: they are diagnostic, and the
    # question they answer is "how much of the window does this condition cover", which
    # contiguity would obscure.
    for cond, name in (("beats_trivial", "width_trivial"), ("beats_local", "width_local")):
        if cond in sub:
            w = (sub.groupby(["alpha", "epsilon"])[cond].sum() / n_th).reset_index(name=name)
            r = r.merge(w, on=["alpha", "epsilon"], how="left")
    return r.sort_values(["alpha", "epsilon"], ascending=[False, False]).reset_index(drop=True)


def report(window=None, reduce="window"):
    """Prints the summary and returns (curves, safe_range)."""
    A, rng = analyze(window=window, reduce=reduce)
    print(f"\nreading: reduce='{reduce}'"
          + (f", window of {WINDOW if window is None else window} rounds"
             if reduce == "window"
             else " (best round on the test set: OPTIMISTIC estimate)"))
    if rng is None:
        print("\nFederated NB, median over thresholds:")
        print(_orient(A.groupby(["alpha", "epsilon"])["nb_fl"].median()
                      .reset_index()
                      .pivot(index="alpha", columns="epsilon", values="nb_fl"))
              .round(4).to_string())
        for w in windows():
            print_costs(A, w)
        return A, None

    print("\nSAFE RANGE")
    print("  A configuration is safe at a threshold when it beats BOTH the trivial "
          "strategies\n  (treat all / treat none) and the average local baseline. "
          "width_trivial and\n  width_local show the two conditions separately, so a "
          "failure can be attributed.")
    print(f"  The Safe Range is the longest CONTIGUOUS run of safe thresholds; a run of "
          f"fewer\n  than {MIN_SAFE_RUN} is reported as a failure point. longest_run is the "
          "raw run before that\n  rule, n_safe_total the count of safe thresholds however "
          "scattered.")
    cols = [c for c in ("alpha", "epsilon", "p_t_min", "p_t_max", "width",
                        "longest_run", "n_safe_total",
                        "width_trivial", "width_local", "failure_point") if c in rng]
    for w in windows():
        lab = win_label(w)
        sub = rng[rng["window"] == lab]
        print(f"\n  window {lab}:")
        print("    " + sub[cols].to_string(index=False).replace("\n", "\n    "))
        print(f"    failure points: {int(sub.failure_point.sum())} of {len(sub)}")
        if {"width_trivial", "width_local"} <= set(sub.columns):
            only_local = sub[(sub.width_local > 0.5) & (sub.width_trivial < 0.2)]
            if len(only_local):
                print(f"    {len(only_local)} configurations beat the local baseline over "
                      "most of the window\n    while being WORSE than treating everyone. "
                      "They are failure points that a\n    local-baseline-only reading "
                      "would have passed:")
                print("      " + only_local[["alpha", "epsilon", "width_trivial",
                                             "width_local"]]
                      .to_string(index=False).replace("\n", "\n      "))

    _compare_windows(rng)
    for w in windows():
        print_costs(A, w)
    return A, rng


def _compare_windows(rng):
    """Flags the configurations whose verdict depends on which window one reads.

    A configuration that beats the local baseline only at thresholds outside the clinically
    plausible range is, in practice, a failure point even though the full grid does not
    record it as one. Those are the cells the results chapter has to discuss rather than
    average away.
    """
    piv = rng.pivot(index=["alpha", "epsilon"], columns="window",
                    values="failure_point")
    disagree = piv[piv.nunique(axis=1) > 1]
    if disagree.empty:
        print("\nEvery window AGREES on every configuration: the failure point does not "
              "depend\non where the clinical range is drawn.")
        return
    print(f"\nWARNING: {len(disagree)} configurations change verdict between windows.")
    print("    " + disagree.to_string().replace("\n", "\n    "))
    print("These are configurations whose usefulness rests on thresholds a clinician would\n"
          "not use. Discuss them in the results chapter rather than reporting only the\n"
          "window that flatters them.")


def print_costs(A, window=None):
    """Prints the cost decomposition, reduced over the given threshold window."""
    if "utility_cost" not in A:
        return
    print(f"\nCOST DECOMPOSITION, {COST_STAT} over {win_label(window)}")
    print("  epsilon runs from infinity (no DP) on the left to the tightest budget on "
          "the right,\n  alpha from the mildest value at the top to the harshest at the "
          "bottom.")
    if "federation_cost" in A:
        f = float(win_slice(A, window)["federation_cost"].agg(COST_STAT))
        print(f"\n  Federation Cost (one number, the whole grid shares it): {f:+.4f}")
    for key in ("utility_cost", "heterogeneity_cost", "privacy_cost",
                "interaction_cost", "collaboration_gain", "trivial_margin"):
        if key in A:
            print(f"\n  {_plain(COST_TITLES[key])}")
            print(cost_map(A, key, window).round(4).to_string())
    amp = amplification(A, window)
    if amp is not None:
        print("\n  Amplification: privacy cost at that alpha, as a multiple of the privacy "
              "cost\n  at the mildest alpha. 1.00 = the two axes do not interact.")
        print(amp.round(2).to_string())

    n_seeds = int(A["n_seeds"].iloc[0]) if "n_seeds" in A else 1
    if n_seeds < 2:
        print("\n  Only one seed: nothing can be said about how much of the variation "
              "across the\n  grid is real. Run a second seed before reading small "
              "differences.")
        return
    print(f"\n  RESOLUTION OF THE GRID  ({n_seeds} seeds)")
    for key in ("privacy_cost", "heterogeneity_cost", "utility_cost"):
        _, typ = seed_spread(A, key, window)
        if np.isfinite(typ):
            print(f"    {_plain(COST_TITLES.get(key, key)):28s} half-range {typ:.4f}   "
                  f"differences below ~{2 * typ:.4f} are not readable")
    print("    The half-range is how far each seed sits from the average of the two. Two\n"
          "    cells whose values differ by less than roughly twice it are not telling you\n"
          "    they are different, whatever the colour of the heat map suggests.")


# --------------------------------------------------------------------------- figures
COST_TITLES = {
    "utility_cost": "Utility Cost",
    "federation_cost": "Federation Cost",
    "heterogeneity_cost": "Heterogeneity Cost at $\\epsilon$",
    "heterogeneity_cost_main": "Heterogeneity Cost, on its own",
    "privacy_cost": "Privacy Cost at $\\alpha$",
    "privacy_cost_main": "Privacy Cost, on its own",
    "interaction_cost": "Interaction",
    "collaboration_gain": "Collaboration Gain",
    "trivial_margin": "Margin over the trivial strategies",
}

# Quantities for which a HIGH value is good. They get their own colour scale, so that the
# reader never has to remember which way round a particular map is.
GAIN_KEYS = ("collaboration_gain", "trivial_margin")

COST_FORMULAS = {
    "utility_cost": "$NB_{cent} - NB_{FL(\\alpha,\\epsilon)}$",
    "federation_cost": "$NB_{cent} - NB_{FL(\\alpha_0,\\infty)}$",
    "heterogeneity_cost": "$NB_{FL(\\alpha_0,\\epsilon)} - NB_{FL(\\alpha,\\epsilon)}$",
    "heterogeneity_cost_main": "$NB_{FL(\\alpha_0,\\infty)} - NB_{FL(\\alpha,\\infty)}$",
    "privacy_cost": "$NB_{FL(\\alpha,\\infty)} - NB_{FL(\\alpha,\\epsilon)}$",
    "privacy_cost_main": "$NB_{FL(\\alpha_0,\\infty)} - NB_{FL(\\alpha_0,\\epsilon)}$",
    "interaction_cost": "$P(\\alpha,\\epsilon) - P(\\alpha_0,\\epsilon)$",
    "collaboration_gain": "$NB_{FL(\\alpha,\\epsilon)} - \\overline{NB}_{Local(\\alpha)}$",
    "trivial_margin": "$NB_{FL(\\alpha,\\epsilon)} - \\max(NB_{treat\\ all}, 0)$",
}


def _sig(v, digits=2):
    """Formats a small number with enough decimals to stay non-zero."""
    if not np.isfinite(v) or v == 0:
        return "0"
    dec = max(digits, digits - 1 - int(np.floor(np.log10(abs(v)))))
    return f"{v:.{min(dec, 6)}f}"


def _plain(text):
    """Strips the inline maths from a title, for printing to a terminal."""
    return re.sub(r"\$\\?([A-Za-z]+)\$", r"\1", text)


def _eps_label(e):
    """Legend entry for a budget. The run without DP is an infinite budget, and saying so
    keeps it on the same axis as the others instead of looking like a separate condition."""
    return "$\\epsilon = \\infty$" if not np.isfinite(e) else f"$\\epsilon$ = {e:g}"


def _eps_tick(e):
    return "$\\infty$" if not np.isfinite(e) else f"{e:g}"


def _orient(piv):
    """The grid orientation used everywhere: epsilon from infinity down to the tightest
    budget left to right, alpha from the mildest at the top to the harshest at the bottom."""
    return piv.sort_index(ascending=False).sort_index(axis=1, ascending=False)


def plot_decision_curve(A, alpha, path=None, show=True, xmax=None):
    """Decision curve for a single alpha, as a figure of its own.

    One figure per plot rather than a single panel: in the document each curve can then be
    placed at the size it deserves, with its own caption.

    Plotted: the federated curves (one per epsilon, plus the run without DP), the average of
    the local baselines, the centralised model, and the two strategies that require no model
    at all -- "treat all" and "treat none", the latter flat at zero by definition.
    """
    import matplotlib.pyplot as plt
    sub = A[A["alpha"] == alpha]
    if sub.empty:
        print(f"no data for alpha={alpha:g}")
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    # From the loosest budget to the tightest, so the legend reads in the same direction as
    # the columns of every heat map. The budget is an ordered quantity, so the curves get a
    # sequential ramp rather than the default cycle: the eye then reads the tightening of
    # the budget off the colour alone. plasma has no green in it, which keeps the federated
    # curves from colliding with the centralised reference.
    eps_sorted = sorted(sub["epsilon"].unique(), reverse=True)
    ramp = plt.cm.plasma(np.linspace(0.05, 0.85, max(len(eps_sorted), 2)))
    for e, colour in zip(eps_sorted, ramp):
        g = sub[sub["epsilon"] == e].sort_values("p_t")
        ax.plot(g.p_t, g.nb_fl, lw=1.4, color=colour, label=_eps_label(e))

    g = sub.drop_duplicates("p_t").sort_values("p_t")
    if "nb_central" in sub:
        ax.plot(g.p_t, g.nb_central, color="tab:green", lw=2.0, ls="-.",
                label="centralised")
    if "nb_local_avg" in sub:
        ax.plot(g.p_t, g.nb_local_avg, color="black", lw=2.2, label="average local")
    if "nb_treat_all" in sub:
        ax.plot(g.p_t, g.nb_treat_all, ls="--", color="grey", lw=1.2,
                label="treat all")
    ax.axhline(0, ls=":", color="dimgrey", lw=1.3, label="treat none")

    # One band per clinical window, in the same figure rather than one figure per window:
    # the curves are identical between them, only the interval one reads them over changes.
    # Drawn widest first so a narrower window nested inside stays visible on top of it.
    bands = sorted(CLINICAL_RANGES, key=lambda w: w[1] - w[0], reverse=True)
    for w, colour in zip(bands, ("gold", "darkorange", "seagreen", "slateblue")):
        ax.axvspan(*w, color=colour, alpha=0.14, zorder=0,
                   label=f"clinical thresholds {w[0]:g}-{w[1]:g}")
    ax.set_title(f"Decision curve   $\\alpha$ = {alpha:g}")
    ax.set_xlabel("threshold probability $p_t$")
    ax.set_ylabel("Net Benefit")
    xmax = CURVE_XMAX if xmax is None else xmax
    if xmax is not None:
        # zoom only: the curves beyond this point are still computed and still enter the
        # "all thresholds" reading, they are simply not drawn
        vis = sub[sub["p_t"] <= xmax]
        cols = [c for c in ("nb_fl", "nb_central", "nb_local_avg", "nb_treat_all")
                if c in vis]
        top = float(np.nanmax(vis[cols].to_numpy())) if len(vis) else None
        ax.set_xlim(0, xmax)
        if top is not None:
            ax.set_ylim(bottom=-0.02, top=top * 1.35)
    else:
        ax.set_ylim(bottom=-0.02)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print("saved:", path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_decision_curves(A, outdir=None, show=True, xmax=None):
    """One figure per alpha. Returns the list of files written."""
    written = []
    for a in sorted(A["alpha"].unique(), reverse=True):
        path = None
        if outdir:
            fname = f"decision_curve_alpha_{a:g}".replace(".", "p") + ".png"
            path = os.path.join(outdir, fname)
            written.append(path)
        plot_decision_curve(A, a, path=path, show=show, xmax=xmax)
    return written


def plot_safe_range_map(rng, window=None, path=None, show=True,
                        criterion="width"):
    """Safe Range width over the grid, for one reading window.

    The width is the longest contiguous run of safe thresholds IN THAT WINDOW, as a fraction
    of it, so 0 marks a failure point and 1 a configuration that is preferable everywhere a
    clinician would operate. Runs shorter than MIN_SAFE_RUN are set to 0 upstream, so a
    non-zero cell here is a range the model holds across, not a point it crosses.
    """
    import matplotlib.pyplot as plt
    sub = rng[rng["window"] == win_label(window)] if "window" in rng else rng
    if criterion not in sub:
        print(f"{criterion} is not available.")
        return
    piv = _orient(sub.pivot(index="alpha", columns="epsilon", values=criterion))
    fig, ax = plt.subplots(figsize=(1.3 * piv.shape[1] + 3, 0.8 * piv.shape[0] + 2.2))
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([_eps_tick(c) for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([f"{i:g}" for i in piv.index])
    ax.set_xlabel("$\\epsilon$   (tighter budget to the right)")
    ax.set_ylabel("$\\alpha$   (more heterogeneous downwards)")
    what = {"width": "Safe Range width   (0 = failure point)",
            "width_trivial": "Beats the trivial strategies",
            "width_local": "Beats the average local baseline"}[criterion]
    ax.set_title(what + "\n"
                 + ("all thresholds" if window is None
                    else f"clinical window $p_t \\in [{window[0]:g}, {window[1]:g}]$"))
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, "—" if v == 0 else f"{v:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print("saved:", path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def cost_map(A, cost="privacy_cost", window=None, stat=None):
    """Median of one cost term over thresholds, pivoted on the alpha x epsilon grid.

    `window=None` uses every threshold; a (lo, hi) tuple restricts the reading. Factored out
    so that the figure, the table and the CSV all apply the same reduction instead of each
    summarising over a different interval.
    """
    if cost not in A:
        return None
    sub = win_slice(A, window)
    stat = COST_STAT if stat is None else stat
    return _orient(sub.groupby(["alpha", "epsilon"])[cost].agg(stat).reset_index()
                   .pivot(index="alpha", columns="epsilon", values=cost))


def amplification(A, window=None, stat=None):
    """Privacy cost at each alpha as a multiple of the privacy cost at the mildest alpha.

    This is the interaction expressed as a ratio rather than a difference, and it is the
    single number the thesis is after: 1.00 means the same budget costs the same whatever
    the data look like, and the two axes of the stress test are independent. Anything above
    1 means heterogeneity makes privacy more expensive.

    Reported only where the reference is a real, positive cost: if privacy costs nothing at
    the mildest alpha -- or the federation happens to gain from the noise there -- dividing
    by it produces a ratio with no meaning, and a blank is the honest entry.
    """
    piv = cost_map(A, "privacy_cost", window, stat)
    if piv is None or piv.empty:
        return None
    ref = piv.loc[piv.index.max()]
    out = piv.divide(ref, axis=1).astype(float)
    # the mask is indexed by epsilon, so it applies to the COLUMNS, not the rows
    usable = (ref > 1e-4).to_numpy()
    if (~usable).any():
        out.loc[:, ~usable] = np.nan
    return out


def plot_cost_map(A, cost="privacy_cost", path=None, show=True, window=None,
                  spread="note"):
    """Heat map of one term of the cost decomposition over the alpha x epsilon grid.

    The colour scale is diverging and centred on zero whenever the term goes negative, so
    that a cell where the federation actually gains from the comparison cannot be mistaken
    for a small loss.

    `window` restricts the thresholds the median is taken over, exactly as `_safe_range`
    does for the Safe Range. The headline metrics of the thesis have to be read over the
    same window, otherwise the results chapter compares numbers summarised over different
    intervals. Above p_t ~ 0.5 the curves collapse towards zero and the differences flatten
    out: including those thresholds dilutes the cost precisely where it matters.
    """
    import matplotlib.pyplot as plt
    piv = cost_map(A, cost, window)
    if piv is None:
        print(f"{cost} is not available: the decomposition could not be computed.")
        return
    # "note"  one line under the figure: the reader calibrates the whole map once
    # "cells" the half-range printed in every cell, which is precise and unreadable
    # None    nothing
    hr = (cost_map(A, cost + SPREAD_SUFFIX, window)
          if spread in ("note", "cells") else None)
    fig, ax = plt.subplots(figsize=(1.3 * piv.shape[1] + 3, 0.8 * piv.shape[0] + 2.2))
    lo_v = float(np.nanmin(piv.values))
    hi_v = float(np.nanmax(piv.values))
    if cost in GAIN_KEYS:
        # a gain: green where the federated model is ahead, red where it is behind, and the
        # scale centred on zero so the sign is readable at a glance
        scale = max(abs(lo_v), abs(hi_v)) or 1.0
        im = ax.imshow(piv.values, cmap="RdYlGn", vmin=-scale, vmax=scale, aspect="auto")
    elif lo_v < 0:
        scale = max(abs(lo_v), abs(hi_v)) or 1.0
        im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
    else:
        scale = hi_v or 1.0
        im = ax.imshow(piv.values, cmap="Reds", vmin=0, vmax=scale, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([_eps_tick(c) for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([f"{i:g}" for i in piv.index])
    ax.set_xlabel("$\\epsilon$   (tighter budget to the right)")
    ax.set_ylabel("$\\alpha$   (more heterogeneous downwards)")
    _tag = ("all thresholds" if window is None
            else f"thresholds {window[0]:g}-{window[1]:g}")
    ax.set_title(f"{COST_STAT.capitalize()} {COST_TITLES.get(cost, cost)}   "
                 f"{COST_FORMULAS.get(cost, '')}\n{_tag}", fontsize=11)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isnan(v):
                continue
            colour = "white" if abs(v) > 0.6 * scale else "black"
            label = f"{v:.3f}"
            if spread == "cells" and hr is not None and not np.isnan(hr.values[i, j]):
                label += f"\n$\\pm${hr.values[i, j]:.3f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color=colour)
    if spread == "note" and hr is not None:
        typ = float(np.nanmedian(hr.values))
        if np.isfinite(typ) and typ > 0:
            # One line rather than a figure in every cell: the reader needs the scale of
            # the noise once, to calibrate the whole map, not cell by cell.
            fig.text(0.01, -0.02,
                     f"seed half-range {_sig(typ)}; differences below "
                     f"{_sig(2 * typ)} are not readable",
                     fontsize=8, color="dimgrey", ha="left")
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print("saved:", path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _profile_axis(cost):
    """Which axis a main effect varies along. The heterogeneity main effect is read with DP
    off, so only alpha appears in it; the privacy main effect is read at the mildest alpha,
    so only epsilon does."""
    return "epsilon" if cost.startswith("privacy") else "alpha"


def cost_profile(A, cost="heterogeneity_cost_main", window=None, stat=None):
    """A main effect as a one-dimensional series, over the axis it actually depends on.

    The value is constant along the other axis by construction, so the grid is collapsed
    rather than summarised: taking the mean across identical entries returns the entry.
    """
    piv = cost_map(A, cost, window, stat)
    if piv is None:
        return None
    return piv.mean(axis=1) if _profile_axis(cost) == "alpha" else piv.mean(axis=0)


def plot_cost_profile(A, cost="heterogeneity_cost_main", path=None, show=True, window=None):
    """Bar chart of a main effect, over the one axis it depends on.

    A main effect is measured with the other axis pinned at its reference value, so it is
    constant along every row (or column) of the grid. Drawn as a heat map it would be a
    block of identical entries, inviting the reader to look for a variation that cannot
    exist. A bar per level says the same thing in the space it deserves.

    The variation one might expect along the other axis is not lost: the difference between
    the main effect and the reading taken at the operating point is exactly the interaction,
    which has a map of its own.
    """
    import matplotlib.pyplot as plt
    vals = cost_profile(A, cost, window)
    if vals is None:
        print(f"{cost} is not available: the decomposition could not be computed.")
        return
    by = _profile_axis(cost)
    labels = [(_eps_tick(v) if by == "epsilon" else f"{v:g}") for v in vals.index]
    fig, ax = plt.subplots(figsize=(6.0, 0.55 * len(vals) + 2.0))
    ypos = np.arange(len(vals))
    ax.barh(ypos, vals.values, color="firebrick", height=0.62)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()             # reference level at the top, as in the maps
    ax.set_xlabel("Net Benefit lost")
    ax.set_ylabel("$\\alpha$   (more heterogeneous downwards)" if by == "alpha"
                  else "$\\epsilon$   (tighter budget downwards)")
    _tag = ("all thresholds" if window is None
            else f"thresholds {window[0]:g}-{window[1]:g}")
    ax.set_title(f"{COST_STAT.capitalize()} {COST_TITLES.get(cost, cost)}   "
                 f"{COST_FORMULAS.get(cost, '')}\n{_tag}", fontsize=11)
    span = float(np.nanmax(np.abs(vals.values))) or 1.0
    for y, v in zip(ypos, vals.values):
        ax.text(v + 0.02 * span, y, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(min(0.0, float(np.nanmin(vals.values)) * 1.1), span * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print("saved:", path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_cost_maps(A, outdir=None, show=True, window=None):
    """One heat map per term of the decomposition. Returns the list of files written.

    The main effects get no figure. Each is measured with the other axis pinned at its
    reference value, so it is constant along every row or column of the grid and a map of it
    would be a block of identical entries. They stay in the table, where they are what makes
    the four terms add up to the Utility Cost, and out of the figures, where they would only
    invite the reader to look for a variation that cannot exist. The Federation Cost gets no
    figure either, for the same reason taken one step further: it is a single number for the
    whole grid, and it is printed.
    """
    written = []
    tag = win_tag(window)
    for cost in ("utility_cost", "heterogeneity_cost", "privacy_cost", "interaction_cost",
                 "collaboration_gain", "trivial_margin"):
        if cost not in A:
            continue
        path = os.path.join(outdir, f"{cost}{tag}.png") if outdir else None
        plot_cost_map(A, cost, path=path, show=show, window=window)
        if path:
            written.append(path)
    return written


def cost_table(A, window=None, stat=None):
    """The whole decomposition, one row per configuration, over one reading window.

    `_hr` columns carry the half-range across seeds for the quantity they are named after.
    """
    if "utility_cost" not in A:
        return None
    base = ("utility_cost", "federation_cost",
            "heterogeneity_cost_main", "privacy_cost_main", "interaction_cost",
            "heterogeneity_cost", "privacy_cost",
            "collaboration_gain", "trivial_margin")
    keys = [k for k in base if k in A] + [k + SPREAD_SUFFIX for k in base
                                          if k + SPREAD_SUFFIX in A]
    sub = win_slice(A, window)
    out = (sub.groupby(["alpha", "epsilon"])[keys].agg(COST_STAT if stat is None else stat)
           .reset_index())
    amp = amplification(A, window, stat)
    if amp is not None:
        out = out.merge(amp.stack(future_stack=True).reset_index(name="amplification"),
                        on=["alpha", "epsilon"], how="left")
    out.insert(2, "window", win_label(window))
    return out.sort_values(["alpha", "epsilon"],
                           ascending=[False, False]).reset_index(drop=True)


def costs_to_latex(A, path=None, window=None):
    """Cost decomposition table for one reading window, ready to paste into the thesis.

    One row per configuration, the terms that add up to the Utility Cost, and the
    amplification factor. The Federation Cost is the same in every row and is stated in a
    comment rather than repeated down a column.
    """
    tbl = cost_table(A, window)
    if tbl is None:
        return ""
    fed = tbl["federation_cost"].agg(COST_STAT) if "federation_cost" in tbl else np.nan
    lines = [f"% Cost decomposition, {COST_STAT} over {win_label(window)}.",
             r"% Utility = Federation + Heterogeneity + Privacy + Interaction, in every row.",
             r"% Heterogeneity and Privacy are the main effects: each is measured with the",
             r"% other axis held at its reference value, so the four terms add up without",
             r"% counting the interaction twice.",
             r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             r"$\alpha$ & $\epsilon$ & Utility & Heterog. & Privacy & Interaction "
             r"& Amplif. \\",
             r"\midrule"]
    for _, r in tbl.iterrows():
        eps = r"$\infty$" if not np.isfinite(r.epsilon) else f"{r.epsilon:g}"

        def g(name, fmt="{:+.3f}"):
            v = r.get(name, np.nan)
            if not np.isfinite(v):
                return "--"
            out = fmt.format(v)
            return out.replace("-0.000", "+0.000")   # no negative zero in a printed table

        lines.append(f"{r.alpha:g} & {eps} & {g('utility_cost')} & "
                     f"{g('heterogeneity_cost_main')} & {g('privacy_cost_main')} & "
                     f"{g('interaction_cost')} & "
                     f"{g('amplification', '{:.2f}')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if np.isfinite(fed):
        lines.append(f"% Federation Cost (constant over the grid): {fed:+.4f}")
    tex = "\n".join(lines)
    if path:
        open(path, "w", encoding="utf-8").write(tex)
        print("saved:", path)
    return tex


def to_latex(rng, window=None, path=None):
    """Safe Range table for one reading window, ready to paste into the thesis.

    The Safe Range is the longest contiguous run of thresholds at which the federated model
    beats both references, and Width is that run as a fraction of the window. A run shorter
    than MIN_SAFE_RUN thresholds is reported as a failure point rather than as a narrow
    range.
    """
    d = rng[rng["window"] == win_label(window)] if "window" in rng else rng
    parts = "width_trivial" in d and "width_local" in d
    lines = [f"% Safe Range over {win_label(window)}. A threshold counts as safe when the",
             r"% federated model beats BOTH the trivial strategies and the average local",
             r"% baseline. The Safe Range is the longest CONTIGUOUS run of such thresholds,",
             f"% and a run shorter than {MIN_SAFE_RUN} thresholds is reported as a failure point.",
             r"% Width is that run as a fraction of the window; vs triv. and vs local are the",
             r"% fraction of the window covered by each condition on its own.",
             r"\begin{tabular}{@{}llrrrr@{}}" if parts else r"\begin{tabular}{@{}llrr@{}}",
             r"\toprule",
             (r"$\alpha$ & $\epsilon$ & Safe Range & Width & vs triv. & vs local \\"
              if parts else r"$\alpha$ & $\epsilon$ & Safe Range & Width \\"),
             r"\midrule"]
    for _, r in d.iterrows():
        eps = r"$\infty$" if not np.isfinite(r.epsilon) else f"{r.epsilon:g}"
        span = "--" if r.failure_point else f"[{r.p_t_min:.2f}, {r.p_t_max:.2f}]"
        row = f"{r.alpha:g} & {eps} & {span} & {r.width:.2f}"
        if parts:
            row += f" & {r.width_trivial:.2f} & {r.width_local:.2f}"
        lines.append(row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines)
    if path:
        open(path, "w", encoding="utf-8").write(tex)
        print("saved:", path)
    return tex


if __name__ == "__main__":
    # non-interactive backend: from a terminal there is no window to show the figures in
    import matplotlib
    matplotlib.use("Agg")

    os.makedirs(FIG_DIR, exist_ok=True)      # creates OUT_DIR as well

    A, rng = report()

    A.to_csv(os.path.join(OUT_DIR, "full_analysis.csv"), index=False)
    print(f"\nsaved: {os.path.join(OUT_DIR, 'full_analysis.csv')}")

    # One figure per alpha, carrying every clinical window as a shaded band: the curves do
    # not depend on the window, only the interval one reads them over does.
    plot_decision_curves(A, FIG_DIR, show=False)

    # Everything downstream of the curves is produced once per window, because there the
    # window is part of the answer and not just a region of the picture.
    tex_blocks = []
    for w in windows():
        tag, lab = win_tag(w), win_label(w)

        # the decomposition needs no local baseline, so it is written even when the Safe
        # Range cannot be computed yet
        ct = cost_table(A, w)
        if ct is not None:
            plot_cost_maps(A, FIG_DIR, show=False, window=w)
            ct.to_csv(os.path.join(OUT_DIR, f"costs{tag}.csv"), index=False)
            print(f"saved: {os.path.join(OUT_DIR, f'costs{tag}.csv')}")
            tex_blocks.append((f"cost decomposition, {lab}",
                               costs_to_latex(A, os.path.join(OUT_DIR, f"costs{tag}.tex"),
                                              window=w)))

        if rng is not None:
            plot_safe_range_map(rng, w, os.path.join(FIG_DIR, f"safe_range{tag}.png"),
                                show=False)
            # the two conditions on their own: which one a configuration failed is part of
            # the result, not a diagnostic
            for crit, name in (("width_trivial", "beats_trivial"),
                               ("width_local", "beats_local")):
                plot_safe_range_map(rng, w, os.path.join(FIG_DIR, f"{name}{tag}.png"),
                                    show=False, criterion=crit)
            tex_blocks.append((f"Safe Range, {lab}",
                               to_latex(rng, w,
                                        os.path.join(OUT_DIR, f"safe_range{tag}.tex"))))

    if rng is not None:
        rng.to_csv(os.path.join(OUT_DIR, "safe_range.csv"), index=False)
        print(f"saved: {os.path.join(OUT_DIR, 'safe_range.csv')}")

    for title, tex in tex_blocks:
        print(f"\n% ---- {title}\n{tex}")
    print(f"\nall outputs in {OUT_DIR}/")