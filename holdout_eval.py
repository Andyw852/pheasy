#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""holdout_eval.py -- cross-method holdout force-prediction RMSE / relL2.

Holds out WHOLE CONFIGURATIONS (groups): rows within one configuration (forces of
different atoms under the same displacement) are highly correlated, so random
row-splitting would leak and underestimate the error. Each method fits on the
remaining configs (its internal debias therefore runs on the TRAINING fold only),
then predicts forces on the held-out configs. Reports RMSE and relative L2, rotated
over several disjoint splits, as mean +/- std.

This closes the gap RMSE_CV cannot: RMSE_CV is the L1 path's pre-debias in-fold
estimate, whereas this measures the DELIVERED (debiased) model on unseen configs.

Usage:
    python3 holdout_eval.py <data_dir> --methods "OLS LASSO ALASSO RFE" \
        --n-configs 45 --n-splits 5 [--rows-per-config 567] [--seed 0]

n-sweep note: the interesting sweep is the CROSSING point where L1 methods start
to beat OLS, not the already-overdetermined 36/27/18. For MnIn2Se4 c3=7.0 (p=3678,
567 rows/config) OLS becomes underdetermined only below ~8 configs, so sweep
--n-configs 24 18 12 9 6 (24/18 cover the overdetermined-but-ill-conditioned band;
12/9/6 cover the underdetermined boundary). Include RIDGE as the control: only if
ALASSO beats RIDGE (not just OLS) does SPARSITY (not mere regularization) pay off.
For small n use --n-splits 3 and repeat with 2-3 --seed values (a single split of
1-2 configs is noisy); read the median, not just the mean, since underdetermined
OLS relL2 can blow up and dominate the mean.

Acceptance criteria for an n-sweep row to be usable (write these down BEFORE
running, not after):
  1. RIDGE R@hi > 0 voids the row: the CV optimum is grid-limited on the HIGH
     side (the true optimum may lie beyond the grid and be missed), so the
     RIDGE baseline is an upper bound and the L1-vs-RIDGE gap is inflated.
     R@lo is self-evidently harmless WHEN RIDGE == OLS in the table (the a->0
     limit is already present, so truncating the grid bottom costs nothing).
     R@lo only needs an extended-grid certification when RIDGE != OLS at that n
     (the a->0 limit is then NOT in the table); ridge_cert.py does exactly that.
  2. The conclusion row is RIDGE - ALASSO (--paired-refs 'OLS RIDGE'): only
     0/k folds favor RIDGE (i.e. ALASSO <= RIDGE on every fold) supports real
     sparsity at that n. (The output prints "k/k folds favor RIDGE" with
     favor = sum(diff < 0), so the acceptance test is favor == 0.)
  3. OLS - RIDGE turning positive at small n is CONFOUNDED: underdetermined OLS
     here is the un-regularized minimum-norm solution (scipy.linalg.lstsq
     cond=None on a float32 rank-deficient matrix), which inverts spurious
     ~3e-10 singular values (float32 rounding in the null directions) and blows
     up (relL2 ~ 10). That +gap measures "regularization fixes ill-posedness",
     not a clean "regularization pays off" signal. RIDGE (not OLS) is the honest
     baseline; the decisive row is RIDGE - L1 (criterion 2), which is unaffected.
  4. At n <= 9 the ALASSO adaptive weights may lose their adaptivity, but the
     mechanism is NOT assumed: the underdetermined ridge pilot can flatten the
     weights (W -> 0) or saturate them at the 1/eps ceiling (F -> 1) -- opposite
     failure modes, both flagged per fold. The OPERATIVE test is the ALASSO -
     LASSO paired row (--paired-refs 'OLS RIDGE' + the LASSO column): if it
     collapses (ALASSO - LASSO ~ 0), read that row as LASSO, not as a sparsity
     result. W/F explain WHY; they are not the test.
"""
import argparse
import io
import os
import sys
import time

import numpy as np
from scipy import sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_sm_f(d, n_configs, rows_per_config, sm_dtype):
    t0 = time.time()
    sm_prime = sp.load_npz(os.path.join(d, "sm_prime.npz"))
    ns_harm = sp.load_npz(os.path.join(d, "ns_harm.npz"))
    ns_anh = sp.load_npz(os.path.join(d, "ns_anharm3.npz"))
    NS = sp.block_diag([ns_harm, ns_anh], format="csr")
    SM = sm_prime @ NS
    n_rows = n_configs * rows_per_config
    _dt = np.float32 if sm_dtype == "float32" else np.float64
    SM = np.asarray(SM[:n_rows].toarray(), dtype=_dt)
    if SM.shape[0] != n_rows:
        raise SystemExit("SM has %d rows, expected %d" % (SM.shape[0], n_rows))
    fm = np.load(os.path.join(d, "fm1d.npz"))
    key = "F" if "F" in fm else list(fm.keys())[0]
    F = np.asarray(fm[key], dtype=np.float64).ravel()[:n_rows]
    print("loaded SM %s F %s (%.1fs)" % (SM.shape, F.shape, time.time() - t0), flush=True)
    return SM, F


# [FIX] single source of truth for the fit settings: the summary header and the
# actual Optimizer call must read the SAME dict, or the header silently drifts
# from the run and becomes misinformation instead of a reproducibility record.
FIT_KW = dict(nalpha=20, cv=5, tol=1e-6, max_iter=20000, alpha_auto=True,
              decades=4.0)
# [FIX] the RIDGE control must actually regularize. The default grid
# [1e-6, 1e-2] on standardized columns is ~unregularized (RIDGE collapses to
# OLS), which would make the "regularization vs sparsity" comparison useless.
# 50 points over 10 decades keeps the step at ~1.6x (the same density P38
# restored for ALASSO), not 2.2x.
#
# The grid is deliberately NOT scaled by n: standardize=True rescales every
# column to unit L2 norm, so X^T X has unit diagonal, trace = p and a spectrum
# independent of n -- alpha is already n-comparable. Scaling only RIDGE would
# BREAK that comparability, because ALASSO's own ridge pilot runs the SAME
# _ridge_solve(A_fit, y, init_alpha=1e-3) on the SAME standardized columns with
# an UNSCALED alpha. Either both scale or neither; neither is correct.
RIDGE_BASE = np.logspace(-6, 4, 50)          # 1e-6 .. 1e4, 50 pts


def _method_fit(method, A, y):
    from pheasy.core.optimizer import Optimizer
    std = method in ("LASSO", "ALASSO", "RIDGE")
    kw = dict(FIT_KW, rand_seed=0, standardize=std)
    if method == "LASSO":
        # [FIX] holdout_eval constructs Optimizer directly (bypassing run_pheasy,
        # the ONLY caller of derive_alpha_grid), so plain LASSO used to get the
        # hardcoded logspace(-6,-2) grid -- an arbitrary scale that pins alpha*
        # (under- or over-regularized depending on the data). Replicate run_pheasy:
        # derive the grid from the data. standardize=True does its own column
        # normalization, so pass the raw A.
        from pheasy.core.optimizer import derive_alpha_grid
        kw["alpha"] = derive_alpha_grid(A, y, nalpha=FIT_KW["nalpha"],
                                        decades=FIT_KW["decades"], standardize=True)
    if method == "RIDGE":
        kw["alpha"] = RIDGE_BASE
        kw["alpha_auto"] = False
    o = Optimizer(method, **kw)
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        o.fit(A, y)
    flag = ""
    if "grid MINIMUM" in buf.getvalue():
        flag += "A@MIN"
    if method in ("LASSO", "ALASSO"):
        # [FIX] decade flag: ALASSO's weighted grid switches regime with n
        # (overdetermined -> max(decades,6)=6 decades; underdetermined -> 4), so
        # report the effective span to make the switch visible per row.
        _al = getattr(o._model, "alphas_", None)
        if _al is not None and np.size(_al) > 1:
            _al = np.asarray(_al)
            flag += ("+" if flag else "") + "D%.1f" % float(
                np.log10(_al.max() / _al.min()))
    if method == "ALASSO":
        # [FIX] diagnostics for the adaptive-weight failure mode at small n:
        # W = std(log w) (uniform -> 0), F = fraction of pilot coefficients at the
        # eps floor (saturating weights at the 1/eps ceiling). They are WHY, not
        # the test: the operative test is the ALASSO - LASSO paired row.
        _w = getattr(o._model, "_weights", None)
        if _w is not None:
            flag += ("+" if flag else "") + "W%.2f" % float(
                np.std(np.log(np.maximum(np.asarray(_w), 1e-300))))
        _bf = getattr(o._model, "_beta0_floor", None)
        if _bf is not None:
            flag += ("+" if flag else "") + "F%.2f" % float(_bf)
    if method == "RIDGE":
        # same edge detection we spent many rounds giving ALASSO: the control is
        # useless if CV picked a grid edge (wants outside the grid).
        a = float(o._model.alpha_)
        if a <= RIDGE_BASE[0] * (1.0 + 1e-9):
            flag += ("+" if flag else "") + "R@lo"
        elif a >= RIDGE_BASE[-1] * (1.0 - 1e-9):
            flag += ("+" if flag else "") + "R@hi"
    # use the formal exit (o.results["coef"]) rather than o._model.coef_; the
    # latter is fragile (depends on a write-back that could be removed).
    return np.asarray(o.results["coef"], dtype=np.float64), flag


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir")
    ap.add_argument("--methods", default="OLS LASSO ALASSO RFE")
    ap.add_argument("--n-configs", type=int, default=45)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--rows-per-config", type=int, default=567)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paired-refs", default="OLS",
                    help="space-separated reference methods for the paired relL2 "
                         "comparison, e.g. 'OLS RIDGE' (RIDGE - ALASSO is the "
                         "sparsity-vs-regularization conclusion row)")
    ap.add_argument("--sm-dtype", default=os.environ.get("PHEASY_SM_DTYPE", "float64"),
                    help="sensing-matrix precision; mirrors PHEASY_SM_DTYPE so the "
                         "holdout describes the same numerical path as the scan")
    args = ap.parse_args()

    methods = args.methods.split()
    # [FIX] the internal CV (for alpha selection / RFE feature count) must also
    # be grouped by config; otherwise it leaks across correlated atom forces
    # within one displacement -> biases alpha* low -> denser models -> result
    # leans toward "L1 ≈ OLS", i.e. toward the conclusion we want to prove
    # (a reviewer would call this out). rows_per_config = 3 × natoms exactly
    # matches what run_pheasy sets, so the train fold divides evenly.
    os.environ.setdefault("PHEASY_CV_GROUP_SIZE", str(args.rows_per_config))
    # [FIX] the setdefault above is silent if the env var was already set to a
    # different value (it would not override) or if n_samples % gs != 0 (the
    # detector then returns None and the inner CV silently falls back to
    # row-random). Assert + announce so a grouped-CV failure cannot be silent.
    _raw = os.environ["PHEASY_CV_GROUP_SIZE"]
    _gs = int(_raw or "0")
    if _gs != args.rows_per_config:
        raise SystemExit(
            "PHEASY_CV_GROUP_SIZE=%r overrides --rows-per-config=%d"
            % (_raw, args.rows_per_config))
    print("inner CV grouped by config (group_size=%d)" % _gs, flush=True)
    SM, F = _load_sm_f(args.data_dir, args.n_configs, args.rows_per_config,
                       args.sm_dtype)
    print("settings: " + " ".join("%s=%s" % kv for kv in FIT_KW.items())
          + " standardize=L1/RIDGE sm_dtype=%s group_size=%d"
          % (args.sm_dtype, args.rows_per_config), flush=True)
    n_conf = args.n_configs
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_conf)
    # array_split divides as evenly as possible (e.g. 18/5 -> [4,4,4,3,3]), whereas
    # n_conf // n_splits drops the remainder (18/5 fold=3 covers only 15, silently
    # leaves 3 configs never held out).
    folds = [f for f in np.array_split(order, args.n_splits) if len(f) > 0]

    rpc = args.rows_per_config
    res = {m: {"rmse": [], "rel_l2": [], "flag": []} for m in methods}

    for fi, held in enumerate(folds):
        val = np.zeros(SM.shape[0], dtype=bool)
        for c in held:
            val[c * rpc:(c + 1) * rpc] = True
        tr = ~val
        print("fold %d/%d: train %d conf, val %d conf"
              % (fi + 1, len(folds), n_conf - len(held), len(held)), flush=True)
        A_tr, y_tr = SM[tr], F[tr]  # hoist the materialization out of the method loop
        A_val = SM[val]
        for m in methods:
            t0 = time.time()
            coef, flag = _method_fit(m, A_tr, y_tr)
            pred = A_val @ coef
            err = pred - F[val]
            rmse = float(np.sqrt(np.mean(err * err)))
            rel = float(np.linalg.norm(err) / np.linalg.norm(F[val]))
            res[m]["rmse"].append(rmse)
            res[m]["rel_l2"].append(rel)
            res[m]["flag"].append(flag)
            print("    %-8s rmse=%.4e  relL2=%.4e  nnz=%d  %-20s  (%.1fs)"
                  % (m, rmse, rel, int(np.count_nonzero(coef)), flag, time.time() - t0),
                  flush=True)

    print()
    print("%-8s %12s %12s %12s %12s %12s %8s" % ("method", "rmse_mean", "rmse_std",
                                                 "relL2_mean", "relL2_std",
                                                 "relL2_med", "A@MIN"))
    for m in methods:
        r = res[m]
        amin = sum(1 for f in r["flag"] if "A@MIN" in f)
        print("%-8s %12.4e %12.4e %12.4e %12.4e %12.4e %5d/%d"
              % (m, np.mean(r["rmse"]), np.std(r["rmse"]),
                 np.mean(r["rel_l2"]), np.std(r["rel_l2"]),
                 np.median(r["rel_l2"]), amin, len(r["flag"])))
    # RIDGE control edge note (the control is useless if CV picked a grid edge)
    if "RIDGE" in methods:
        r = res["RIDGE"]
        lo = sum(1 for f in r["flag"] if "R@lo" in f)
        hi = sum(1 for f in r["flag"] if "R@hi" in f)
        print("RIDGE selected alpha at grid edge in %d/%d folds (lo %d, hi %d)"
              % (lo + hi, len(r["flag"]), lo, hi))
    # paired comparison: same fold, each reference vs every other method.
    # The decisive row for the sparsity question is RIDGE - ALASSO (what the
    # sparse structure adds beyond plain regularization).
    for ref in args.paired_refs.split():
        if ref not in methods:
            print("note: --paired-refs %r not in --methods; skipped" % ref)
            continue
        ref_rel = np.array(res[ref]["rel_l2"])
        others = [m for m in methods if m != ref]
        if not others:
            continue
        print()
        print("paired relL2 diff (%s - method, negative = %s better):" % (ref, ref))
        for m in others:
            diff = ref_rel - np.array(res[m]["rel_l2"])
            favor = int(np.sum(diff < 0))
            print("  %s - %-8s: mean %.4e  std %.4e  (%d/%d folds favor %s; all: %s)"
                  % (ref, m, np.mean(diff), np.std(diff), favor, len(diff), ref,
                     " ".join("%.3e" % d for d in diff)))


if __name__ == "__main__":
    main()
