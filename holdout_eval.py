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
RIDGE_ALPHAS = np.logspace(-6, 4, 30)  # 1e-6 .. 1e4


def _method_fit(method, A, y):
    from pheasy.core.optimizer import Optimizer
    std = method in ("LASSO", "ALASSO", "RIDGE")
    kw = dict(FIT_KW, rand_seed=0, standardize=std)
    if method == "RIDGE":
        kw["alpha"] = RIDGE_ALPHAS
        kw["alpha_auto"] = False
    o = Optimizer(method, **kw)
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        o.fit(A, y)
    flag = ""
    if "grid MINIMUM" in buf.getvalue():
        flag += "A@MIN"
    if method == "RIDGE":
        # same edge detection we spent many rounds giving ALASSO: the control is
        # useless if CV picked a grid edge (wants outside the grid).
        a = float(o._model.alpha_)
        if a <= RIDGE_ALPHAS[0] * (1.0 + 1e-9):
            flag += ("+" if flag else "") + "R@lo"
        elif a >= RIDGE_ALPHAS[-1] * (1.0 - 1e-9):
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
    _gs = int(os.environ["PHEASY_CV_GROUP_SIZE"])
    if _gs != args.rows_per_config:
        raise SystemExit(
            "PHEASY_CV_GROUP_SIZE=%d overrides --rows-per-config=%d"
            % (_gs, args.rows_per_config))
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
            print("    %-8s rmse=%.4e  relL2=%.4e  nnz=%d  %5s  (%.1fs)"
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
    # paired comparison: same fold, OLS vs each other method (relL2 difference)
    if "OLS" in methods and len(methods) > 1:
        print()
        print("paired relL2 diff (OLS - method, negative = OLS better):")
        ols_rel = np.array(res["OLS"]["rel_l2"])
        for m in methods:
            if m == "OLS":
                continue
            diff = ols_rel - np.array(res[m]["rel_l2"])
            favor = int(np.sum(diff < 0))
            print("  OLS - %-8s: mean %.4e  std %.4e  (%d/%d folds favor OLS; all: %s)"
                  % (m, np.mean(diff), np.std(diff), favor, len(diff),
                     " ".join("%.3e" % d for d in diff)))


if __name__ == "__main__":
    main()
