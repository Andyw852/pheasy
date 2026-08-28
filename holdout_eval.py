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
"""
import argparse
import io
import os
import sys
import time

import numpy as np
from scipy import sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_sm_f(d, n_configs, rows_per_config):
    t0 = time.time()
    sm_prime = sp.load_npz(os.path.join(d, "sm_prime.npz"))
    ns_harm = sp.load_npz(os.path.join(d, "ns_harm.npz"))
    ns_anh = sp.load_npz(os.path.join(d, "ns_anharm3.npz"))
    NS = sp.block_diag([ns_harm, ns_anh], format="csr")
    SM = sm_prime @ NS
    n_rows = n_configs * rows_per_config
    SM = np.asarray(SM[:n_rows].toarray(), dtype=np.float64)
    assert SM.shape[0] == n_rows, "SM has %d rows, expected %d" % (SM.shape[0], n_rows)
    fm = np.load(os.path.join(d, "fm1d.npz"))
    key = "F" if "F" in fm else list(fm.keys())[0]
    F = np.asarray(fm[key], dtype=np.float64).ravel()[:n_rows]
    print("loaded SM %s F %s (%.1fs)" % (SM.shape, F.shape, time.time() - t0), flush=True)
    return SM, F


def _method_fit(method, A, y):
    from pheasy.core.optimizer import Optimizer
    std = method in ("LASSO", "ALASSO", "RIDGE")
    o = Optimizer(method, nalpha=20, cv=5, tol=1e-6, max_iter=20000,
                  rand_seed=0, standardize=std, alpha_auto=True, decades=4.0)
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        o.fit(A, y)
    flag = "A@MIN" if "grid MINIMUM" in buf.getvalue() else ""
    # use the formal exit (o.results["coef"]) rather than o._model.coef_; the
    # latter is fragile (depends on L1975 write-back which could be removed).
    return np.asarray(o.results["coef"], dtype=np.float64), flag


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir")
    ap.add_argument("--methods", default="OLS LASSO ALASSO RFE")
    ap.add_argument("--n-configs", type=int, default=45)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--rows-per-config", type=int, default=567)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    methods = args.methods.split()
    # [FIX] the internal CV (for alpha selection / RFE feature count) must also
    # be grouped by config; otherwise it leaks across correlated atom forces
    # within one displacement -> biases alpha* low -> denser models -> result
    # leans toward "L1 ≈ OLS", i.e. toward the conclusion we want to prove
    # (審稿人會直接挑這點). rows_per_config = 3 × natoms exactly matches what
    # run_pheasy sets, so the train fold (36 configs × 567 rows) divides evenly.
    os.environ.setdefault("PHEASY_CV_GROUP_SIZE", str(args.rows_per_config))
    SM, F = _load_sm_f(args.data_dir, args.n_configs, args.rows_per_config)
    n_conf = args.n_configs
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_conf)
    # array_split divides as evenly as possible (e.g. 18/5 -> [4,4,4,3,3]), whereas
    # n_conf // n_splits drops the remainder (18/5 fold=3 covers only 15, silently
    # leaves 3 configs never held out).
    folds = [f for f in np.array_split(order, args.n_splits) if len(f) > 0]

    rpc = args.rows_per_config
    res = {m: {"rmse": [], "rel_l2": []} for m in methods}

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
            print("    %-8s rmse=%.4e  relL2=%.4e  nnz=%d  %5s  (%.1fs)"
                  % (m, rmse, rel, int(np.count_nonzero(coef)), flag, time.time() - t0),
                  flush=True)

    print()
    print("%-8s %12s %12s %12s %12s" % ("method", "rmse_mean", "rmse_std",
                                        "relL2_mean", "relL2_std"))
    for m in methods:
        r = res[m]
        print("%-8s %12.4e %12.4e %12.4e %12.4e"
              % (m, np.mean(r["rmse"]), np.std(r["rmse"]),
                 np.mean(r["rel_l2"]), np.std(r["rel_l2"])))
    # paired comparison: same fold, OLS vs each other method (relL2 difference)
    if "OLS" in methods and len(methods) > 1:
        print()
        print("paired relL2 diff (OLS - method, negative = OLS better):")
        ols_rel = np.array(res["OLS"]["rel_l2"])
        for m in methods:
            if m == "OLS":
                continue
            diff = ols_rel - np.array(res[m]["rel_l2"])
            print("  OLS - %-8s: mean %.4e  std %.4e  (all %d folds: %s)"
                  % (m, np.mean(diff), np.std(diff), len(diff),
                     " ".join("%.3e" % d for d in diff)))


if __name__ == "__main__":
    main()
