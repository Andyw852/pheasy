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
    with contextlib.redirect_stdout(io.StringIO()):
        o.fit(A, y)
    return np.asarray(o._model.coef_, dtype=np.float64)


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
    SM, F = _load_sm_f(args.data_dir, args.n_configs, args.rows_per_config)
    n_conf = args.n_configs
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_conf)
    fold = max(1, n_conf // args.n_splits)
    folds = [order[i * fold:(i + 1) * fold] for i in range(args.n_splits)]
    folds = [f for f in folds if len(f) > 0]

    rpc = args.rows_per_config
    res = {m: {"rmse": [], "rel_l2": []} for m in methods}

    for fi, held in enumerate(folds):
        val = np.zeros(SM.shape[0], dtype=bool)
        for c in held:
            val[c * rpc:(c + 1) * rpc] = True
        tr = ~val
        print("fold %d/%d: train %d conf, val %d conf"
              % (fi + 1, len(folds), n_conf - len(held), len(held)), flush=True)
        for m in methods:
            t0 = time.time()
            coef = _method_fit(m, SM[tr], F[tr])
            pred = SM[val] @ coef
            err = pred - F[val]
            rmse = float(np.sqrt(np.mean(err * err)))
            rel = float(np.linalg.norm(err) / np.linalg.norm(F[val]))
            res[m]["rmse"].append(rmse)
            res[m]["rel_l2"].append(rel)
            print("    %-8s rmse=%.4e  relL2=%.4e  nnz=%d  (%.1fs)"
                  % (m, rmse, rel, int(np.count_nonzero(coef)), time.time() - t0),
                  flush=True)

    print()
    print("%-8s %12s %12s %12s %12s" % ("method", "rmse_mean", "rmse_std",
                                        "relL2_mean", "relL2_std"))
    for m in methods:
        r = res[m]
        print("%-8s %12.4e %12.4e %12.4e %12.4e"
              % (m, np.mean(r["rmse"]), np.std(r["rmse"]),
                 np.mean(r["rel_l2"]), np.std(r["rel_l2"])))


if __name__ == "__main__":
    main()
