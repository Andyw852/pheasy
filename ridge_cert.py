#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ridge_cert.py -- extended-grid certification for the RIDGE control at small n.

When the holdout table reports R@lo at some n AND RIDGE != OLS there, the a->0
limit is NOT in the table (RIDGE_BASE bottoms out at 1e-6), so the CV optimum
could be grid-truncated and the L1-vs-RIDGE gap inflated. This script re-runs
the grouped (by-config) holdout for RIDGE ONLY at that n with the grid bottom
pushed to 1e-12, and reports whether alpha* is INTERIOR (the true optimum was
found) or still pinned at an edge.

Verdict PASSES iff every fold's alpha* is interior (grid_min < alpha* < grid_max):
then the main table's R@lo was NOT truncating a smaller optimum, and the RIDGE
number (and the L1-vs-RIDGE gap built on it) stands as reported.

Usage:
    python3 ridge_cert.py <data_dir> --n-configs 6 --n-splits 3 --seed 0 --rows-per-config 567 --sm-dtype float32
"""
import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np
from scipy import sparse as sp

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
try:
    import pheasy  # noqa: F401
except ImportError:
    import types as _types
    _pheasy = _types.ModuleType("pheasy")
    _pheasy.__path__ = [_ROOT]
    sys.modules["pheasy"] = _pheasy

FIT_KW = dict(nalpha=20, cv=5, tol=1e-6, max_iter=20000, alpha_auto=True,
              decades=4.0)


def load_sm_f(d, n_configs, rows_per_config, sm_dtype):
    sm_prime = sp.load_npz(os.path.join(d, "sm_prime.npz"))
    ns_harm = sp.load_npz(os.path.join(d, "ns_harm.npz"))
    ns_anh = sp.load_npz(os.path.join(d, "ns_anharm3.npz"))
    NS = sp.block_diag([ns_harm, ns_anh], format="csr")
    n_rows = n_configs * rows_per_config
    dt = np.float32 if sm_dtype == "float32" else np.float64
    SM = np.asarray((sm_prime[:n_rows] @ NS).toarray(), dtype=dt)
    if SM.shape[0] != n_rows:
        raise SystemExit("SM has %d rows, expected %d (check --n-configs/"
                         "--rows-per-config)" % (SM.shape[0], n_rows))
    fm = np.load(os.path.join(d, "fm1d.npz"))
    key = "F" if "F" in fm else next(k for k in fm.files if not k.startswith("_"))
    F = np.asarray(fm[key], dtype=np.float64).ravel()[:n_rows]
    return SM, F


def fit_ridge(A, y, grid):
    """Fit RIDGE (standardized) with grouped CV over grid; return coef, flag, alpha."""
    from pheasy.core.optimizer import Optimizer
    kw = dict(FIT_KW, rand_seed=0, standardize=True, alpha=grid, alpha_auto=False)
    o = Optimizer("RIDGE", **kw)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        o.fit(A, y)
    a = float(o.results["alpha"])
    flag = ""
    if a <= grid[0] * (1.0 + 1e-9):
        flag = "R@lo"
    elif a >= grid[-1] * (1.0 - 1e-9):
        flag = "R@hi"
    return np.asarray(o.results["coef"], dtype=np.float64), flag, a


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("--n-configs", type=int, default=6)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--rows-per-config", type=int, default=567)
    ap.add_argument("--sm-dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--grid-min", type=float, default=1e-12)
    ap.add_argument("--grid-max", type=float, default=1e4)
    ap.add_argument("--n-alpha", type=int, default=80,
                    help="grid points over [grid-min, grid-max]; 80 -> ~1.6x step "
                         "(same density as RIDGE_BASE's 50 pts / 10 decades)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("PHEASY_CV_GROUP_SIZE", str(args.rows_per_config))
    _gs = int(os.environ["PHEASY_CV_GROUP_SIZE"] or "0")
    if _gs != args.rows_per_config:
        raise SystemExit("PHEASY_CV_GROUP_SIZE=%r overrides --rows-per-config=%d"
                         % (os.environ["PHEASY_CV_GROUP_SIZE"], args.rows_per_config))
    print("inner CV grouped by config (group_size=%d)" % _gs, flush=True)

    SM, F = load_sm_f(args.data_dir, args.n_configs, args.rows_per_config,
                      args.sm_dtype)
    print("loaded SM %s F %s (dtype=%s)" % (SM.shape, F.shape, SM.dtype), flush=True)
    grid = np.logspace(np.log10(args.grid_min), np.log10(args.grid_max), args.n_alpha)
    step = 10 ** (np.log10(grid[-1] / grid[0]) / (len(grid) - 1))
    print("cert grid  = %.2e .. %.2e (%d pts, step ~%.2fx)"
          % (grid[0], grid[-1], len(grid), step), flush=True)

    n_conf = args.n_configs
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_conf)
    folds = [f for f in np.array_split(order, args.n_splits) if len(f) > 0]
    rpc = args.rows_per_config

    rels, alphas, flags = [], [], []
    for fi, held in enumerate(folds):
        val = np.zeros(SM.shape[0], dtype=bool)
        for c in held:
            val[c * rpc:(c + 1) * rpc] = True
        tr = ~val
        print("fold %d/%d: train %d conf, val %d conf"
              % (fi + 1, len(folds), n_conf - len(held), len(held)), flush=True)
        t0 = time.time()
        coef, flag, a = fit_ridge(SM[tr], F[tr], grid)
        pred = SM[val] @ coef
        err = pred - F[val]
        rel = float(np.linalg.norm(err) / np.linalg.norm(F[val]))
        rels.append(rel)
        alphas.append(a)
        flags.append(flag)
        print("    RIDGE   relL2=%.4e  alpha*=%.3e  nnz=%d  %-8s  (%.1fs)"
              % (rel, a, int(np.count_nonzero(coef)), flag, time.time() - t0),
              flush=True)

    interior = all(f == "" for f in flags)
    print()
    print("per-fold alpha* = [" + ", ".join("%.3e" % a for a in alphas) + "]")
    print("per-fold relL2  = [" + ", ".join("%.4e" % r for r in rels) + "]")
    print("relL2 mean      = %.4e" % float(np.mean(rels)))
    print("relL2 median    = %.4e" % float(np.median(rels)))
    print("verdict         = %s" % ("INTERIOR (PASS)" if interior
                                    else "EDGE (FAIL): " + " ".join(flags)))


if __name__ == "__main__":
    main()
