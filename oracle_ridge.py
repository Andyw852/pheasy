#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""oracle_ridge.py -- oracle (holdout-tuned) ridge baseline at small n.

The CV-selected alpha* can be wrong when the inner CV is deep-underdetermined
(n=6/n=9: inner training folds have ~1701-3402 rows < p=3678), so the delivered
RIDGE row may understate the best ridge achievable. This script computes the
ORACLE ridge: for every alpha in a WIDE grid (1e-12..1e4), the out-of-fold
relL2 on the SAME grouped folds the holdout uses, then takes the grid minimum
of the mean curve. RIDGE_oracle is the strongest honest ridge baseline -- it is
immune to both "CV picked a bad alpha" and "grid truncation" (the grid bottom
1e-12 is below the ~1e-10 the old leaked GCV found).

Replicates holdout_eval.py exactly: per outer fold, standardize the TRAIN fold
columns to unit L2 norm, economic SVD, closed-form w(a)=Vt.T@((s/(s^2+a))*Uty),
un-scale coef = w/col_scale, predict the RAW val fold.

Usage:
    python3 oracle_ridge.py <data_dir> --n-configs 6 --n-splits 3 --seed 0 --rows-per-config 567 --sm-dtype float32
"""
import argparse
import os
import sys

import numpy as np
from scipy import sparse as sp

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def load_sm_f(d, n_configs, rows_per_config, sm_dtype):
    sm_prime = sp.load_npz(os.path.join(d, "sm_prime.npz"))
    ns_harm = sp.load_npz(os.path.join(d, "ns_harm.npz"))
    ns_anh = sp.load_npz(os.path.join(d, "ns_anharm3.npz"))
    NS = sp.block_diag([ns_harm, ns_anh], format="csr")
    n_rows = n_configs * rows_per_config
    dt = np.float32 if sm_dtype == "float32" else np.float64
    SM = np.asarray((sm_prime[:n_rows] @ NS).toarray(), dtype=dt)
    fm = np.load(os.path.join(d, "fm1d.npz"))
    key = "F" if "F" in fm else next(k for k in fm.files if not k.startswith("_"))
    F = np.asarray(fm[key], dtype=np.float64).ravel()[:n_rows]
    return SM, F


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir")
    ap.add_argument("--n-configs", type=int, required=True)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--rows-per-config", type=int, default=567)
    ap.add_argument("--sm-dtype", default="float32")
    ap.add_argument("--grid-min", type=float, default=1e-12)
    ap.add_argument("--grid-max", type=float, default=1e4)
    ap.add_argument("--n-alpha", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    SM, F = load_sm_f(args.data_dir, args.n_configs, args.rows_per_config,
                      args.sm_dtype)
    print("loaded SM %s F %s (dtype=%s)" % (SM.shape, F.shape, SM.dtype),
          flush=True)
    alphas = np.logspace(np.log10(args.grid_min), np.log10(args.grid_max),
                         args.n_alpha)

    n_conf = args.n_configs
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_conf)
    folds = [f for f in np.array_split(order, args.n_splits) if len(f) > 0]
    rpc = args.rows_per_config

    A64 = np.asarray(SM, dtype=np.float64)
    n_alpha = len(alphas)
    oracle = np.zeros((n_alpha, len(folds)))

    for k, held in enumerate(folds):
        val = np.zeros(SM.shape[0], dtype=bool)
        for c in held:
            val[c * rpc:(c + 1) * rpc] = True
        tr = ~val
        # per-fold standardization (matches holdout _method_fit)
        Atr = A64[tr]  # materialize the train slice once (bool-index copies)
        col_scale = np.sqrt(np.einsum("ij,ij->j", Atr, Atr))
        col_scale = np.where(col_scale < 1e-30, 1.0, col_scale)
        A_tr_s = Atr / col_scale[None, :]
        U, s, Vt = np.linalg.svd(A_tr_s, full_matrices=False)
        Uty = U.T @ F[tr]
        A_val_s = A64[val] / col_scale[None, :]
        Fv_norm = float(np.linalg.norm(F[val]))
        for j, a in enumerate(alphas):
            w = Vt.T @ ((s / (s * s + a)) * Uty)
            pred = A_val_s @ w
            oracle[j, k] = float(np.linalg.norm(pred - F[val]) / Fv_norm)
        print("fold %d/%d: train %d conf, val %d conf (done)"
              % (k + 1, len(folds), n_conf - len(held), len(held)), flush=True)

    curve = oracle.mean(axis=1)
    jbest = int(np.argmin(curve))
    ridge_oracle = float(curve[jbest])
    a_oracle = float(alphas[jbest])
    print()
    print("oracle ridge over %d alphas (%.2e..%.2e):" % (n_alpha, alphas[0], alphas[-1]))
    print("  RIDGE_oracle = %.4e  at alpha* = %.3e" % (ridge_oracle, a_oracle))
    print("  per-fold oracle relL2 at best alpha = [%s]"
          % ", ".join("%.4e" % x for x in oracle[jbest]))
    # also print the curve around the min (coarse) for the plateau check
    idx = [int(np.argmin(np.abs(alphas - a))) for a in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4)]
    print("  curve sample (alpha -> mean relL2):")
    for i in idx:
        print("    %.1e -> %.4e" % (alphas[i], curve[i]))


if __name__ == "__main__":
    main()
