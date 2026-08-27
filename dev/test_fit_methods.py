#!/usr/bin/env python3
"""Local verification of the five pheasy force-constant fitting methods.

Loads the cached sensing matrix (sm_dense.npy) and forces (fm1d.npz) from
MnIn2Se4_lasso and runs OLS / LASSO / ALASSO / RFE / RFE-OLS-TSQR, then reports
the fit quality and the reconstructed force-constant amplitudes.

Usage:
    cd MnIn2Se4_lasso
    PYTHONPATH=~/software python3 ../dev/test_fit_methods.py
"""
import os
import time
import warnings

import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")

from pheasy.core.optimizer import Optimizer


def load_coo(path):
    z = np.load(path)
    shape = tuple(int(x) for x in z["shape"])
    return sp.coo_matrix((z["data"], (z["row"], z["col"])), shape=shape).tocsr()


def main():
    sm = np.load("sm_dense.npy")
    fm = np.load("fm1d.npz")["F"]
    ns_harm = load_coo("ns_harm.npz")
    ns_anharm = load_coo("ns_anharm3.npz")
    ns_full = sp.block_diag([ns_harm, ns_anharm]).tocsr()

    def fc_amplitudes(coef):
        phi = ns_full @ coef
        fc2 = phi[: ns_harm.shape[0]]
        fc3 = phi[ns_harm.shape[0]:]
        return float(np.max(np.abs(fc2))), float(np.max(np.abs(fc3)))

    ref_fc3 = float(open("fc3_max.txt").read().strip())
    print("SM {} {}   FM {} {}".format(sm.shape, sm.dtype, fm.shape, fm.dtype))
    print("ns_harm {}  ns_anharm {}".format(ns_harm.shape, ns_anharm.shape))
    print("reference fc3_max.txt = {}".format(ref_fc3))
    print("=" * 100)

    methods = [
        "OLS",
        "LASSO",
        "ALASSO",
        "RFE",
        "RFE-OLS-TSQR",
    ]

    for meth in methods:
        if meth == "RFE":
            os.environ["PHEASY_RFE_STEP"] = os.environ.get("PHEASY_RFE_STEP", "0.3")
            os.environ["PHEASY_RFE_MIN_FEATURES"] = os.environ.get("PHEASY_RFE_MIN_FEATURES", "100")
        if meth == "RFE-OLS-TSQR":
            os.environ["PHEASY_TSQR_STEP"] = os.environ.get("PHEASY_TSQR_STEP", "0.3")
            os.environ["PHEASY_TSQR_MIN_FEATURES"] = os.environ.get("PHEASY_TSQR_MIN_FEATURES", "100")

        t0 = time.time()
        # alpha_min/alpha_max are POWERS OF 10 (exponents), same as the
        # pheasy CLI (--mu_min/--alpha_min): 10^-10 ... 10^-3.5
        opt = Optimizer(
            meth, nalpha=40, alpha_min=-10, alpha_max=-3.5,
            cv=5, max_iter=100000, rand_seed=0,
        )
        opt.fit(sm, fm)
        m = opt.metrics
        r = opt.results
        coef = r["coef"]
        fc2, fc3 = fc_amplitudes(coef)
        alpha = r.get("alpha")
        n_iter = r.get("n_iter")
        cv_rmse = m.get("rmse_path_mean", float("nan"))
        print(
            "[{:<12}] re={:.5e}  rmse={:.5e}  rmse_cv={:.5e}  "
            "nonzero={}/{}  fc2_max={:.4f}  fc3_max={:.4f}  "
            "alpha={:.3e}  iters={}  ({:.0f}s)".format(
                meth, m["re"], m["rmse"], cv_rmse,
                np.count_nonzero(coef), coef.shape[0], fc2, fc3,
                alpha if alpha is not None else float("nan"),
                n_iter, time.time() - t0,
            )
        )


if __name__ == "__main__":
    main()
