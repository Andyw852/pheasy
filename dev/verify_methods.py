#!/usr/bin/env python3
"""Comprehensive verification: 5 methods from-scratch + alpha path + independence."""
import os, time, warnings
import numpy as np, scipy.sparse as sp
warnings.filterwarnings("ignore")
from pheasy.core.optimizer import Optimizer

def load_coo(p):
    z = np.load(p); s = tuple(int(x) for x in z["shape"])
    return sp.coo_matrix((z["data"], (z["row"], z["col"])), shape=s).tocsr()

sm = np.load("sm_dense.npy"); fm = np.load("fm1d.npz")["F"]
ns_h = load_coo("ns_harm.npz"); ns_a = load_coo("ns_anharm3.npz")
ns_full = sp.block_diag([ns_h, ns_a]).tocsr()
def fc3max(c):
    phi = ns_full @ c
    return float(np.max(np.abs(phi[ns_h.shape[0]:])))

print("=" * 100)
print("REFERENCE: OLS fc3_max = 37.04 eV/A^3  (from fit.out '参考值 37.04')")
print("Settings: standardize=True (--std), debias=1, mu grid [-8, -0.4] (10^-8..0.4), cv=5 GroupKFold")
print("=" * 100)

results = {}
for meth in ["OLS", "LASSO", "ALASSO", "RFE", "RFE-OLS-TSQR"]:
    if meth == "RFE":
        os.environ["PHEASY_RFE_STEP"] = "0.3"; os.environ["PHEASY_RFE_MIN_FEATURES"] = "100"
    if meth == "RFE-OLS-TSQR":
        os.environ["PHEASY_TSQR_STEP"] = "0.3"; os.environ["PHEASY_TSQR_MIN_FEATURES"] = "100"
    t0 = time.time()
    opt = Optimizer(meth, nalpha=40, alpha_min=-8, alpha_max=-0.4, cv=5,
                    max_iter=100000, rand_seed=0, standardize=True)
    opt.fit(sm, fm)
    c = opt.results["coef"]
    m = opt.metrics
    results[meth] = c
    print("[{:<12}] re={:.5e}  rmse_cv={:.5e}  nonzero={}/{}  fc3_max={:.4f}  "
          "alpha={}  ({:.0f}s)".format(
              meth, m["re"], m.get("rmse_path_mean", float("nan")),
              np.count_nonzero(c), c.shape[0], fc3max(c),
              opt.results.get("alpha"), time.time() - t0))

print()
print("=" * 100)
print("Independence check: each method is a SEPARATE from-scratch fit;")
print("pairwise max |coef_i - coef_j| of the coefficient vectors:")
print("=" * 100)
keys = list(results.keys())
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        d = float(np.max(np.abs(results[keys[i]] - results[keys[j]])))
        rel = float(np.linalg.norm(results[keys[i]] - results[keys[j]]) /
                    max(np.linalg.norm(results[keys[i]]), 1e-30))
        print("  {:>12} vs {:<12}  max|diff|={:.4e}   rel={:.4e}".format(keys[i], keys[j], d, rel))

print()
print("=" * 100)
print("LASSO alpha path (proves L1 sparsity is active):")
print("=" * 100)
from sklearn.linear_model import Lasso
for mu in [-0.4, -1, -2, -3, -4, -5, -6, -8]:
    a = 10.0 ** mu
    est = Lasso(alpha=a, max_iter=100000, fit_intercept=False)
    est.fit(sm.astype(np.float64), fm.astype(np.float64))
    print("  alpha=10^{:<3} = {:.2e}  nonzero={}  train re={:.4e}".format(
        mu, a, int(np.count_nonzero(est.coef_)),
        float(np.linalg.norm(sm @ est.coef_ - fm) / np.linalg.norm(fm))))
