#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fit.py -- single-method force-constant fit with the corrected settings.

通用拟合脚本: 在含 sm_prime.npz / ns_harm.npz / ns_anharm3.npz / fm1d.npz 的
材料目录上, 用给定方法拟合二阶/三阶力常数, 报告 nnz / alpha / 拟合误差 / 诊断标志.

数据准备 (生成 sm_prime.npz 等) 由预热步骤完成:
    bash pheasy_fit.sh FIT_METHOD=OLS C3_CUTOFF=7.0 SM_DTYPE=float32
(或 pheasy --dim ... -d ... --disp_file ...); 本脚本只做"拟合 + 报告".

用法:
    python fit.py <data_dir> --method LASSO --n-configs 45 --sm-dtype float32 --write-coef c.npz

方法: OLS | LASSO | ALASSO | RFE | RFE-OLS-TSQR | RIDGE

固定设置 (与 holdout_eval.py 的 FIT_KW 一致, 是 P37-P48 逐轮校正后的结论):
    nalpha=20 cv=5 tol=1e-6 max_iter=20000 alpha_auto=True decades=4.0
    LASSO/ALASSO/RIDGE 列标准化; LASSO 用 derive_alpha_grid (数据驱动, 非硬编码网格);
    RIDGE 用 logspace(-6,4,50) 控制网格 (alpha_auto=False).
诊断标志:
    A@MIN  CV 选了网格最小 alpha (欠正则边缘)
    D#.#   LASSO/ALASSO 实际 alpha 网格跨度 (decade)
    W#.##  ALASSO 自适应权重 log 标准差 (0 = 权重摊平)
    F#.##  ALASSO 权重饱和在 1/eps 的比例 (1 = 全饱和)
    R@lo / R@hi  RIDGE 的 CV 最优 alpha 落在控制网格底端/顶端
"""
import argparse
import contextlib
import io
import os
import sys

import numpy as np
from scipy import sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIT_KW = dict(nalpha=20, cv=5, tol=1e-6, max_iter=20000, alpha_auto=True,
              decades=4.0)
RIDGE_BASE = np.logspace(-6, 4, 50)          # 1e-6 .. 1e4, 50 pts


def load_sm_f(d, n_configs, rows_per_config, sm_dtype):
    """Load the sensing matrix SM = sm_prime @ NS and force vector F."""
    sm_prime = sp.load_npz(os.path.join(d, "sm_prime.npz"))
    ns_harm = sp.load_npz(os.path.join(d, "ns_harm.npz"))
    ns_anh = sp.load_npz(os.path.join(d, "ns_anharm3.npz"))
    NS = sp.block_diag([ns_harm, ns_anh], format="csr")
    SM = sm_prime @ NS
    n_rows = n_configs * rows_per_config
    dt = np.float32 if sm_dtype == "float32" else np.float64
    SM = np.asarray(SM[:n_rows].toarray(), dtype=dt)
    if SM.shape[0] != n_rows:
        raise SystemExit("SM has %d rows, expected %d (check --n-configs/"
                         "--rows-per-config)" % (SM.shape[0], n_rows))
    fm = np.load(os.path.join(d, "fm1d.npz"))
    key = "F" if "F" in fm else list(fm.keys())[0]
    F = np.asarray(fm[key], dtype=np.float64).ravel()[:n_rows]
    return SM, F


def fit_method(method, A, y):
    """Fit one method with the corrected settings; return (coef, flag, alpha)."""
    from pheasy.core.optimizer import Optimizer
    std = method in ("LASSO", "ALASSO", "RIDGE")
    kw = dict(FIT_KW, rand_seed=0, standardize=std)
    if method == "LASSO":
        from pheasy.core.optimizer import derive_alpha_grid
        kw["alpha"] = derive_alpha_grid(A, y, nalpha=FIT_KW["nalpha"],
                                        decades=FIT_KW["decades"], standardize=True)
    if method == "RIDGE":
        kw["alpha"] = RIDGE_BASE
        kw["alpha_auto"] = False
    o = Optimizer(method, **kw)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        o.fit(A, y)
    flag = ""
    if "grid MINIMUM" in buf.getvalue():
        flag += "A@MIN"
    if method in ("LASSO", "ALASSO"):
        _al = getattr(o._model, "alphas_", None)
        if _al is not None and np.size(_al) > 1:
            _al = np.asarray(_al)
            flag += ("+" if flag else "") + "D%.1f" % float(
                np.log10(_al.max() / _al.min()))
    if method == "ALASSO":
        _w = getattr(o._model, "_weights", None)
        if _w is not None:
            flag += ("+" if flag else "") + "W%.2f" % float(
                np.std(np.log(np.maximum(np.asarray(_w), 1e-300))))
        _bf = getattr(o._model, "_beta0_floor", None)
        if _bf is not None:
            flag += ("+" if flag else "") + "F%.2f" % float(_bf)
    if method == "RIDGE":
        a = float(o._model.alpha_)
        if a <= RIDGE_BASE[0] * (1.0 + 1e-9):
            flag += ("+" if flag else "") + "R@lo"
        elif a >= RIDGE_BASE[-1] * (1.0 - 1e-9):
            flag += ("+" if flag else "") + "R@hi"
    coef = np.asarray(o.results["coef"], dtype=np.float64)
    alpha = o.results.get("alpha")
    return coef, flag, alpha, o._model


def cv_report(method, A, y, rows_per_config, seed, alpha, model):
    """Grouped K-fold CV: per-fold RMSE of the final fitted model.

    LASSO/ALASSO reuse the alpha-selection CV path already stored on the model
    (mse_path_, n_alphas x n_folds) at the selected alpha -- no re-fit.  The
    others do a manual grouped refit per fold at the fixed settings.
    """
    from pheasy.core.optimizer import Optimizer
    std = method in ("LASSO", "ALASSO", "RIDGE")
    mse_path = getattr(model, "mse_path_", None)
    if (mse_path is not None and np.ndim(mse_path) == 2
            and mse_path.shape[1] > 1 and method in ("LASSO", "ALASSO")):
        alphas = np.asarray(getattr(model, "alphas_", model.alphas))
        a = float(getattr(model, "alpha_", alphas[0]))
        idx = int(np.argmin(np.abs(alphas - a)))
        return [float(np.sqrt(v)) for v in mse_path[idx]]

    group_size = rows_per_config
    n = A.shape[0]
    n_conf = n // group_size
    k = min(FIT_KW["cv"], n_conf)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_conf)
    folds = [f for f in np.array_split(order, k) if len(f) > 0]
    rmses = []
    for held in folds:
        val = np.zeros(n, dtype=bool)
        for c in held:
            val[c * group_size:(c + 1) * group_size] = True
        tr = ~val
        kw = dict(FIT_KW, rand_seed=seed, standardize=std)
        if method in ("LASSO", "ALASSO", "RIDGE"):
            kw["alpha"] = [alpha]
            kw["alpha_auto"] = False
        o = Optimizer(method, **kw)
        with contextlib.redirect_stdout(io.StringIO()):
            o.fit(A[tr], y[tr])
        pred = A[val] @ o.results["coef"]
        rmses.append(float(np.sqrt(np.mean((pred - y[val]) ** 2))))
    return rmses


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("--method", required=True,
                    help="OLS / LASSO / ALASSO / RFE / RFE-OLS-TSQR / RIDGE")
    ap.add_argument("--n-configs", type=int, default=45)
    ap.add_argument("--rows-per-config", type=int, default=567,
                    help="每构型行数 = 3 x 超胞原子数 (默认 567)")
    ap.add_argument("--sm-dtype", default=os.environ.get("PHEASY_SM_DTYPE", "float64"),
                    choices=["float32", "float64"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cv-report", action="store_true",
                    help="额外输出每组 CV 折的 RMSE (分组, 按构型)")
    ap.add_argument("--write-coef", default=None,
                    help="可选: 把拟合系数存为 .npz (e.g. c.npz)")
    args = ap.parse_args()

    method = args.method.upper().replace("_", "-")
    if method == "RFE-TSQR":
        method = "RFE-OLS-TSQR"
    valid = {"OLS", "LASSO", "ALASSO", "RFE", "RFE-OLS-TSQR", "RIDGE"}
    if method not in valid:
        raise SystemExit("未知方法 %r; 可选: %s" % (args.method, sorted(valid)))

    SM, F = load_sm_f(args.data_dir, args.n_configs, args.rows_per_config,
                      args.sm_dtype)
    print("loaded SM %s F %s (dtype=%s, rows_per_config=%d)"
          % (SM.shape, F.shape, SM.dtype, args.rows_per_config), flush=True)

    coef, flag, alpha, fit_model = fit_method(method, SM, F)

    pred = SM @ coef
    err = pred - F
    rmse = float(np.sqrt(np.mean(err * err)))
    rel = float(np.linalg.norm(err) / np.linalg.norm(F))
    nnz = int(np.count_nonzero(coef))

    print()
    print("method     = %s" % method)
    print("nnz        = %d / %d" % (nnz, coef.size))
    print("alpha      = %s" % ("%.6e" % alpha if alpha is not None else "n/a"))
    print("rmse       = %.4e" % rmse)
    print("relL2      = %.4e" % rel)
    print("flag       = %s" % (flag or "-"))

    if args.cv_report:
        folds = cv_report(method, SM, F, args.rows_per_config, args.seed, alpha,
                          fit_model)
        print("cv_folds   = [" + ", ".join("%.4e" % r for r in folds) + "]")
        print("cv_mean    = %.4e" % float(np.mean(folds)))
        print("cv_std     = %.4e" % float(np.std(folds)))

    if args.write_coef:
        np.savez(args.write_coef, coef=coef, alpha=alpha, nnz=nnz,
                 rmse=rmse, rel_l2=rel, flag=flag, method=method)
        print("coef saved -> %s" % args.write_coef)


if __name__ == "__main__":
    main()
