#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_scan.py —— 把 scan_methods.sh 的结果汇总成收敛表和图。

比较对象是 phi.npz 里的 Phi 向量（NS_full 展开后的完整 IFC，长度 =
1296 + 1836 = 3132，前 1296 是二阶、其余是三阶）。用它而不是 fc3.hdf5：
两者信息等价，但前者 30 KB、后者 1.5 GB。

误差定义（相对基准 = 构型数最多的那次 OLS）：
    rel_L2 = ||Phi - Phi_ref|| / ||Phi_ref||     分二阶 / 三阶块各算一次
另外报告 fc3max 的相对偏差，因为它对三阶的尾部最敏感。

用法：
    python3 analyze_scan.py scan/results
    python3 analyze_scan.py scan/results --ref OLS --plot conv.png
"""
import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np

N_FC2 = 1296          # 二阶 IFC 数（该体系；脚本会按实际长度自适应）


def load_all(d):
    pat = re.compile(r"^phi_(?P<m>.+)_n(?P<n>\d+)\.npz$")
    out = {}
    for fn in sorted(os.listdir(d)):
        mo = pat.match(fn)
        if not mo:
            continue
        phi = np.load(os.path.join(d, fn))["Phi"].ravel()
        out[(mo.group("m"), int(mo.group("n")))] = phi
    return out


def parse_log(d, m, n):
    """Pull rel_err / nnz / alpha out of the saved fit log, if present."""
    path = os.path.join(d, "fit_%s_n%d.log" % (m, n))
    info = {}
    if not os.path.exists(path):
        return info
    with open(path, errors="ignore") as fh:
        for line in fh:
            if "Relative error:" in line:
                info["rel_err"] = float(line.rsplit(":", 1)[1])
            elif "Non-zero IFC terms:" in line:
                info["nnz"] = int(line.rsplit(":", 1)[1])
            elif "Free IFC terms:" in line:
                info["p"] = int(line.rsplit(":", 1)[1])
            elif "alpha_opt:" in line:
                info["alpha"] = float(line.rsplit(":", 1)[1])
            elif "best CV RMSE:" in line:
                info["cv"] = float(line.rsplit(":", 1)[1].split()[0])
            elif "RMSE_CV:" in line:
                info["cv"] = float(line.rsplit(":", 1)[1].split()[0])
            elif "alphas tie at CV MSE" in line:
                info["tie"] = True
            elif "sits at the grid MINIMUM" in line:
                info["atmin"] = True
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("resultdir", nargs="?", default="scan/results")
    ap.add_argument("--ref", default="OLS", help="基准方法（默认 OLS）")
    ap.add_argument("--n-fc2", type=int, default=None,
                    help="二阶自由度数；默认从 %d 猜，或用 --n-fc2 指定" % N_FC2)
    ap.add_argument("--plot", default=None, help="输出收敛图（png/pdf）")
    ap.add_argument("--csv", default=None, help="输出 CSV")
    args = ap.parse_args()

    data = load_all(args.resultdir)
    if not data:
        sys.exit("在 %s 里没找到 phi_*_n*.npz" % args.resultdir)

    methods = sorted({k[0] for k in data})
    ndatas = sorted({k[1] for k in data})
    n_ref = max(ndatas)
    if (args.ref, n_ref) not in data:
        sys.exit("缺少基准 phi_%s_n%d.npz" % (args.ref, n_ref))
    ref = data[(args.ref, n_ref)]

    n2 = args.n_fc2 if args.n_fc2 else (N_FC2 if len(ref) > N_FC2 else len(ref))
    ref2, ref3 = ref[:n2], ref[n2:]
    print("基准: %s @ ndata=%d   |Phi|=%d  (二阶 %d + 三阶 %d)"
          % (args.ref, n_ref, len(ref), n2, len(ref) - n2))
    print()

    rows = []
    for m in methods:
        for n in ndatas:
            phi = data.get((m, n))
            if phi is None:
                continue
            e2 = np.linalg.norm(phi[:n2] - ref2) / np.linalg.norm(ref2)
            e3 = (np.linalg.norm(phi[n2:] - ref3) / np.linalg.norm(ref3)
                  if len(ref3) else np.nan)
            a3 = np.abs(phi[n2:]).max() if len(ref3) else np.nan
            r3 = np.abs(ref3).max() if len(ref3) else np.nan
            info = parse_log(args.resultdir, m, n)
            _p = info.get("p")
            info["nnz_p"] = (float(info["nnz"]) / _p
                             if (_p and info.get("nnz")) else np.nan)
            rows.append(dict(method=m, ndata=n, relL2_fc2=e2, relL2_fc3=e3,
                             fc3max=a3, dev3=(a3 - r3) / r3 * 100 if r3 else np.nan,
                             **info))

    hdr = ("%-14s %5s %11s %11s %9s %9s %10s %6s %6s %10s %s"
           % ("method", "ndata", "relL2_fc2", "relL2_fc3", "fc3max", "dev3%",
              "rel_err", "nnz", "nnz/p", "RMSE_CV", "note"))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        _note = []
        if r.get("tie"):
            _note.append("TIE!")
        if r.get("atmin"):
            _note.append("A@MIN!")
        note = "+".join(_note)
        _nnzp = ("%.2f" % r["nnz_p"]) if (r.get("nnz_p") is not None
                                          and np.isfinite(r["nnz_p"])) else "-"
        _cv = ("%.4e" % r["cv"]) if (r.get("cv") is not None
                                     and np.isfinite(r["cv"])) else "-"
        print("%-14s %5d %11.3e %11.3e %9.4f %9.3f %10.6f %6s %6s %10s %s"
              % (r["method"], r["ndata"], r["relL2_fc2"], r["relL2_fc3"],
                 r["fc3max"], r["dev3"], r.get("rel_err", float("nan")),
                 r.get("nnz", "-"), _nnzp, _cv, note))

    # [FIX P43] right below the table: RMSE_CV is the L1 path's grouped-CV RMSE
    # (pre-debias); the delivered IFC is a debiased (OLS) refit with no holdout
    # number. Skip for OLS-only scans where the column is all "-".
    if any(r.get("cv") is not None for r in rows):
        print("\nnote: RMSE_CV is the L1 path's grouped-CV RMSE (pre-debias); the "
              "delivered IFC is a debiased OLS refit with no holdout number.")

    if args.csv:
        import csv
        keys = ["method", "ndata", "relL2_fc2", "relL2_fc3", "fc3max", "dev3",
                "rel_err", "nnz", "cv", "alpha", "nnz_p", "tie", "atmin"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print("\nCSV -> %s" % args.csv)

    # 达到 1% / 0.1% 三阶精度所需的最少构型数
    print("\n达到给定三阶精度所需的最少构型数：")
    print("%-14s %10s %10s" % ("method", "<1e-2", "<1e-3"))
    for m in methods:
        pts = sorted((r["ndata"], r["relL2_fc3"]) for r in rows if r["method"] == m)
        need = {}
        for thr, lab in ((1e-2, "<1e-2"), (1e-3, "<1e-3")):
            hit = [n for n, e in pts if np.isfinite(e) and e < thr]
            need[lab] = str(min(hit)) if hit else "-"
        print("%-14s %10s %10s" % (m, need["<1e-2"], need["<1e-3"]))

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            sys.exit("需要 matplotlib 才能出图")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
        for m in methods:
            pts = sorted((r["ndata"], r["relL2_fc2"], r["relL2_fc3"])
                         for r in rows if r["method"] == m)
            ns = [p[0] for p in pts]
            axes[0].plot(ns, [p[1] for p in pts], "o-", label=m, ms=4)
            axes[1].plot(ns, [p[2] for p in pts], "o-", label=m, ms=4)
        for ax, t in zip(axes, ("2nd order", "3rd order")):
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("number of configurations")
            ax.set_ylabel(r"$\|\Phi-\Phi_{\rm ref}\|/\|\Phi_{\rm ref}\|$")
            ax.set_title(t)
            ax.grid(True, which="both", alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=200)
        print("图 -> %s" % args.plot)


if __name__ == "__main__":
    main()
