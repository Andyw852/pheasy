#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""recompute_pairs.py -- recompute the paired relL2 rows after the RIDGE re-run.

New RIDGE per-fold relL2 comes from holdout_ridge_n{N}_s0.out; the (unchanged)
OLS/LASSO/ALASSO per-fold relL2 come from holdout_olal_n{N}_s0.out (n=45/24/9/6,
re-run this session) or the old holdout_n18.out / holdout_n12_s0.out (n=18/12).
All runs use seed 0 + the same n-splits, so folds are identical.

Pairs (matching holdout_eval.py convention; diff = ref - method, favor = #folds
with diff < 0, i.e. #folds where ref is better):
  RIDGE - ALASSO   (conclusion row: acceptance test is favor == 0)
  RIDGE - LASSO
  OLS   - RIDGE
Also prints R@lo/R@hi counts from the RIDGE flags.
"""
import os
import re
import sys

_LINE = re.compile(r"^\s+(\S+)\s+rmse=([0-9.eE+-]+)\s+relL2=([0-9.eE+-]+)")


def parse_per_fold(path, method):
    rel = []
    flags = []
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        for line in f:
            m = _LINE.match(line)
            if m and m.group(1) == method:
                rel.append(float(m.group(3)))
                fl = re.search(r"\b(R@lo|R@hi)\b", line)
                flags.append(fl.group(1) if fl else "")
    return rel, flags


def main():
    tiers = [(45, 5), (24, 5), (18, 5), (12, 3), (9, 3), (6, 3)]
    d = "MnIn2Se4_c7_local"
    print("%-4s %12s %12s %12s %12s %12s %8s" % (
        "n", "RIDGE_rel", "OLS_rel", "LASSO_rel", "ALASSO_rel",
        "RIDGE_edge", "R@lo/hi"))
    print("-" * 78)
    for n, k in tiers:
        ridge_path = os.path.join(d, "holdout_ridge_n%d_s0.out" % n)
        if n in (18, 12):
            ola_path = os.path.join(d, "holdout_n%d%s.out" % (n, "_s0" if n == 12 else ""))
        else:
            ola_path = os.path.join(d, "holdout_olal_n%d_s0.out" % n)
        r, rflags = parse_per_fold(ridge_path, "RIDGE")
        o, _ = parse_per_fold(ola_path, "OLS")
        la, _ = parse_per_fold(ola_path, "LASSO")
        al, _ = parse_per_fold(ola_path, "ALASSO")
        if not r or not o or not la or not al:
            print("%-4s MISSING (ridge=%s ola=%s)" % (n, ridge_path, ola_path))
            continue
        nf = min(len(r), len(o), len(la), len(al))
        r, o, la, al = r[:nf], o[:nf], la[:nf], al[:nf]
        rflags = rflags[:nf]
        nlo = sum(1 for f in rflags if f == "R@lo")
        nhi = sum(1 for f in rflags if f == "R@hi")
        print("%-4d %12.4e %12.4e %12.4e %12.4e %9s %4d/%d" % (
            n, sum(r) / nf, sum(o) / nf, sum(la) / nf, sum(al) / nf,
            "%d/%d" % (nlo, nhi), nlo, nf))
        print()
        print("  paired RIDGE - ALASSO : mean %.4e  std %.4e  (%d/%d folds favor RIDGE; all: %s)"
              % (sum(x - y for x, y in zip(r, al)) / nf,
                 (sum((x - y) ** 2 for x, y in zip(r, al)) / nf) ** 0.5,
                 sum(1 for x, y in zip(r, al) if x < y), nf,
                 " ".join("%.3e" % (x - y) for x, y in zip(r, al))))
        print("  paired RIDGE - LASSO : mean %.4e  std %.4e  (%d/%d folds favor RIDGE; all: %s)"
              % (sum(x - y for x, y in zip(r, la)) / nf,
                 (sum((x - y) ** 2 for x, y in zip(r, la)) / nf) ** 0.5,
                 sum(1 for x, y in zip(r, la) if x < y), nf,
                 " ".join("%.3e" % (x - y) for x, y in zip(r, la))))
        print("  paired OLS - RIDGE   : mean %.4e  std %.4e  (%d/%d folds favor OLS; all: %s)"
              % (sum(x - y for x, y in zip(o, r)) / nf,
                 (sum((x - y) ** 2 for x, y in zip(o, r)) / nf) ** 0.5,
                 sum(1 for x, y in zip(o, r) if x < y), nf,
                 " ".join("%.3e" % (x - y) for x, y in zip(o, r))))
        print()


if __name__ == "__main__":
    main()
