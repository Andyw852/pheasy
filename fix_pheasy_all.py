#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fix_pheasy_all.py —— pheasy 一键修复脚本

用法（在 pheasy 包目录下运行，即含 run_pheasy.py / core/optimizer.py 的那层）：

    python3 fix_pheasy_all.py              # 应用全部修复
    python3 fix_pheasy_all.py --dry-run    # 只检查、不写盘
    python3 fix_pheasy_all.py --list       # 列出所有补丁
    python3 fix_pheasy_all.py --only P01 P03
    python3 fix_pheasy_all.py --revert     # 从本次备份回滚

每个被改动的文件在第一次写入前备份为 <file>.bak_fixall_<时间戳>。
所有补丁幂等：重复运行会报 SKIP(已应用) 而不是重复插入。
"""
import argparse
import datetime
import os
import shutil
import sys

STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
BAK_SUFFIX = ".bak_fixall_" + STAMP

PATCHES = []          # (id, file, title, fn)
_backed_up = set()
_cache = {}


def patch(pid, path, title):
    def deco(fn):
        PATCHES.append((pid, path, title, fn))
        return fn
    return deco


def read(path):
    if path not in _cache:
        with open(path, "r", encoding="utf-8") as f:
            _cache[path] = f.read()
    return _cache[path]


def write(path, text, dry):
    if dry:
        _cache[path] = text
        return
    if path not in _backed_up and os.path.exists(path):
        shutil.copy2(path, path + BAK_SUFFIX)
        _backed_up.add(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    _cache[path] = text


def sub1(path, old, new, dry, marker=None):
    """Replace `old` exactly once. Returns 'OK' / 'SKIP' / 'FAIL: ...'."""
    s = read(path)
    if marker and marker in s:
        return "SKIP (already applied)"
    n = s.count(old)
    if n == 0:
        return "FAIL (anchor not found)"
    if n > 1:
        return "FAIL (anchor matches %d times)" % n
    write(path, s.replace(old, new, 1), dry)
    return "OK"


RP = "run_pheasy.py"
OPT = os.path.join("core", "optimizer.py")
UTL = os.path.join("core", "utilities.py")
BIO = "basic_io.py"


# ---------------------------------------------------------------- P01
@patch("P01", RP, "-l RFE / -l RFE-OLS-TSQR 现在能触发 sparse/two-level SM 路径")
def p01(dry):
    old = """                    _is_rfe = _os_p.environ.get('PHEASY_USE_RFE','').lower() in ('1','true','yes')
                    _is_rfe_tsqr = _os_p.environ.get('PHEASY_USE_RFE_TSQR','').lower() in ('1','true','yes')"""
    new = """                    # [FIX P01] 方法判定必须以 settings.MODEL 为准。
                    # 旧代码只认 PHEASY_USE_* 环境变量, 而新 CLI (-l RFE /
                    # -l RFE-OLS-TSQR) 根本不设这些变量 -> _use_sparse 恒为 False,
                    # 于是 RFE 总是走 dense 分支物化 sm_dense.npy (大 cutoff 下数百 GB),
                    # 所有 sparse / two-level 的内存优化对新接口全部失效。
                    _model_up = settings.MODEL.upper().replace('_', '-')
                    _is_rfe = (_model_up == 'RFE') or (
                        _os_p.environ.get('PHEASY_USE_RFE','').lower() in ('1','true','yes'))
                    _is_rfe_tsqr = (_model_up in ('RFE-OLS-TSQR', 'RFE-TSQR')) or (
                        _os_p.environ.get('PHEASY_USE_RFE_TSQR','').lower() in ('1','true','yes'))"""
    return sub1(RP, old, new, dry, marker="[FIX P01]")


# ---------------------------------------------------------------- P02
@patch("P02", RP, "sm_dense.npy 缓存校验来源指纹（不再只看 shape/dtype）")
def p02(dry):
    old = """                        _hit = (not _force) and _os_p.path.exists(_SM_F)
                        if _hit:
                            try:
                                _chk = _np_p.load(_SM_F, mmap_mode='r')
                                if _chk.shape != _exp or _chk.dtype != _np_p.float32:
                                    print(f'[SM-cache] mismatch {_chk.shape}/{_chk.dtype} vs {_exp}/float32, rebuild', flush=True)
                                    _hit = False
                                    del _chk
                            except Exception as _e:
                                print(f'[SM-cache] read fail: {_e}, rebuild', flush=True)
                                _hit = False"""
    new = """                        # [FIX P02] 旧缓存只按 (shape, dtype) 命中, 换一份位移数据
                        # 或 cp -r 过来的目录会静默复用陈旧 SM。改为额外校验来源指纹:
                        # sm_prime.npz 的 size+mtime、SM_prime 的 shape/nnz、NS 的 shape。
                        _SM_META = _SM_F + '.meta.json'
                        import json as _json_p

                        def _sm_fingerprint():
                            _src = getattr(self, 'SensingMatrixFile', 'sm_prime.npz')
                            try:
                                _st = _os_p.stat(_src)
                                _src_sig = [_src, int(_st.st_size), int(_st.st_mtime)]
                            except OSError:
                                _src_sig = [_src, -1, -1]
                            return {
                                'src': _src_sig,
                                'smp_shape': list(SM_prime.shape),
                                'smp_nnz': int(getattr(SM_prime, 'nnz', -1)),
                                'ns_shape': list(self.NS_full.shape),
                                'dtype': _np_p.dtype(_sm_dtype()).name,
                                'ndata': int(settings.NDATA),
                            }

                        _fp_now = _sm_fingerprint()
                        _hit = (not _force) and _os_p.path.exists(_SM_F)
                        if _hit:
                            try:
                                _chk = _np_p.load(_SM_F, mmap_mode='r')
                                if _chk.shape != _exp:
                                    print(f'[SM-cache] shape mismatch {_chk.shape} vs {_exp}, rebuild', flush=True)
                                    _hit = False
                                del _chk
                            except Exception as _e:
                                print(f'[SM-cache] read fail: {_e}, rebuild', flush=True)
                                _hit = False
                        if _hit:
                            try:
                                with open(_SM_META) as _fh:
                                    _fp_old = _json_p.load(_fh)
                            except Exception:
                                _fp_old = None
                            if _fp_old != _fp_now:
                                if _fp_old is None:
                                    print('[SM-cache] no fingerprint sidecar '
                                          f'({_SM_META}), rebuild to be safe', flush=True)
                                else:
                                    _diff = [k for k in _fp_now
                                             if _fp_old.get(k) != _fp_now[k]]
                                    print(f'[SM-cache] STALE: source changed {_diff}, rebuild', flush=True)
                                _hit = False"""
    r = sub1(RP, old, new, dry, marker="[FIX P02]")
    if r != "OK":
        return r
    old2 = """                                _np_p.save(_SM_F, SM)
                                print(f'[SM-cache] saved in {_ts.time()-_t0:.1f}s', flush=True)"""
    new2 = """                                _np_p.save(_SM_F, SM)
                                with open(_SM_META, 'w') as _fh:      # [FIX P02]
                                    _json_p.dump(_fp_now, _fh)
                                print(f'[SM-cache] saved in {_ts.time()-_t0:.1f}s', flush=True)"""
    return sub1(RP, old2, new2, dry)


# ---------------------------------------------------------------- P03
@patch("P03", RP, "-l RIDGE 不再崩；未知方法给出可读报错；补 TSQR 汇总分支")
def p03(dry):
    old = """            elif settings.MODEL.upper() == "RFE-OLS-TSQR":
                logger.info(
                    "Fitting force constants via strict OLS + RFE "
                    "(tall-skinny Householder QR solver, scale-invariant importance).")
            else:
                logger.error(
                    "Unknown linear model for fitting force constants, {}".format(
                        settings.MODEL
                    )
                )
                raise ValueError"""
    new = """            elif settings.MODEL.upper() in ("RFE-OLS-TSQR", "RFE_TSQR", "RFE-TSQR"):
                logger.info(
                    "Fitting force constants via strict OLS + RFE "
                    "(tall-skinny Householder QR solver, scale-invariant importance).")
            elif settings.MODEL.upper() == "RIDGE":
                # [FIX P03] RIDGE 在 argparse choices 里、Optimizer 也实现了,
                # 但这里原来没有分支 -> 落到 else 直接 raise ValueError(空消息)。
                logger.info("Fitting force constants via Ridge regression (L2, CV alpha).")
            else:
                _msg = (
                    "Unknown linear model for fitting force constants: {!r}. "
                    "Expected one of OLS, LASSO, ALASSO, RFE, RFE-OLS-TSQR, RIDGE."
                ).format(settings.MODEL)
                logger.error(_msg)
                raise ValueError(_msg)"""
    r = sub1(RP, old, new, dry, marker="[FIX P03]")
    if r != "OK":
        return r
    old2 = """            elif settings.MODEL.upper() == "RFE":
                logger.info("- RFE finished after {} rounds.".format(fit_results["n_iter"]))
                logger.info("- ridge_alpha: {}".format(fit_results["alpha"]))
                logger.info("- best CV RMSE: {} eV/A".format(fit_metrics["rmse_path_mean"]))"""
    new2 = """            elif settings.MODEL.upper() in ("RFE", "RFE-OLS-TSQR", "RFE_TSQR", "RFE-TSQR"):
                # [FIX P05] TSQR 原来没有汇总分支, 明明算了 CV 却不打印。
                logger.info("- RFE finished after {} rounds.".format(fit_results["n_iter"]))
                logger.info("- ridge_alpha: {}".format(fit_results["alpha"]))
                logger.info("- best CV RMSE: {} eV/A".format(fit_metrics["rmse_path_mean"]))
                logger.info("- selected features: {} of {}".format(
                    int(np.count_nonzero(fit_results["coef"])),
                    fit_results["coef"].shape[0]))
            elif settings.MODEL.upper() == "RIDGE":
                logger.info("- alpha_opt: {}".format(fit_results.get("alpha")))"""
    return sub1(RP, old2, new2, dry)


# ---------------------------------------------------------------- P04
@patch("P04", RP, "-d 并行 worker 不再静默 OOM：按内存限并发 + 失败自动串行回退")
def p04(dry):
    old = """                    if _n_jobs != 1:
                        logger.info(
                            "- Parallel sensing matrix construction: {} workers.".format(_n_jobs)
                        )
                        _results = Parallel(n_jobs=_n_jobs)(
                            delayed(_build_sensing_matrix_sparse)(self.CS_full, _u)
                            for _u in _u_list
                        )
                        sensing_mat_list.extend(_results)
                    else:
                        for _u in _u_list:
                            sensing_mat = build_sensing_matrix(self.CS_full, _u)
                            sensing_mat_list.append(sensing_mat)"""
    new = """                    # [FIX P04] joblib 会把整个 CS_full pickle 给每个 worker
                    # (大体系可达 GB 级/进程)。原代码既不限并发也不捕获 worker
                    # 被 OOM-killer 干掉的情况, 日志停在 "N workers." 之后什么
                    # 都没有, 看起来像卡死。这里: 先按可用内存压 n_jobs, 再对
                    # worker 崩溃做串行回退。
                    _n_jobs = _cap_n_jobs_by_memory(_n_jobs, self.CS_full)
                    if _n_jobs != 1:
                        logger.info(
                            "- Parallel sensing matrix construction: {} workers.".format(_n_jobs)
                        )
                        try:
                            _results = Parallel(n_jobs=_n_jobs)(
                                delayed(_build_sensing_matrix_sparse)(self.CS_full, _u)
                                for _u in _u_list
                            )
                            sensing_mat_list.extend(_results)
                        except Exception as _e:
                            logger.error(
                                "- Parallel sensing-matrix workers failed (%s: %s). "
                                "This is almost always the OOM killer: every worker "
                                "holds its own copy of the cluster space. Falling back "
                                "to serial construction; set PHEASY_N_JOBS=1 to skip "
                                "this attempt next time.",
                                type(_e).__name__, _e,
                            )
                            sensing_mat_list.clear()
                            _n_jobs = 1
                    if _n_jobs == 1:
                        for _i, _u in enumerate(_u_list):
                            sensing_mat = build_sensing_matrix(self.CS_full, _u)
                            sensing_mat_list.append(sensing_mat)
                            if (_i + 1) % 10 == 0 or (_i + 1) == len(_u_list):
                                logger.info("- Sensing matrix: {} / {} configurations."
                                            .format(_i + 1, len(_u_list)))"""
    r = sub1(RP, old, new, dry, marker="[FIX P04]")
    if r != "OK":
        return r
    helper = '''

def _cap_n_jobs_by_memory(n_jobs, cs_full):
    """[FIX P04] Cap joblib workers so the pickled cluster space fits in RAM.

    Each worker receives its own copy of ``cs_full``; on a memory-tight node
    that silently triggers the OOM killer and joblib reports nothing useful.
    Override the safety margin with ``PHEASY_SM_MEM_FRACTION`` (default 0.6)
    or bypass entirely with ``PHEASY_N_JOBS_NO_CAP=1``.
    """
    import os as _os
    import pickle as _pk

    if n_jobs in (0, 1):
        return 1
    if _os.environ.get("PHEASY_N_JOBS_NO_CAP", "").lower() in ("1", "true", "yes"):
        return n_jobs
    try:
        avail = _os.sysconf("SC_AVPHYS_PAGES") * _os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return n_jobs
    try:
        per_worker = len(_pk.dumps(cs_full, protocol=_pk.HIGHEST_PROTOCOL))
    except Exception:
        return n_jobs
    if per_worker <= 0:
        return n_jobs
    frac = float(_os.environ.get("PHEASY_SM_MEM_FRACTION", "0.6"))
    # x3: pickle buffer in the parent + unpickled object in the child + slack
    budget = max(1, int(avail * frac / (per_worker * 3.0)))
    n_cpu = _os.cpu_count() or 1
    capped = max(1, min(n_jobs if n_jobs > 0 else n_cpu, budget, n_cpu))
    if capped < n_jobs:
        logger.warning(
            "- PHEASY_N_JOBS=%d would need ~%.1f GB for the cluster-space copies "
            "but only %.1f GB is available; capping to %d worker(s).",
            n_jobs, n_jobs * per_worker * 3.0 / 1e9, avail / 1e9, capped,
        )
    return capped

'''
    s = read(RP)
    anchor = "\n\nclass WorkFlow(object):"
    if anchor not in s:
        return "FAIL (WorkFlow anchor not found)"
    write(RP, s.replace(anchor, "\n" + helper + "\nclass WorkFlow(object):", 1), dry)
    return "OK"


# ---------------------------------------------------------------- P06
@patch("P06", RP, "SM/FM 精度统一由 PHEASY_SM_DTYPE 决定（去掉硬编码 float32）")
def p06(dry):
    reps = [
        ("""                    if _b.dtype != np.float32:
                        _b = _b.astype(_sm_dtype())
                else:
                    # 串行 fallback 路径仍可能给 dense
                    _b = spmat.csr_matrix(_b, dtype=np.float32)""",
         """                    if _b.dtype != np.dtype(_sm_dtype()):   # [FIX P06]
                        _b = _b.astype(_sm_dtype())
                else:
                    # 串行 fallback 路径仍可能给 dense
                    _b = spmat.csr_matrix(_b, dtype=_sm_dtype())"""),
        ("""            if SM_prime.dtype != np.float32:
                SM_prime = SM_prime.astype(_sm_dtype())""",
         """            if SM_prime.dtype != np.dtype(_sm_dtype()):     # [FIX P06]
                SM_prime = SM_prime.astype(_sm_dtype())"""),
        ("""            logger.info(
                f'SM_prime kept sparse {SM_prime.dtype} CSR: shape={SM_prime.shape}, '
                f'nnz={SM_prime.nnz}, mem={_mem:.2f} GB'
            )""",
         """            logger.info(
                f'SM_prime kept sparse {SM_prime.dtype} CSR: shape={SM_prime.shape}, '
                f'nnz={SM_prime.nnz}, mem={_mem:.2f} GB'
            )
            # [FIX P06] SM / NS / FM now all follow PHEASY_SM_DTYPE (default
            # float64). float32 halves the memory at the cost of lsmr only
            # reaching ~1e-7 relative accuracy (istop=5).
            if _mem > float(os.environ.get('PHEASY_SM_MEM_HINT_GB', '8')):
                logger.warning(
                    '- SM_prime is %.1f GB in %s. Set PHEASY_SM_DTYPE=float32 '
                    'to halve it (all matrices stay consistent).',
                    _mem, SM_prime.dtype)"""),
        ("""                        _ns_sp = _ns_sp.astype(_np_p.float32)
                        if not _sp_p.issparse(SM_prime):
                            SM_prime = _sp_p.csr_matrix(SM_prime)
                        SM_prime = SM_prime.astype(_np_p.float32)""",
         """                        _ns_sp = _ns_sp.astype(_sm_dtype())     # [FIX P06]
                        if not _sp_p.issparse(SM_prime):
                            SM_prime = _sp_p.csr_matrix(SM_prime)
                        SM_prime = SM_prime.astype(_sm_dtype())"""),
        ("""                            SM = _np_p.empty((SM_prime.shape[0], _NS.shape[1]), dtype=_np_p.float32)""",
         """                            SM = _np_p.empty((SM_prime.shape[0], _NS.shape[1]),
                                             dtype=_sm_dtype())     # [FIX P06]"""),
    ]
    if "[FIX P06]" in read(RP):
        return "SKIP (already applied)"
    for old, new in reps:
        r = sub1(RP, old, new, dry)
        if r != "OK":
            return r + "  <- while replacing %r" % old.strip().splitlines()[0]
    return "OK"


# ---------------------------------------------------------------- P07
@patch("P07", OPT, "_scale_columns 不再静默把 LinearOperator 稠密化（预算+清晰报错）")
def p07(dry):
    old = '''def _scale_columns(A, w):
    """Column-scaled copy of A: A[:, j] / w[j] (dense or sparse)."""
    inv_w = 1.0 / np.asarray(w, dtype=np.float64)
    if sp.issparse(A):
        return A.astype(np.float64).multiply(inv_w[None, :]).tocsr()
    A64 = _to_dense_f64(A)'''
    new = '''def _scale_columns(A, w):
    """Column-scaled copy of A: A[:, j] / w[j] (dense or sparse).

    [FIX P07] A LinearOperator (e.g. TwoLevelSM) has no columns to scale, so it
    must be materialized -- sklearn's coordinate descent cannot consume an
    operator anyway.  The old code did that silently via ``A @ np.eye(n)``,
    which quietly undoes the whole two-level memory optimization and OOM-kills
    the job on large cutoffs.  Now we say so, and refuse above a budget.
    """
    inv_w = 1.0 / np.asarray(w, dtype=np.float64)
    if sp.issparse(A):
        return A.astype(np.float64).multiply(inv_w[None, :]).tocsr()
    if _is_linear_operator(A):
        need = float(A.shape[0]) * float(A.shape[1]) * 8.0
        budget = float(os.environ.get("PHEASY_MAX_DENSE_GB", "16")) * 1e9
        msg = ("--std / LASSO / ALASSO / RIDGE need a materialized design "
               "matrix, so the two-level operator %s must be densified "
               "(%.1f GB in float64)." % (str(A.shape), need / 1e9))
        if need > budget:
            raise MemoryError(
                msg + " That exceeds PHEASY_MAX_DENSE_GB=%.1f GB. Use -l OLS or "
                "-l RFE (which stream the operator), drop --std, or raise "
                "PHEASY_MAX_DENSE_GB if you really have the RAM."
                % (budget / 1e9))
        print("[optimizer] WARNING: " + msg, flush=True)
    A64 = _to_dense_f64(A)'''
    return sub1(OPT, old, new, dry, marker="[FIX P07]")


# ---------------------------------------------------------------- P08
@patch("P08", OPT, "_debias 对 LinearOperator 不再静默跳过（改为告警）")
def p08(dry):
    old = """        if _is_linear_operator(A):
            # no cheap column slicing for a LinearOperator; skip debias safely
            return coef"""
    new = """        if _is_linear_operator(A):
            # [FIX P08] no cheap column slicing for a LinearOperator. Skipping
            # debias silently is dangerous: the L1 shrinkage bias is exactly
            # what makes LASSO force constants come out ~10% too small.
            print("[optimizer] WARNING: LASSO debias skipped -- the design "
                  "matrix is a LinearOperator and cannot be column-sliced. "
                  "The returned force constants still carry the L1 shrinkage "
                  "bias. Set PHEASY_OLS_TWOLEVEL=0 / PHEASY_LASSO_SPARSE=0 to "
                  "get a sliceable matrix.", flush=True)
            return coef"""
    return sub1(OPT, old, new, dry, marker="[FIX P08]")


# ---------------------------------------------------------------- P09
@patch("P09", OPT, "RFE/TSQR 构造参数真正生效 + TSQR 改为真正的分块 tall-skinny QR")
def p09(dry):
    if "[FIX P09]" in read(OPT):
        return "SKIP (already applied)"

    # 9a: blocked, Q-less TSQR solver
    old = '''def _solve_qr(A, y):
    """OLS via tall-skinny QR (Householder QR + triangular solve).

    Numerically stable for full-column-rank matrices. Sparse / LinearOperator
    input falls back to LSQR (scipy has no sparse QR least-squares driver; LSQR
    is a Golub-Kahan bidiagonalization, QR-like, and memory efficient).
    """
    if _is_linear_operator(A):
        return _solve_sparse_lsqr(A, y)
    if sp.issparse(A) and not _should_densify_sparse(A):
        return _solve_sparse_lsqr(A, y)
    A64 = _to_dense_f64(A)
    y64 = np.asarray(y, dtype=np.float64).ravel()
    Q, R = spla.qr(A64, mode="economic", check_finite=False)
    diag = np.abs(np.diag(R))
    if diag.size == 0:'''
    new = '''def _tsqr_qless(A, y, block_rows=40000, diag_floor=1e-12):
    """[FIX P09] Q-less tall-skinny QR least squares, streamed by row blocks.

    Maintains only R (n x n) and z = Q^T y (n), never a full Q or a full copy
    of A, so peak memory is O(n^2 + block_rows * n) instead of O(m * n).  This
    is what "RFE-OLS-TSQR" advertises; the previous implementation just called
    a plain dense ``scipy.linalg.qr`` on the whole matrix.

    Returns (coef, rank_ok, cond_estimate).
    """
    m, n = A.shape
    y64 = np.asarray(y, dtype=np.float64).ravel()
    block_rows = max(int(block_rows), n + 1)
    R = None
    z = None
    for i0 in range(0, m, block_rows):
        i1 = min(i0 + block_rows, m)
        blk = A[i0:i1]
        blk = np.asarray(blk.toarray() if sp.issparse(blk) else blk, dtype=np.float64)
        rhs = y64[i0:i1]
        if R is None:
            M, b = blk, rhs
        else:
            M = np.vstack([R, blk])
            b = np.concatenate([z, rhs])
        Qb, Rb = spla.qr(M, mode="economic", check_finite=False)
        R = Rb
        z = Qb.T @ b
        del Qb, M, b
    diag = np.abs(np.diag(R))
    dmax = float(diag.max()) if diag.size else 0.0
    dmin = float(diag.min()) if diag.size else 0.0
    cond = (dmax / dmin) if dmin > 0 else np.inf
    if dmax == 0.0 or dmin <= diag_floor * max(dmax, 1.0):
        return None, False, cond
    coef = spla.solve_triangular(R, z, lower=False, check_finite=False)
    return np.asarray(coef, dtype=np.float64), True, cond


def _solve_qr(A, y, block_rows=None, diag_floor=1e-12):
    """OLS via tall-skinny QR (Householder QR + triangular solve).

    Numerically stable for full-column-rank matrices. Sparse / LinearOperator
    input falls back to LSQR (scipy has no sparse QR least-squares driver; LSQR
    is a Golub-Kahan bidiagonalization, QR-like, and memory efficient).

    [FIX P09] ``block_rows`` now actually streams the factorization instead of
    being an ignored constructor argument; ``diag_floor`` guards the triangular
    solve against a rank-deficient R.
    """
    if _is_linear_operator(A):
        return _solve_sparse_lsqr(A, y)
    if sp.issparse(A) and not _should_densify_sparse(A):
        return _solve_sparse_lsqr(A, y)
    if block_rows and A.shape[0] > int(block_rows):
        coef, ok, _cond = _tsqr_qless(A, y, block_rows, diag_floor)
        if ok:
            return coef
        return _solve_lstsq(A, y)     # rank deficient -> SVD
    A64 = _to_dense_f64(A)
    y64 = np.asarray(y, dtype=np.float64).ravel()
    Q, R = spla.qr(A64, mode="economic", check_finite=False)
    diag = np.abs(np.diag(R))
    if diag.size == 0:'''
    r = sub1(OPT, old, new, dry)
    if r != "OK":
        return r + " (9a)"

    # 9b: _solve_subset honours lsmr_* / block_rows / diag_floor
    old = '''def _solve_subset(A, y, row_idx, col_idx, ridge_alpha=0.0, qr=False):
    """Solve min ||A[row_idx][:, col_idx] x - y[row_idx]||^2 (+ optional ridge)."""
    y_sub = np.asarray(y, dtype=np.float64).ravel()
    if row_idx is not None:
        y_sub = y_sub[row_idx]
    if _is_linear_operator(A):
        # LSMR on a masked operator: no materialization, memory ~O(n_features).
        op = _make_masked_op(A, row_idx, col_idx)
        n = len(col_idx)
        atol = float(os.environ.get("PHEASY_LSQR_ATOL", "1e-8"))
        btol = float(os.environ.get("PHEASY_LSQR_BTOL", "1e-8"))
        maxiter = int(os.environ.get("PHEASY_LSQR_MAXITER", "5000"))'''
    new = '''def _solve_subset(A, y, row_idx, col_idx, ridge_alpha=0.0, qr=False,
                  lsmr_atol=None, lsmr_btol=None, lsmr_maxiter=None,
                  block_rows=None, diag_floor=1e-12):
    """Solve min ||A[row_idx][:, col_idx] x - y[row_idx]||^2 (+ optional ridge).

    [FIX P09] the lsmr_* / block_rows / diag_floor knobs are threaded through
    from the estimator instead of being silently dropped on the floor.
    """
    y_sub = np.asarray(y, dtype=np.float64).ravel()
    if row_idx is not None:
        y_sub = y_sub[row_idx]
    if _is_linear_operator(A):
        # LSMR on a masked operator: no materialization, memory ~O(n_features).
        op = _make_masked_op(A, row_idx, col_idx)
        n = len(col_idx)
        atol = float(os.environ.get("PHEASY_LSQR_ATOL", str(
            lsmr_atol if lsmr_atol is not None else 1e-8)))
        btol = float(os.environ.get("PHEASY_LSQR_BTOL", str(
            lsmr_btol if lsmr_btol is not None else 1e-8)))
        maxiter = int(os.environ.get("PHEASY_LSQR_MAXITER", str(
            lsmr_maxiter if lsmr_maxiter is not None else 5000)))'''
    r = sub1(OPT, old, new, dry)
    if r != "OK":
        return r + " (9b)"

    old = """    if qr:
        return _solve_qr(A_sub, y_sub)
    return _solve_lstsq(A_sub, y_sub)"""
    new = """    if qr:
        return _solve_qr(A_sub, y_sub, block_rows=block_rows,
                         diag_floor=diag_floor)
    return _solve_lstsq(A_sub, y_sub)"""
    r = sub1(OPT, old, new, dry)
    if r != "OK":
        return r + " (9c)"

    # 9d: _RFECVBase accepts & uses the knobs
    old = '''    def __init__(self, step=0.1, cv=5, min_features=1, n_jobs=-1,
                 verbose=False, random_state=None, solver="lstsq", ridge_alpha=0.0):
        self.step = float(step)
        self.cv = int(cv)
        self.min_features = int(min_features)
        self.n_jobs = int(n_jobs)
        self.verbose = verbose
        self.random_state = random_state
        self.ridge_alpha = float(ridge_alpha)
        self._solver_name = solver'''
    new = '''    def __init__(self, step=0.1, cv=5, min_features=1, n_jobs=-1,
                 verbose=False, random_state=None, solver="lstsq", ridge_alpha=0.0,
                 patience=5, lsmr_maxiter=5000, lsmr_atol=1e-8, lsmr_btol=1e-8,
                 block_rows=None, diag_floor=1e-12):
        self.step = float(step)
        self.cv = int(cv)
        self.min_features = int(min_features)
        self.n_jobs = int(n_jobs)
        self.verbose = verbose
        self.random_state = random_state
        self.ridge_alpha = float(ridge_alpha)
        self._solver_name = solver
        # [FIX P09] previously accepted-and-ignored constructor arguments
        self.patience = int(patience)
        self.lsmr_maxiter = int(lsmr_maxiter)
        self.lsmr_atol = float(lsmr_atol)
        self.lsmr_btol = float(lsmr_btol)
        self.block_rows = None if block_rows is None else int(block_rows)
        self.diag_floor = float(diag_floor)'''
    r = sub1(OPT, old, new, dry)
    if r != "OK":
        return r + " (9d)"

    old = """        def solve(col_idx, row_idx=None):
            return _solve_subset(A, y, row_idx, col_idx,
                                 self.ridge_alpha, _qr)"""
    new = """        def solve(col_idx, row_idx=None):
            return _solve_subset(A, y, row_idx, col_idx,
                                 self.ridge_alpha, _qr,
                                 lsmr_atol=self.lsmr_atol,
                                 lsmr_btol=self.lsmr_btol,
                                 lsmr_maxiter=self.lsmr_maxiter,
                                 block_rows=self.block_rows,
                                 diag_floor=self.diag_floor)"""
    r = sub1(OPT, old, new, dry)
    if r != "OK":
        return r + " (9e)"

    old = """        patience = int(os.environ.get("PHEASY_RFE_PATIENCE", "5"))"""
    new = """        # [FIX P09] constructor value is the default; env var is an override
        patience = int(os.environ.get("PHEASY_RFE_PATIENCE", str(self.patience)))"""
    r = sub1(OPT, old, new, dry)
    if r != "OK":
        return r + " (9f)"

    # 9g: subclasses forward everything
    old = '''class PheasyRFECV(_RFECVBase):
    """RFE with an OLS (optionally ridge-regularized) base estimator."""

    def __init__(self, step=0.1, cv=5, ridge_alpha=0.0, lsmr_maxiter=3000,
                 lsmr_atol=1e-8, lsmr_btol=1e-8, n_jobs=-1, min_features=1,
                 verbose=True, random_state=None):
        super().__init__(step=step, cv=cv, min_features=min_features, n_jobs=n_jobs,
                         verbose=verbose, random_state=random_state,
                         solver="lstsq", ridge_alpha=ridge_alpha)


class PheasyRFE_OLS_TSQR(_RFECVBase):
    """RFE with a strict OLS base estimator solved via QR (TSQR-flavored)."""

    def __init__(self, step=0.05, patience=5, min_features=100, block_rows=40000,
                 recalibrate=4, diag_floor=1e-12, cv=5, verbose=True,
                 random_state=None, n_jobs=-1):
        super().__init__(step=step, cv=cv, min_features=min_features, n_jobs=n_jobs,
                         verbose=verbose, random_state=random_state,
                         solver="qr", ridge_alpha=0.0)'''
    new = '''class PheasyRFECV(_RFECVBase):
    """RFE with an OLS (optionally ridge-regularized) base estimator."""

    def __init__(self, step=0.1, cv=5, ridge_alpha=0.0, lsmr_maxiter=3000,
                 lsmr_atol=1e-8, lsmr_btol=1e-8, n_jobs=-1, min_features=1,
                 verbose=True, random_state=None, patience=5):
        # [FIX P09] lsmr_* used to be dropped here
        super().__init__(step=step, cv=cv, min_features=min_features, n_jobs=n_jobs,
                         verbose=verbose, random_state=random_state,
                         solver="lstsq", ridge_alpha=ridge_alpha,
                         patience=patience, lsmr_maxiter=lsmr_maxiter,
                         lsmr_atol=lsmr_atol, lsmr_btol=lsmr_btol)


class PheasyRFE_OLS_TSQR(_RFECVBase):
    """RFE with a strict OLS base estimator solved by Q-less tall-skinny QR.

    [FIX P09] ``patience``, ``block_rows`` and ``diag_floor`` are now honoured:
    the base solve streams the factorization block by block (see
    ``_tsqr_qless``) instead of running a plain dense QR on the whole matrix.
    The former ``recalibrate`` argument is gone: it never had an effect, and
    the base estimator re-solves exactly every round, so there is nothing to
    recalibrate.
    """

    def __init__(self, step=0.05, patience=5, min_features=100, block_rows=40000,
                 diag_floor=1e-12, cv=5, verbose=True,
                 random_state=None, n_jobs=-1):
        super().__init__(step=step, cv=cv, min_features=min_features, n_jobs=n_jobs,
                         verbose=verbose, random_state=random_state,
                         solver="qr", ridge_alpha=0.0, patience=patience,
                         block_rows=block_rows, diag_floor=diag_floor)'''
    r = sub1(OPT, old, new, dry)
    if r != "OK":
        return r + " (9g)"

    # 9h: Optimizer.fit passes them
    old = '''        elif method == "RFE":
            self._model = PheasyRFECV(
                step=float(os.environ.get("PHEASY_RFE_STEP", "0.1")),
                cv=self._cv,
                ridge_alpha=float(os.environ.get("PHEASY_RFE_RIDGE_ALPHA", "0")),
                n_jobs=int(os.environ.get("PHEASY_N_JOBS", "-1")),
                min_features=int(os.environ.get("PHEASY_RFE_MIN_FEATURES", "1")),
                verbose=True, random_state=self._rand_seed)
            self._model.fit(A, F64, sample_weight=weights)
            coef = self._model.coef_
        elif method == "RFE-OLS-TSQR":
            self._model = PheasyRFE_OLS_TSQR(
                step=float(os.environ.get("PHEASY_TSQR_STEP", "0.05")),
                cv=self._cv,
                min_features=int(os.environ.get("PHEASY_TSQR_MIN_FEATURES", "1")),
                verbose=True, random_state=self._rand_seed)'''
    new = '''        elif method == "RFE":
            self._model = PheasyRFECV(
                step=float(os.environ.get("PHEASY_RFE_STEP", "0.1")),
                cv=self._cv,
                ridge_alpha=float(os.environ.get("PHEASY_RFE_RIDGE_ALPHA", "0")),
                n_jobs=int(os.environ.get("PHEASY_N_JOBS", "-1")),
                min_features=int(os.environ.get("PHEASY_RFE_MIN_FEATURES", "1")),
                patience=int(os.environ.get("PHEASY_RFE_PATIENCE", "5")),
                lsmr_maxiter=int(os.environ.get("PHEASY_LSQR_MAXITER", "5000")),
                lsmr_atol=float(os.environ.get("PHEASY_LSQR_ATOL", "1e-8")),
                lsmr_btol=float(os.environ.get("PHEASY_LSQR_BTOL", "1e-8")),
                verbose=True, random_state=self._rand_seed)
            self._model.fit(A, F64, sample_weight=weights)
            coef = self._model.coef_
        elif method == "RFE-OLS-TSQR":
            # [FIX P13] default matches the class default (100), not 1
            self._model = PheasyRFE_OLS_TSQR(
                step=float(os.environ.get("PHEASY_TSQR_STEP", "0.05")),
                cv=self._cv,
                min_features=int(os.environ.get("PHEASY_TSQR_MIN_FEATURES", "100")),
                block_rows=int(os.environ.get("PHEASY_TSQR_BLOCK_ROWS", "40000")),
                diag_floor=float(os.environ.get("PHEASY_TSQR_DIAG_FLOOR", "1e-12")),
                patience=int(os.environ.get("PHEASY_RFE_PATIENCE", "5")),
                verbose=True, random_state=self._rand_seed)'''
    return sub1(OPT, old, new, dry)


# ---------------------------------------------------------------- P10
@patch("P10", OPT, "LASSO/ALASSO 的 alpha 并列时取最小（不再挑最强正则）+ 可选 1-SE")
def p10(dry):
    old = """        model.fit(A, y, sample_weight=sample_weight)
        self.model_ = model
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_
        self.alpha_ = model.alpha_
        self.alphas_ = np.asarray(model.alphas_)
        self.mse_path_ = np.asarray(model.mse_path_)
        self.n_iter_ = int(model.n_iter_)
        self.n_features_in_ = A.shape[1]
        return self"""
    new = """        model.fit(A, y, sample_weight=sample_weight)
        self.model_ = model
        _reselect_alpha(model, A, y)      # [FIX P10]
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_
        self.alpha_ = model.alpha_
        self.alphas_ = np.asarray(model.alphas_)
        self.mse_path_ = np.asarray(model.mse_path_)
        self.n_iter_ = int(model.n_iter_)
        self.n_features_in_ = A.shape[1]
        return self"""
    r = sub1(OPT, old, new, dry, marker="[FIX P10]")
    if r != "OK":
        return r

    helper = '''

def _reselect_alpha(model, A, y):
    """[FIX P10] Re-pick alpha from the CV path and refit if it changed.

    Two problems with sklearn's plain ``argmin`` here:

    1. When coordinate descent stops early (a loose ``--tol`` is scaled by
       ``||y||^2``, so 1e-3 is very loose), the low-alpha end of the path
       returns literally the same solution and the CV curve goes flat.  Since
       ``alphas_`` is sorted descending, ``argmin`` then picks the *most*
       regularized member of the tie -- which is how LASSO ends up with force
       constants ~10% too small.  Ties are now broken toward the smallest alpha
       and a warning is printed, because a flat tail means "not converged".
    2. ``PHEASY_LASSO_1SE=1`` optionally applies the one-standard-error rule
       (largest alpha within 1 SE of the best), matching what RFE already does.
    """
    alphas = np.asarray(model.alphas_, dtype=np.float64)
    mse = np.asarray(model.mse_path_, dtype=np.float64)
    if alphas.size < 2 or mse.ndim != 2:
        return
    mean = mse.mean(axis=1)
    best = float(mean.min())
    rtol = float(os.environ.get("PHEASY_LASSO_TIE_RTOL", "1e-9"))
    tied = np.flatnonzero(mean <= best * (1.0 + rtol) + 1e-300)
    if os.environ.get("PHEASY_LASSO_1SE", "0").lower() in ("1", "true", "yes"):
        k = int(np.argmin(mean))
        se = float(mse[k].std(ddof=1) / np.sqrt(mse.shape[1])) if mse.shape[1] > 1 else 0.0
        cand = np.flatnonzero(mean <= best + se)
        new_alpha = float(alphas[cand].max())
    else:
        new_alpha = float(alphas[tied].min())
    if tied.size > 1:
        print("[CV] WARNING: %d alphas tie at CV MSE %.6e (%.3e ... %.3e). The "
              "coordinate descent almost certainly stopped on max_iter/tol "
              "rather than converging -- lower --tol (1e-6) and/or raise "
              "--max_iter, otherwise the fit is over-regularized."
              % (tied.size, best, float(alphas[tied].min()),
                 float(alphas[tied].max())), flush=True)
    if new_alpha == float(model.alpha_):
        return
    print("[CV] alpha reselected: %.6e -> %.6e" % (float(model.alpha_), new_alpha),
          flush=True)
    from sklearn.linear_model import Lasso as _Lasso
    est = _Lasso(alpha=new_alpha, fit_intercept=model.fit_intercept,
                 max_iter=model.max_iter, tol=model.tol,
                 selection=getattr(model, "selection", "cyclic"),
                 random_state=getattr(model, "random_state", None))
    est.fit(A, y)
    model.alpha_ = new_alpha
    model.coef_ = est.coef_
    model.intercept_ = est.intercept_
    model.n_iter_ = int(np.max(np.atleast_1d(est.n_iter_)))

'''
    s = read(OPT)
    anchor = "\n\nclass _OLSModel:"
    if anchor not in s:
        return "FAIL (_OLSModel anchor not found)"
    write(OPT, s.replace(anchor, "\n" + helper + "\nclass _OLSModel:", 1), dry)
    return "OK"


# ---------------------------------------------------------------- P11
@patch("P11", OPT, "系数硬阈值可配（默认只对稀疏方法生效，OLS/RFE 不再被截）")
def p11(dry):
    old = """        self._results["coef"] = np.where(np.abs(coef) < 1e-12, 0.0, coef)
        self._model.coef_ = self._results["coef"]"""
    new = """        # [FIX P11] the hard threshold used to hit every method, including OLS
        # and RFE, where zeroing tiny-but-real coefficients is not wanted.
        # Default: only the L1 methods (whose exact zeros are the point).
        _default_tol = "1e-12" if method in ("LASSO", "ALASSO") else "0"
        _zero_tol = float(os.environ.get("PHEASY_COEF_ZERO_TOL", _default_tol))
        if _zero_tol > 0:
            coef = np.where(np.abs(coef) < _zero_tol, 0.0, coef)
        self._results["coef"] = coef
        self._model.coef_ = self._results["coef"]"""
    return sub1(OPT, old, new, dry, marker="[FIX P11]")


# ---------------------------------------------------------------- P12
@patch("P12", OPT, "metrics 键名拼写 n_featrues -> n_features（保留旧别名）")
def p12(dry):
    s = read(OPT)
    if "[FIX P12]" in s:
        return "SKIP (already applied)"
    if s.count('self._metrics["n_featrues"] = self._model.n_features_in_') != 2:
        return "FAIL (expected 2 occurrences of the misspelled key)"
    new = ('self._metrics["n_features"] = self._model.n_features_in_\n'
           '            self._metrics["n_featrues"] = self._metrics["n_features"]  '
           '# [FIX P12] deprecated alias')
    s = s.replace('self._metrics["n_featrues"] = self._model.n_features_in_', new)
    write(OPT, s, dry)
    return "OK"


# ---------------------------------------------------------------- P14
@patch("P14", UTL, "get_sm_dtype 文档与代码一致（默认 float64）")
def p14(dry):
    old = """    float32 内存减半, 但 lsmr 只能收敛到约 1e-7 相对精度 (istop=5);
    float64 精度更高而内存翻倍。同一次求解中所有矩阵必须使用同一 dtype,
    否则 scipy 会静默升型, 内存优势消失且行为难以预期。

    默认值保持 float32, 以确保不改变既有行为。
    \"\"\""""
    new = """    float32 内存减半, 但 lsmr 只能收敛到约 1e-7 相对精度 (istop=5);
    float64 精度更高而内存翻倍。同一次求解中所有矩阵必须使用同一 dtype,
    否则 scipy 会静默升型, 内存优势消失且行为难以预期。

    [FIX P14] 默认值是 **float64** (函数签名的 default 参数), 早先的
    docstring 写成 float32 与代码相反; 且 run_pheasy.py 里若干处曾硬编码
    float32 而绕过本函数, 造成 SM=float32 / FM=float64 的混合精度。
    \"\"\""""
    old2 = """    由环境变量 ``PHEASY_SM_DTYPE`` 控制, 取值 ``float32`` (默认) 或 ``float64``。"""
    new2 = """    由环境变量 ``PHEASY_SM_DTYPE`` 控制, 取值 ``float32`` 或 ``float64`` (默认)。"""
    r = sub1(UTL, old, new, dry, marker="[FIX P14]")
    if r != "OK":
        return r
    return sub1(UTL, old2, new2, dry)


# ---------------------------------------------------------------- P15
@patch("P15", "(multiple)", "消除 SyntaxWarning: invalid escape sequence")
def p15(dry):
    import glob
    import warnings as _warn

    def offending(path, src):
        """Line numbers (1-based) of docstrings that raise a SyntaxWarning."""
        with _warn.catch_warnings(record=True) as caught:
            _warn.simplefilter("always")
            try:
                compile(src, path, "exec")
            except SyntaxError:
                return []
        return sorted({w.lineno for w in caught
                       if issubclass(w.category, SyntaxWarning)})

    targets = sorted(set(glob.glob("*.py") + glob.glob("*/*.py")))
    total = 0
    for path in targets:
        if path.endswith("fix_pheasy_all.py") or not os.path.exists(path):
            continue
        s = read(path)
        bad = offending(path, s)
        if not bad:
            continue
        lines = s.splitlines(keepends=True)
        changed = 0
        for ln in bad:
            # walk backwards to the line that opens this string literal
            for j in range(min(ln, len(lines)) - 1, -1, -1):
                stripped = lines[j].lstrip()
                if stripped.startswith(('"""', "'''")):
                    if stripped[0] in "rR":
                        break
                    indent = len(lines[j]) - len(stripped)
                    lines[j] = lines[j][:indent] + "r" + stripped
                    changed += 1
                    break
        if not changed:
            continue
        new = "".join(lines)
        if offending(path, new):
            continue        # did not actually help -- leave the file alone
        # keep it conservative: only rewrite when the file still compiles
        try:
            compile(new, path, "exec")
        except SyntaxError:
            continue
        if new != s:
            write(path, new, dry)
            total += changed
    return "OK (%d docstrings made raw)" % total if total else "SKIP (nothing to do)"


# ---------------------------------------------------------------- P16
@patch("P16", "(new files)", "补 pyproject.toml / bin/pheasy 入口 / README 骨架")
def p16(dry):
    made = []
    pyproject = '''[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pheasy"
version = "0.0.2"
description = "Force-constant extraction from finite displacements via compressive sensing"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.22",
    "scipy>=1.8",
    "scikit-learn>=1.1",
    "spglib>=2.0",
    "phonopy>=2.17",
    "ase>=3.22",
    "h5py>=3.6",
    "joblib>=1.2",
    "f90nml>=1.4",
]

[project.optional-dependencies]
fast = ["celer>=0.7"]

[project.scripts]
pheasy = "pheasy.run_pheasy:main"

# Flat layout: the repository root IS the ``pheasy`` package (it holds
# __init__.py directly), so packages.find with include=["pheasy*"] matches
# nothing and installs an empty distribution. Map the package names explicitly.
[tool.setuptools]
packages = ["pheasy", "pheasy.core", "pheasy.interface", "pheasy.structure"]

[tool.setuptools.package-dir]
"pheasy" = "."
"pheasy.core" = "core"
"pheasy.interface" = "interface"
"pheasy.structure" = "structure"
'''
    binscript = '''#!/usr/bin/env python3
"""Console entry point (also installed as the ``pheasy`` command)."""
import sys

from pheasy.run_pheasy import main

if __name__ == "__main__":
    sys.exit(main())
'''
    readme = '''# pheasy

Force-constant extraction from finite-displacement / AIMD data.

## Install

```bash
pip install -e .          # exposes the `pheasy` command
pip install -e '.[fast]'  # + celer, a faster LASSO solver
```

## Fitting methods (`-l`)

| flag | method |
|---|---|
| `OLS` | ordinary least squares (LSMR / SVD) |
| `LASSO` | L1 with cross-validated alpha, then debias refit |
| `ALASSO` | adaptive LASSO (Zou 2006) |
| `RFE` | recursive feature elimination, OLS base, grouped CV |
| `RFE-OLS-TSQR` | RFE with a Q-less tall-skinny QR base solver |
| `RIDGE` | L2 with cross-validated alpha |

## Typical workflow

```bash
python3 tools/prepare_dataset.py POSCAR SPOSCAR dataset_disps.npy dataset_forces.npy
pheasy --dim 3 3 3 -w 3 -s --c3 5.2
pheasy --dim 3 3 3 -w 3 -c --c3 5.2
pheasy --dim 3 3 3 -w 3 -d --c3 5.2 --ndata 45 --disp_file
pheasy --dim 3 3 3 -w 3 -f --c3 5.2 --ndata 45 -l OLS --full_ifc --hdf5
```

## Notes

- LASSO / ALASSO need a tight tolerance. `--tol 1e-3` is *not* tight: sklearn
  scales it by `||y||^2`, coordinate descent stops early at small alpha, the CV
  curve goes flat and the fit ends up over-regularized. Use `--tol 1e-6`.
- `PHEASY_SM_DTYPE` (`float64` default) controls the precision of SM / NS / FM.
  All of them must agree, otherwise scipy silently upcasts.
'''
    prepare = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert a displacement/force dataset into the pickles pheasy expects.

pheasy's ``--disp_file`` path reads ``disp_matrix.pkl`` (Cartesian
displacements, shape ``(ndata, natoms, 3)``, Angstrom) and ``force_matrix.pkl``
(forces, same shape, eV/A). This script builds both from a pair of .npy files.

The displacement array may hold either

  * Cartesian displacements already, or
  * **fractional coordinates** of each configuration -- in which case pass
    ``--frac``; the reference frame (``--ref``, default the last one) is
    subtracted, minimum-image wrapped and converted with the supercell lattice.

The corresponding residual forces of the reference frame are subtracted from
every configuration unless ``--no-subtract-residual`` is given.

Usage
-----
    python3 tools/prepare_dataset.py SPOSCAR dataset_disps.npy dataset_forces.npy --frac
"""
import argparse
import pickle

import numpy as np


def read_lattice(sposcar):
    lines = open(sposcar).read().splitlines()
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    return cell * scale


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sposcar")
    ap.add_argument("disps_npy")
    ap.add_argument("forces_npy")
    ap.add_argument("--frac", action="store_true",
                    help="input holds fractional coordinates, not displacements")
    ap.add_argument("--ref", type=int, default=-1,
                    help="index of the undisplaced reference frame (default: last)")
    ap.add_argument("--no-subtract-residual", action="store_true")
    ap.add_argument("--disp-out", default="disp_matrix.pkl")
    ap.add_argument("--force-out", default="force_matrix.pkl")
    args = ap.parse_args()

    d = np.load(args.disps_npy)
    f = np.load(args.forces_npy)
    if d.shape != f.shape or d.ndim != 3 or d.shape[2] != 3:
        raise SystemExit("expected matching (ndata, natoms, 3) arrays, got "
                         "%s and %s" % (d.shape, f.shape))

    if args.frac:
        cell = read_lattice(args.sposcar)
        ref = d[args.ref]
        u = d - ref[None]
        u = u - np.round(u)                      # minimum image
        keep = [i for i in range(d.shape[0]) if i != args.ref % d.shape[0]]
        U = u[keep] @ cell
        F = f[keep]
        if not args.no_subtract_residual:
            F = F - f[args.ref][None]
    else:
        U, F = d, f
        if not args.no_subtract_residual:
            U = np.delete(U, args.ref % d.shape[0], axis=0)
            F = np.delete(F, args.ref % d.shape[0], axis=0) - f[args.ref][None]

    print("displacements %s  rms=%.4f A  max=%.4f A" %
          (U.shape, np.sqrt((U ** 2).mean()), np.abs(U).max()))
    print("forces        %s  max=%.4f eV/A" % (F.shape, np.abs(F).max()))
    if np.abs(U).max() > 0.5:
        print("WARNING: displacements above 0.5 A -- did you mean --frac?")

    with open(args.disp_out, "wb") as fh:
        pickle.dump(np.ascontiguousarray(U, dtype=np.float64), fh)
    with open(args.force_out, "wb") as fh:
        pickle.dump(np.ascontiguousarray(F, dtype=np.float64), fh)
    with open("ndata_total.txt", "w") as fh:
        fh.write(str(U.shape[0]))
    print("wrote %s, %s, ndata_total.txt (--ndata %d)"
          % (args.disp_out, args.force_out, U.shape[0]))


if __name__ == "__main__":
    main()
'''
    new_files = [("pyproject.toml", pyproject, False),
                 (os.path.join("bin", "pheasy"), binscript, True),
                 (os.path.join("tools", "prepare_dataset.py"), prepare, True)]
    if not os.path.exists("README.md"):
        new_files.append(("README.md", readme, False))
    for path, content, executable in new_files:
        if os.path.exists(path):
            continue
        if not dry:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            if executable:
                os.chmod(path, 0o755)
        made.append(path)
    return "OK (%s)" % ", ".join(made) if made else "SKIP (all present)"


# ---------------------------------------------------------------- P18
@patch("P18", RP, "近乎稠密的 sparse SM 自动转 dense（sparse 路径不再比 dense 更慢更费内存）")
def p18(dry):
    old = """                            SM = SM_prime.dot(_ns_sp).tocsr()
                            print(f'[SM-sparse] done in {_ts_sp.time()-_t0_sp:.1f}s, '
                                  f'SM={SM.shape} nnz={SM.nnz} '
                                  f'density={SM.nnz/(SM.shape[0]*SM.shape[1])*100:.2f}% '
                                  f'mem={(SM.data.nbytes+SM.indices.nbytes+SM.indptr.nbytes)/1e9:.1f}GB (sparse)',
                                  flush=True)
                            del SM_prime, _ns_sp"""
    new = """                            SM = SM_prime.dot(_ns_sp).tocsr()
                            _dens = SM.nnz / float(SM.shape[0] * SM.shape[1])
                            print(f'[SM-sparse] done in {_ts_sp.time()-_t0_sp:.1f}s, '
                                  f'SM={SM.shape} nnz={SM.nnz} '
                                  f'density={_dens*100:.2f}% '
                                  f'mem={(SM.data.nbytes+SM.indices.nbytes+SM.indptr.nbytes)/1e9:.1f}GB (sparse)',
                                  flush=True)
                            del SM_prime, _ns_sp
                            # [FIX P18] CSR costs ~2x dense per stored element
                            # (data + int32 index) and column-slicing it in the
                            # RFE loop is much slower. When SM comes out nearly
                            # dense anyway, and dense fits the budget, densify.
                            _dens_thr = float(_os_p.environ.get('PHEASY_SM_DENSIFY_DENSITY', '0.5'))
                            _dens_gb = float(_os_p.environ.get('PHEASY_MAX_DENSE_GB', '16'))
                            _need = SM.shape[0] * SM.shape[1] * _np_p.dtype(_sm_dtype()).itemsize
                            if _dens >= _dens_thr and _need <= _dens_gb * 1e9:
                                SM = _np_p.ascontiguousarray(SM.toarray(), dtype=_sm_dtype())
                                print(f'[SM-sparse] density {_dens*100:.1f}% >= '
                                      f'{_dens_thr*100:.0f}%, densified to '
                                      f'{SM.nbytes/1e9:.2f} GB {SM.dtype} '
                                      f'(PHEASY_SM_DENSIFY_DENSITY / PHEASY_MAX_DENSE_GB)',
                                      flush=True)"""
    return sub1(RP, old, new, dry, marker="[FIX P18]")


# ---------------------------------------------------------------- P19
@patch("P19", OPT, "清理早期 P09 留下的 recalibrate 兼容层（参数彻底移除）")
def p19(dry):
    """Early builds of P09 kept ``recalibrate`` alive behind a
    DeprecationWarning. It never had an effect, so remove it outright. Installs
    patched with the final P09 already lack the shim and just report SKIP.
    """
    s = read(OPT)
    if "recalibrate=None" not in s and "recalibrate=4" not in s:
        return "SKIP (no recalibrate shim present)"
    old = """    ``recalibrate`` is accepted for backward compatibility only -- the current
    algorithm re-solves exactly every round, so there is nothing to
    recalibrate; passing it emits a DeprecationWarning.
    \"\"\"

    def __init__(self, step=0.05, patience=5, min_features=100, block_rows=40000,
                 recalibrate=None, diag_floor=1e-12, cv=5, verbose=True,
                 random_state=None, n_jobs=-1):
        if recalibrate is not None:
            import warnings as _w
            _w.warn("PheasyRFE_OLS_TSQR(recalibrate=...) has no effect and will "
                    "be removed; the base estimator re-solves every round.",
                    DeprecationWarning, stacklevel=2)
        super().__init__("""
    new = """    The former ``recalibrate`` argument is gone: it never had an effect, and
    the base estimator re-solves exactly every round, so there is nothing to
    recalibrate.
    \"\"\"

    def __init__(self, step=0.05, patience=5, min_features=100, block_rows=40000,
                 diag_floor=1e-12, cv=5, verbose=True,
                 random_state=None, n_jobs=-1):
        super().__init__("""
    return sub1(OPT, old, new, dry)


# ---------------------------------------------------------------- P20
@patch("P20", "pyproject.toml", "修复扁平布局下 packages.find 找不到包、pip install -e 装出空包")
def p20(dry):
    """An early P16 shipped ``[tool.setuptools.packages.find] include=["pheasy*"]``.

    This repository uses a flat layout -- the root directory itself is the
    ``pheasy`` package -- so ``find`` scans the root's subdirectories (core,
    interface, structure, ...) and matches none of them. pip then installs an
    empty distribution and ``import pheasy`` fails with ModuleNotFoundError.
    Replace it with an explicit package-dir mapping.
    """
    path = "pyproject.toml"
    if not os.path.exists(path):
        return "SKIP (no pyproject.toml -- run P16 first)"
    s = read(path)
    if "[tool.setuptools.package-dir]" in s:
        return "SKIP (already applied)"
    old = """[tool.setuptools.packages.find]
where = ["."]
include = ["pheasy*"]
"""
    new = """# Flat layout: the repository root IS the ``pheasy`` package (it holds
# __init__.py directly), so packages.find with include=["pheasy*"] matches
# nothing and installs an empty distribution. Map the package names explicitly.
[tool.setuptools]
packages = ["pheasy", "pheasy.core", "pheasy.interface", "pheasy.structure"]

[tool.setuptools.package-dir]
"pheasy" = "."
"pheasy.core" = "core"
"pheasy.interface" = "interface"
"pheasy.structure" = "structure"
"""
    r = sub1(path, old, new, dry)
    if r == "OK":
        print("         -> reinstall afterwards:  pip install -e .")
    return r


# ---------------------------------------------------------------- P21
@patch("P21", OPT, "统一 RFE / TSQR 的默认 step=0.05（两者默认不同会让横向比较失真）")
def p21(dry):
    """RFE defaulted to step=0.1 while RFE-OLS-TSQR defaulted to 0.05.

    That single difference makes the two look like different *methods*: with
    step=0.1 the elimination path jumps 1303 -> 1173 and skips the CV optimum
    at 1238 that step=0.05 lands on. Run both at step=0.05 and their CV paths
    agree to seven digits -- they are one algorithm with two back-end solvers.
    Unify the default and let PHEASY_RFE_STEP drive both.
    """
    if "[FIX P21]" in read(OPT):
        return "SKIP (already applied)"
    reps = [
        ("    def __init__(self, step=0.1, cv=5, ridge_alpha=0.0, lsmr_maxiter=3000,",
         "    def __init__(self, step=0.05, cv=5, ridge_alpha=0.0, lsmr_maxiter=3000,   # [FIX P21]"),
        ('                step=float(os.environ.get("PHEASY_RFE_STEP", "0.1")),',
         '                step=float(os.environ.get("PHEASY_RFE_STEP", "0.05")),  # [FIX P21]'),
        ('                step=float(os.environ.get("PHEASY_TSQR_STEP", "0.05")),',
         '                # [FIX P21] PHEASY_TSQR_STEP overrides, else PHEASY_RFE_STEP\n'
         '                step=float(os.environ.get(\n'
         '                    "PHEASY_TSQR_STEP",\n'
         '                    os.environ.get("PHEASY_RFE_STEP", "0.05"))),'),
    ]
    for old, new in reps:
        r = sub1(OPT, old, new, dry)
        if r != "OK":
            return r + "  <- %r" % old.strip()[:60]
    return "OK"


# ---------------------------------------------------------------- P22
@patch("P22", RP, "新增 PHEASY_SKIP_FC_WRITE：跳过 fc2/fc3/fc4 展开写出（扫描时省内存和时间）")
def p22(dry):
    """The order-3 expansion is (natoms^3, 3, 3, 3) -- 1.8e8 doubles (1.5 GB)
    for a 189-atom supercell -- and it is rebuilt and written on every single
    run. For a method/ndata sweep that is pure waste: phi.npz already holds the
    complete IFC vector (NS_full @ coef) that fc*.hdf5 is merely an expansion
    of. Worse, running several fits in parallel makes this the OOM trigger.

    PHEASY_SKIP_FC_WRITE=1 keeps phi.npz and skips the expansion entirely.
    """
    old = """                FC_model.set_force_constants(Phi)
                FC_model.write_force_constants(settings, self.CS_full, order=2)
                logger.info("Writing second-order force constants into file.")

            if settings.MAX_ORDER > 2:
                FC_model.write_force_constants(settings, self.CS_full, order=3)
                logger.info("Writing third-order force constants into file.")

            if settings.MAX_ORDER > 3:
                FC_model.write_force_constants(settings, self.CS_full, order=4)
                logger.info("Writing fourth-order force constants into file.")"""
    new = """                FC_model.set_force_constants(Phi)

            # [FIX P22] phi.npz (written just above) already carries the full
            # IFC vector; fc*.hdf5 is only its expansion over atom triplets and
            # costs O(natoms^order) memory. Skip it for parameter sweeps.
            _skip_fc = os.environ.get(
                "PHEASY_SKIP_FC_WRITE", "").lower() in ("1", "true", "yes")
            if _skip_fc:
                logger.info(
                    "PHEASY_SKIP_FC_WRITE=1: skipping fc2/fc3/fc4 expansion; "
                    "phi.npz holds the complete force constants.")
            elif not settings.FIX_FC2:
                FC_model.write_force_constants(settings, self.CS_full, order=2)
                logger.info("Writing second-order force constants into file.")

            if not _skip_fc and settings.MAX_ORDER > 2:
                FC_model.write_force_constants(settings, self.CS_full, order=3)
                logger.info("Writing third-order force constants into file.")

            if not _skip_fc and settings.MAX_ORDER > 3:
                FC_model.write_force_constants(settings, self.CS_full, order=4)
                logger.info("Writing fourth-order force constants into file.")"""
    return sub1(RP, old, new, dry, marker="[FIX P22]")


# ---------------------------------------------------------------- P17
@patch("P17", "MnIn2Se4_lasso/pheasy_fit.sh", "拟合脚本：--tol 1e-6、方法白名单、去掉未生效项")
def p17(dry):
    path = os.path.join("MnIn2Se4_lasso", "pheasy_fit.sh")
    if not os.path.exists(path):
        return "SKIP (script not in this checkout)"
    s = read(path)
    # Recognize an already-modernized script, not just this patch's own marker:
    # the file may have been replaced wholesale by a newer hand-written version.
    if "[FIX P17]" in s or "LASSO_TOL" in s or "_ALLOWED" in s:
        return "SKIP (already applied)"
    if "--tol 0.001" not in s:
        return ("SKIP (script does not contain the loose '--tol 0.001'; "
                "nothing to fix -- check LASSO_TOL / --std for RIDGE by hand)")
    old = '''  FIT_FLAGS="${FIT_FLAGS} --alpha_auto --max_iter 3000 --cv 5 --nmu 20 --tol 0.001"'''
    new = '''  # [FIX P17] --tol 0.001 is NOT tight: sklearn scales tol by ||y||^2, so
  # coordinate descent stops long before convergence at small alpha, the CV
  # curve goes flat and the tie-break picks the most-regularized alpha ->
  # force constants come out ~10% too small. 1e-6 costs seconds, not hours.
  FIT_FLAGS="${FIT_FLAGS} --alpha_auto --max_iter 20000 --cv 5 --nmu 20 --tol ${LASSO_TOL}"'''
    r = sub1(path, old, new, dry)
    if r != "OK":
        return r
    old2 = '''NULL_SPACE_EPS=0.001
STANDARDIZE=true'''
    new2 = '''NULL_SPACE_EPS=0.001
STANDARDIZE=true
LASSO_TOL=1e-6          # [FIX P17] see the note next to FIT_FLAGS below'''
    r = sub1(path, old2, new2, dry)
    if r != "OK":
        return r
    old3 = '''echo "拟合: ${FIT_METHOD} | 阶次 ${FIT_ORDER}'''
    new3 = '''case "${FIT_METHOD}" in                      # [FIX P17] fail fast on typos
  OLS|LASSO|ALASSO|RFE|RFE-OLS-TSQR|RIDGE) ;;
  *) echo "FIT_METHOD=${FIT_METHOD} 不是合法方法; 可选: OLS LASSO ALASSO RFE RFE-OLS-TSQR RIDGE" >&2; exit 2 ;;
esac

echo "拟合: ${FIT_METHOD} | 阶次 ${FIT_ORDER}'''
    return sub1(path, old3, new3, dry)


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="+", metavar="Pxx")
    ap.add_argument("--revert", metavar="STAMP", nargs="?", const="latest")
    args = ap.parse_args()

    if args.list:
        for pid, path, title, _ in PATCHES:
            print("%-5s %-24s %s" % (pid, path, title))
        return 0

    if args.revert:
        import glob
        pats = sorted(glob.glob("**/*.bak_fixall_*", recursive=True))
        if not pats:
            print("no fixall backups found")
            return 1
        stamps = sorted({p.split(".bak_fixall_")[-1] for p in pats})
        stamp = stamps[-1] if args.revert == "latest" else args.revert
        n = 0
        for b in pats:
            if not b.endswith(stamp):
                continue
            orig = b.split(".bak_fixall_")[0]
            shutil.copy2(b, orig)
            print("restored", orig)
            n += 1
        print("reverted %d file(s) from %s" % (n, stamp))
        return 0

    if not (os.path.exists("run_pheasy.py") and
            os.path.exists(os.path.join("core", "optimizer.py"))):
        print("ERROR: run this from the pheasy package directory "
              "(the one containing run_pheasy.py and core/optimizer.py).",
              file=sys.stderr)
        return 1

    selected = PATCHES
    if args.only:
        want = {p.upper() for p in args.only}
        selected = [p for p in PATCHES if p[0] in want]
        if not selected:
            print("no patch matches", args.only, file=sys.stderr)
            return 1

    print("pheasy fix-all  (%s)%s" % (STAMP, "   [DRY RUN]" if args.dry_run else ""))
    print("=" * 78)
    ok = fail = skip = 0
    for pid, path, title, fn in selected:
        try:
            res = fn(args.dry_run)
        except Exception as e:
            res = "FAIL (%s: %s)" % (type(e).__name__, e)
        tag = "OK" if res.startswith("OK") else ("SKIP" if res.startswith("SKIP") else "FAIL")
        ok += tag == "OK"
        skip += tag == "SKIP"
        fail += tag == "FAIL"
        print("[%-4s] %-5s %-22s %s" % (tag, pid, path, title))
        if res not in ("OK",):
            print("         -> %s" % res)
    print("=" * 78)
    print("applied %d, skipped %d, failed %d" % (ok, skip, fail))

    if not args.dry_run and _backed_up:
        print("backups: " + ", ".join(sorted(p + BAK_SUFFIX for p in _backed_up)))
        print("rollback: python3 %s --revert" % os.path.basename(__file__))

    if not args.dry_run and fail == 0:
        print("\nsyntax check:")
        import glob
        bad = 0
        for f in sorted(set(glob.glob("*.py") + glob.glob("*/*.py"))):
            if f.endswith("fix_pheasy_all.py"):
                continue
            try:
                with open(f, encoding="utf-8") as fh:
                    compile(fh.read(), f, "exec")
                print("  ok    %s" % f)
            except SyntaxError as e:
                print("  BAD   %s: %s" % (f, e))
                bad += 1
        if bad:
            return 1
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())