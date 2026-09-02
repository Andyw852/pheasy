"""Classes and functions for force constant regression.

Implements the five force-constant fitting methods exposed by pheasy:
OLS, RFE, RFE-OLS-TSQR (RFE_TSQR), LASSO and ALASSO (adaptive LASSO), plus
the legacy RIDGE method. The public entry point is the Optimizer class.

References:
  H. Zou, "The Adaptive Lasso and Its Oracle Properties", JASA 101 (2006).
  F. Eriksson et al., Adv. Theory Simul. 2 (2019) (hiphive).
  J. Demmel et al., SIAM J. Sci. Comput. 34 (2012) A206 (TSQR).
"""
import contextlib
import os
import threading

import numpy as np
import scipy.sparse as sp
from scipy import linalg as spla
from scipy.sparse.linalg import LinearOperator, lsmr as _lsmr

from sklearn.linear_model import LassoCV, Ridge, RidgeCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, KFold

__all__ = ["Optimizer", "TwoLevelSM"]


def _avail_cores():
    """Usable CPU count for THIS job (affinity/cgroup aware), not the node's.

    os.cpu_count() reports the node's physical core count, which on a shared
    Slurm box overstates what --cpus-per-task actually granted. Prefer the
    process's CPU affinity; PHEASY_MAX_CORES overrides both (the shell sets it
    to $NCPU so the code never guesses the node width).
    """
    n = os.environ.get("PHEASY_MAX_CORES")
    if n:
        return max(1, int(n))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _resolve_n_jobs(method=None, default=-1):
    """Unified outer-parallelism resolver.

    Precedence: PHEASY_<METHOD>_N_JOBS > PHEASY_N_JOBS > ``default``.
    0 means serial, a value < 0 means "all available cores", and a positive
    value is clamped to [1, _avail_cores()]. Method names use '-' -> '_'
    (e.g. "RFE-OLS-TSQR" -> "RFE_OLS_TSQR").
    """
    raw = None
    if method:
        raw = os.environ.get("PHEASY_%s_N_JOBS" % method.upper().replace("-", "_"))
    if raw in (None, ""):
        raw = os.environ.get("PHEASY_N_JOBS")
    n = int(raw) if raw not in (None, "") else int(default)
    if n == 0:
        return 1          # 0 = serial (old behaviour + joblib convention)
    n_cpu = _avail_cores()
    return max(1, min(n if n > 0 else n_cpu, n_cpu))


_BLAS_WARNED = False  # one-time no-threadpoolctl warning


@contextlib.contextmanager
def _blas_limit(n_outer):
    """Cap each worker's BLAS threads when the OUTER loop already runs n_outer.

    Without this, K outer folds x many inner BLAS threads oversubscribe the box
    and can be slower than serial. threadpool_limits sets the limit for the
    current process (all joblib threads see it); a missing threadpoolctl
    degrades to a no-op instead of raising.
    """
    per = max(1, _avail_cores() // max(1, n_outer))
    # Import OUTSIDE the yield scope: a `yield` inside `try` means any
    # ImportError raised by the *body* (inside `with _blas_limit`) would be
    # caught here and trigger a second yield -> "RuntimeError: generator
    # didn't stop after throw()", destroying the original error.
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits = None
    if threadpool_limits is None:
        global _BLAS_WARNED
        if not _BLAS_WARNED:
            print("[BLAS] threadpoolctl 缺失: 内外层线程不再协调, "
                  "外层 %d 折 x 内层满核可能超订" % n_outer, flush=True)
            _BLAS_WARNED = True
        yield
    else:
        with threadpool_limits(limits=per):
            yield


def _to_dense_f64(A):
    """Return A as a dense float64 ndarray (C-contiguous)."""
    if sp.issparse(A):
        return np.ascontiguousarray(A.toarray(), dtype=np.float64)
    if isinstance(A, np.ndarray):
        return np.ascontiguousarray(A, dtype=np.float64)
    if hasattr(A, "matvec"):  # LinearOperator (e.g. TwoLevelSM)
        n = A.shape[1]
        return np.ascontiguousarray(A @ np.eye(n, dtype=np.float64), dtype=np.float64)
    return np.ascontiguousarray(A, dtype=np.float64)


def _is_linear_operator(A):
    return (not isinstance(A, np.ndarray)) and (not sp.issparse(A)) and hasattr(A, "matvec")


def _col_norms(A):
    """Exact ||A[:, j]|| in float64."""
    if sp.issparse(A):
        sq = np.asarray(A.multiply(A).sum(axis=0)).ravel()
        return np.sqrt(sq.astype(np.float64))
    if isinstance(A, np.ndarray):
        A64 = A.astype(np.float64, copy=False)
        return np.sqrt(np.einsum("ij,ij->j", A64, A64))
    n = A.shape[1]
    norms = np.zeros(n, dtype=np.float64)
    block = 64
    for j0 in range(0, n, block):
        j1 = min(j0 + block, n)
        I = np.zeros((n, j1 - j0), dtype=np.float64)
        I[j0:j1, :] = np.eye(j1 - j0, dtype=np.float64)
        col = A @ I
        norms[j0:j1] = np.sqrt(np.einsum("ij,ij->j", col, col))
    return norms


def derive_alpha_grid(A, y, nalpha=100, decades=4.0, standardize=False,
                     mu_shift=0.0):
    """Derive a LASSO/ALASSO alpha grid from the data.

    alpha_max = max_j |X_j^T y| / n  is the smallest alpha for which the LASSO
    solution is all zeros (the KKT threshold, matching sklearn's convention).
    The grid spans ``[alpha_max * 10**-decades, alpha_max]``.  When
    ``standardize`` is True the columns are first scaled to unit L2 norm (the
    same scaling the Optimizer applies), so the returned grid lives in the
    standardized space.

    Memory efficient: chunked accumulation for dense (mmap-friendly) input,
    sparse matvec for sparse / LinearOperator input.

    Returns a float64 array of alpha VALUES.
    """
    n = A.shape[0]
    p = A.shape[1]
    y64 = np.asarray(y, dtype=np.float64).ravel()
    g = np.zeros(p, dtype=np.float64)

    if sp.issparse(A) or _is_linear_operator(A):
        g = np.asarray(A.T @ y64).ravel().astype(np.float64)
        if standardize:
            cn = _col_norms(A)
            cn = np.where(cn < 1e-30, 1.0, cn)
            g = g / cn
    else:
        s2 = np.zeros(p, dtype=np.float64)
        blk = max(1, int(2e8 // max(p, 1)))
        for i0 in range(0, n, blk):
            B = np.asarray(A[i0:i0 + blk], dtype=np.float64)
            g += B.T @ y64[i0:i0 + B.shape[0]]
            if standardize:
                s2 += (B * B).sum(axis=0)
        if standardize:
            cn = np.sqrt(s2)
            cn = np.where(cn < 1e-30, 1.0, cn)
            g = g / cn

    a_max = float(np.abs(g).max()) / n
    a_max *= 10.0 ** float(mu_shift)
    if not np.isfinite(a_max) or a_max <= 0:
        raise ValueError("alpha_max = %r, invalid" % a_max)
    a_min = a_max * 10.0 ** (-float(decades))
    return np.logspace(np.log10(a_min), np.log10(a_max), nalpha)


def _make_cv_splits(n_samples, cv, random_state=None, group_size=None):
    """Return a list of (train_idx, val_idx) index arrays."""
    if cv is None or cv <= 1:
        cv = min(3, n_samples)
    cv = int(cv)
    if group_size and group_size > 1 and n_samples % group_size == 0:
        groups = np.arange(n_samples) // group_size
        n_groups = int(groups[-1]) + 1
        if n_groups >= 2:
            # Clamp cv to the number of configurations. A row-based KFold here
            # would leak rows of the same configuration across train/val folds,
            # silently biasing the CV estimate (FIX P23).
            eff_cv = int(min(cv, n_groups))
            gkf = GroupKFold(n_splits=eff_cv)
            return list(gkf.split(np.zeros(n_samples, dtype=np.int8),
                                  np.zeros(n_samples, dtype=np.int8), groups))
    # no group info (or a single configuration): fall back to row-based KFold
    cv = max(2, min(cv, n_samples))
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    return list(kf.split(np.arange(n_samples)))


def _solve_sparse_lsqr(A, y):
    """Iterative least squares (LSQR) for sparse / LinearOperator input.

    Only needs matvec / rmatvec, so peak memory is ~O(n_features) instead of
    densifying the sensing matrix (which for e.g. 685968 x 51590 would be
    ~280 GB). This is the same Krylov approach used by symfc / phonopy.
    """
    from scipy.sparse.linalg import lsqr as _sp_lsqr
    # [FIX] tolerance must follow operator dtype: float32 (eps ~1.2e-7)
    # cannot converge to atol=1e-8 (unreachable -> LSQR always hits iter_lim).
    _eps = np.finfo(np.dtype(getattr(A, "dtype", np.float64))).eps
    _dflt = "1e-8" if _eps < 1e-10 else "1e-6"
    atol = float(os.environ.get("PHEASY_LSQR_ATOL", _dflt))
    btol = float(os.environ.get("PHEASY_LSQR_BTOL", _dflt))
    iter_lim = int(os.environ.get("PHEASY_LSQR_MAXITER", "5000"))
    y64 = np.asarray(y, dtype=np.float64).ravel()
    res = _sp_lsqr(A, y64, atol=atol, btol=btol, iter_lim=iter_lim,
                   show=(os.environ.get("PHEASY_LSQR_SHOW", "0") == "1"))
    print("[LSQR] istop={} itn={} r1norm={:.3e} (atol={:.0e}, lim={})".format(
        res[1], res[2], res[3], atol, iter_lim), flush=True)
    return np.asarray(res[0], dtype=np.float64)


def _available_memory_bytes():
    """Available physical RAM in bytes (None if undetectable).

    [FIX P26] Reads the OS's available-memory estimate (not total) so the
    dense/iterative dispatch reflects what the node can actually hand out right
    now, not just a fixed element budget.
    """
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return None


def _should_densify_sparse(A):
    """True when densifying a sparse matrix is cheap enough (memory budget).

    A sparse container that is actually 100% dense (e.g. SM = SM_prime @ NS for
    small systems) is densified so the faster SVD/QR solvers run; genuinely
    sparse / huge matrices stay sparse and use the iterative LSQR solver.

    [FIX P26] now memory-aware: besides the PHEASY_MAX_DENSE element cap, the
    float64 footprint must fit in PHEASY_SOLVER_MEM_FRACTION of the available
    RAM (default 0.25). This keeps a large system on the iterative path even
    when its element count is small on paper but the node is already loaded.
    """
    n, m = A.shape
    max_dense = int(os.environ.get("PHEASY_MAX_DENSE", "200000000"))
    if (n * m) > max_dense:
        return False
    avail = _available_memory_bytes()
    if avail is not None:
        frac = float(os.environ.get("PHEASY_SOLVER_MEM_FRACTION", "0.25"))
        if (n * m * 8) > avail * frac:
            return False
    return True


def _lasso_backend(A):
    """Choose the LASSO/ALASSO backend: "dense" (sklearn) or "iterative" (FISTA).

    sklearn's LassoCV / RidgeCV only accept a materialized array, so any
    LinearOperator -- and any sparse matrix too big to densify -- must go
    through the matvec-only FISTA solver instead. This is the same dispatch
    policy _solve_lstsq already uses for OLS.
    """
    if _is_linear_operator(A):
        return "iterative"
    if sp.issparse(A) and not _should_densify_sparse(A):
        return "iterative"
    return "dense"


def _solve_lstsq(A, y, driver="gelsd"):
    """Ordinary least squares: SVD for dense/small, LSQR for sparse/huge."""
    if _is_linear_operator(A):
        return _solve_sparse_lsqr(A, y)
    if sp.issparse(A) and not _should_densify_sparse(A):
        return _solve_sparse_lsqr(A, y)
    A64 = _to_dense_f64(A)
    y64 = np.asarray(y, dtype=np.float64).ravel()
    try:
        coef, *_ = spla.lstsq(A64, y64, cond=None, lapack_driver=driver)
    except Exception:
        coef, *_ = spla.lstsq(A64, y64, cond=None, lapack_driver="gelsd")
    return np.asarray(coef, dtype=np.float64)


def _tsqr_qless(A, y, block_rows=40000, diag_floor=1e-12):
    """[FIX P09/P24] Q-less tall-skinny QR least squares via a binary TREE.

    Level 0 QR-factors each row block independently, then the R factors are
    paired and re-QR'd in a binary tree (log-depth).  This is the classic TSQR
    of Demmel et al. and is numerically more stable than sequentially re-QRing
    one growing [R; A_i] stack (the previous implementation), which lets
    rounding error accumulate along the chain for ill-conditioned matrices.

    Peak memory is O(ceil(log2(nblocks)) * n^2 + block_rows * n): same-level
    R factors are merged on the fly, so at most one R per tree level is alive.
    Never a full Q or a full copy of A.

    Returns (coef, rank_ok, cond_estimate).
    """
    m, n = A.shape
    y64 = np.asarray(y, dtype=np.float64).ravel()
    # [FIX P36] no longer force block_rows >= n+1: the wide-matrix branch in
    # the final solve handles level-0 R factors that are wider than tall (which
    # the tree reduction accumulates to full rank). Letting the user pick a
    # small block_rows shrinks the block densification buffer from
    # block_rows*n*8 to blk*n*8 (e.g. cutoff 5.5: 21 GB -> 1.7 GB at blk=4000).
    # [FIX P36c] block-height guardrail: below ~n/4 the QR-call overhead grows
    # without any memory savings (the final n^2 R dominates the peak), so clamp
    # the effective height to at least min(n, 2048) and say so.  Correctness is
    # unaffected -- this only keeps users out of the counterproductive range.
    _blk = int(block_rows) if block_rows else 0
    _floor = min(n, 2048)
    if 0 < _blk < _floor and _blk < n // 4:
        print("[optimizer] WARNING: TSQR block_rows=%d << n=%d; effective height "
              "raised to %d (tiny blocks add QR-call overhead, not memory "
              "savings)." % (_blk, n, _floor), flush=True)
    block_rows = max(_blk, _floor)

    # ---- level 0: independent QR of each block --------------------------------
    # [FIX P36] streaming binary-tree TSQR: merge same-level R factors on the
    # fly, so at most ceil(log2(nblocks))+1 R matrices are alive at any time
    # (peak O(ceil(log2(nblocks))*n^2 + block_rows*n)) instead of materialising
    # every level-0 R before reducing (O(nblocks*n^2), which for wide/small-
    # block problems exceeds even O(m*n)). Same binary tree as the batch
    # version, just a different traversal order -- numerically equivalent.
    stack = []  # list of (level, R, z)
    for i0 in range(0, m, block_rows):
        i1 = min(i0 + block_rows, m)
        blk = A[i0:i1]
        blk = np.asarray(blk.toarray() if sp.issparse(blk) else blk, dtype=np.float64)
        Q, R = spla.qr(blk, mode="economic", check_finite=False)
        z = np.asarray(Q.T @ y64[i0:i1], dtype=np.float64)
        del Q, blk
        lvl = 0
        while stack and stack[-1][0] == lvl:   # merge same-level on the fly
            _l0, R0, z0 = stack.pop()
            M = np.vstack([R0, R])
            b = np.concatenate([z0, z])
            Q, R = spla.qr(M, mode="economic", check_finite=False)
            z = np.asarray(Q.T @ b, dtype=np.float64)
            del Q, M, b, R0, z0
            lvl += 1
        stack.append((lvl, R, z))
    while len(stack) > 1:                       # finish the residual tree
        l1, R1, z1 = stack.pop()
        l0, R0, z0 = stack.pop()
        M = np.vstack([R0, R1])
        b = np.concatenate([z0, z1])
        Q, R = spla.qr(M, mode="economic", check_finite=False)
        z = np.asarray(Q.T @ b, dtype=np.float64)
        stack.append((max(l0, l1) + 1, R, z))
    R = stack[0][1]
    z = stack[0][2]
    wide = R.shape[0] < R.shape[1]
    diag = np.abs(np.diag(R))
    dmax = float(diag.max()) if diag.size else 0.0
    dmin = float(diag.min()) if diag.size else 0.0
    if wide:
        # [FIX P36] diag of a wide R does not reflect conditioning.
        cond = np.nan
    else:
        cond = (dmax / dmin) if dmin > 0 else np.inf
    if dmax == 0.0 or (not wide and dmin <= diag_floor * max(dmax, 1.0)):
        return None, False, cond
    if wide:
        # [FIX P35] underdetermined (m < n): the reduced R is (m, n) and not
        # square, so the diag-based rank criterion is meaningless and
        # solve_triangular would raise. min-norm lstsq == gelsd on the
        # original A (e.g. c3=7.0 with a small NDATA, or small block_rows).
        coef = spla.lstsq(R, z, check_finite=False)[0]
    else:
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
        # [FIX P35] _tsqr_qless can raise (e.g. wide matrices); catch so the
        # SVD fallback actually runs instead of propagating the exception.
        try:
            coef, ok, _cond = _tsqr_qless(A, y, block_rows, diag_floor)
        except Exception:
            ok = False
        if ok:
            return coef
        return _solve_lstsq(A, y)     # rank deficient / wide -> SVD
    A64 = _to_dense_f64(A)
    y64 = np.asarray(y, dtype=np.float64).ravel()
    Q, R = spla.qr(A64, mode="economic", check_finite=False)
    diag = np.abs(np.diag(R))
    if diag.size == 0:
        return _solve_lstsq(A, y)
    dmax = float(diag.max())
    if dmax == 0.0:
        return _solve_lstsq(A, y)
    # rank threshold relative to the largest R diagonal (proxy for largest
    # singular value); fall back to SVD when (nearly) rank deficient.
    tol = np.finfo(float).eps * max(A64.shape) * dmax
    if float(diag.min()) <= tol:
        return _solve_lstsq(A, y)
    if R.shape[0] == R.shape[1]:
        coef = spla.solve_triangular(R, Q.T @ y64, lower=False, check_finite=False)
    else:
        # [FIX P35] wide / underdetermined (n_rows < n_features): the economic
        # QR gives a non-square R, so solve_triangular fails.  The min-norm
        # least-squares solution of R x = Q^T y equals gelsd on the original
        # A, which is what the other solvers return for the underdetermined
        # case (e.g. c3=7.0 with a small NDATA).
        coef = spla.lstsq(R, Q.T @ y64, check_finite=False)[0]
    return np.asarray(coef, dtype=np.float64)


def _make_masked_op(A, row_idx, col_idx):
    """LinearOperator for A[row_idx][:, col_idx] without materializing A."""
    n_rows_full, n_cols_full = A.shape
    n_rows = n_rows_full if row_idx is None else len(row_idx)
    n_cols = len(col_idx)
    dt = np.dtype(A.dtype) if hasattr(A, "dtype") else np.dtype(np.float64)

    def mv(v):
        v = np.asarray(v, dtype=dt).ravel()
        v_full = np.zeros(n_cols_full, dtype=dt)
        v_full[col_idx] = v
        out = np.asarray(A @ v_full).ravel()
        return out if row_idx is None else out[row_idx]

    def rmv(u):
        u = np.asarray(u, dtype=dt).ravel()
        if row_idx is not None:
            u_full = np.zeros(n_rows_full, dtype=dt)
            u_full[row_idx] = u
        else:
            u_full = u
        return np.asarray(A.T @ u_full).ravel()[col_idx]

    return LinearOperator((n_rows, n_cols), matvec=mv, rmatvec=rmv, dtype=dt)


def _soft_threshold(x, thr):
    """Elementwise soft-thresholding: sign(x) * max(|x| - thr, 0)."""
    x = np.asarray(x)
    return np.sign(x) * np.maximum(np.abs(x) - thr, 0.0)


def _estimate_lipschitz(A, power_iters=15):
    """Estimate L = ||A||_2^2 (top eigenvalue of A^T A) by power iteration.

    Only needs matvec/rmatvec, so it works for dense, sparse and
    LinearOperator (TwoLevelSM) alike. L is the Lipschitz constant of the
    LASSO smooth part.
    """
    n = A.shape[1]
    v = np.random.RandomState(0).randn(n)
    v = v / (np.linalg.norm(v) + 1e-300)
    for _ in range(power_iters):
        u = np.asarray(A @ v, dtype=np.float64).ravel()
        v = np.asarray(A.T @ u, dtype=np.float64).ravel()
        vn = np.linalg.norm(v)
        if vn < 1e-30:
            break
        v = v / vn
    u = np.asarray(A @ v, dtype=np.float64).ravel()
    L = max(float(np.dot(u, u)), 1e-12)
    # [FIX P29] power iteration approaches lambda_max from BELOW, so the bare
    # estimate under-shoots L and makes step = 1/L too large: FISTA can become
    # non-monotonic or even diverge when the spectral gap is small.  A small
    # safety margin keeps the step conservative (override via env if needed).
    safety = float(os.environ.get("PHEASY_FISTA_LIPSCHITZ_SAFETY", "1.02"))
    return L * safety


def _top_eigval(G):
    """Largest eigenvalue of a symmetric PSD matrix (exact LAPACK).

    [FIX P34] ||A||^2 = lambda_max(A^T A) exactly for the Gram path, replacing
    the power-iteration estimate (and its safety factor) with a tighter step.
    """
    G = np.asarray(G, dtype=np.float64)
    n = G.shape[0]
    if n == 0:
        return 0.0
    try:
        return float(spla.eigvalsh(G, subset_by_index=(n - 1, n - 1))[0])
    except TypeError:
        # older scipy without subset_by_index: full eigendecomposition fallback
        return float(spla.eigvalsh(G)[-1])


def _gram_smprime(SM_prime, block_rows=2000):
    """P = SM_prime^T SM_prime via blocked densification + BLAS gemm.

    [FIX P34] SM_prime (n x mid) may be huge and sparse; densifying it all at
    once costs n*mid*8 bytes. Processing row blocks keeps peak memory at
    block_rows*mid*8 and accumulates the mid x mid Gram with BLAS.
    """
    n, mid = SM_prime.shape
    P = np.zeros((mid, mid), dtype=np.float64)
    for i0 in range(0, n, block_rows):
        B = np.asarray(SM_prime[i0:i0 + block_rows].toarray(), dtype=np.float64)
        P += B.T @ B
    return P


def _gram_budget_ok(A, max_gb=None):
    """[FIX P34] True if the Gram matrices fit the memory budget.

    For TwoLevelSM the dominant intermediate is P = SM_prime^T SM_prime
    (mid x mid), which can exceed G (n_features x n_features) when the null
    space projects a lot of columns away. Default 4 GB ~ 23000 dof.
    """
    max_gb = float(os.environ.get("PHEASY_GRAM_MAX_GB", "4")) if max_gb is None else float(max_gb)
    mid = getattr(A, "SM_prime", None)
    peak_dim = mid.shape[1] if mid is not None else A.shape[1]
    return peak_dim * peak_dim * 8.0 / 1e9 <= max_gb


def _compute_gram(A, y):
    """Precompute G = A^T A (n_features x n_features) and b = A^T y.

    [FIX P34] For TwoLevelSM the Gram is built factorized (P = SM_prime^T
    SM_prime, then G = NS^T P NS) without ever materializing the full SM.
    Returns (G, b, P): P is the SM_prime Gram kept so folds can use the cheap
    P_full - P_va identity instead of recomputing per fold.
    """
    n, m = A.shape
    y64 = np.asarray(y, dtype=np.float64).ravel()
    b = np.asarray(A.T @ y64, dtype=np.float64).ravel()
    P = None
    if hasattr(A, "SM_prime"):  # TwoLevelSM
        P = _gram_smprime(A.SM_prime)
        G = np.asarray(A.NS.T @ (P @ A.NS), dtype=np.float64)
    elif sp.issparse(A):
        G = np.asarray((A.T @ A).toarray(), dtype=np.float64)
    elif _is_linear_operator(A):
        # [FIX P34] build G column-block-wise so a bare LinearOperator does NOT
        # materialize the full dense SM (n_rows x n_features).  Peak memory is
        # n_rows x blk instead.
        G = np.zeros((m, m), dtype=np.float64)
        blk = int(os.environ.get("PHEASY_GRAM_BLOCK", "64"))
        for j0 in range(0, m, blk):
            j1 = min(j0 + blk, m)
            I_blk = np.zeros((m, j1 - j0), dtype=np.float64)
            I_blk[j0:j1, :] = np.eye(j1 - j0, dtype=np.float64)
            A_blk = np.asarray(A @ I_blk, dtype=np.float64)   # n_rows x blk
            G[:, j0:j1] = np.asarray(A.T @ A_blk, dtype=np.float64)
    else:
        A64 = np.asarray(A, dtype=np.float64)
        G = A64.T @ A64
    return G, b, P


def _fista_lasso(A, y, alpha, x0=None, max_iter=3000, tol=1e-7,
                 lipschitz=None, penalty_weights=None, _info=None,
                 gram=None, n_samples=None):
    """Solve min 0.5||A x - y||^2 + alpha * sum_j w_j |x_j| via FISTA.

    [FIX P26] Matvec-only LASSO so LASSO / ALASSO can run on a TwoLevelSM /
    LinearOperator and on genuinely-huge sparse matrices without ever
    building the dense sensing matrix; peak memory is ~O(n_features).
    Warm-started from x0 when given (the alpha path is warm-started from
    the previous alpha).

    penalty_weights (w_j) implements the adaptive-LASSO penalty directly in
    the ORIGINAL column space (per-coordinate soft-threshold) instead of
    column-scaling A by 1/w. That keeps the Lipschitz constant at ||A||^2
    rather than ||A / w||^2, so the ALASSO adaptive scaling no longer slows
    FISTA down.
    """
    n = A.shape[1]
    if n_samples is None:
        n_samples = A.shape[0]
    y64 = np.asarray(y, dtype=np.float64).ravel()
    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = np.asarray(x0, dtype=np.float64).copy()

    if alpha <= 0:
        # [FIX P34] alpha<=0 is never produced by derive_alpha_grid, so this only
        # matters for direct API use. It still solves A x = y via LSQR (not the
        # Gram normal equation G x = b, which squares the condition number).
        if _info is not None:
            _info["n_iter"] = 0
        return _solve_sparse_lsqr(A, y64)

    if gram is not None:
        G, b = gram
    else:
        G = b = None
    if lipschitz is None:
        lipschitz = _top_eigval(G) if gram is not None else _estimate_lipschitz(A)
    step = 1.0 / max(float(lipschitz), 1e-12)
    # sklearn LassoCV convention is (1/(2 n_samples))||Ax-y||^2 + alpha||x||_1,
    # which equals 0.5||Ax-y||^2 + (n_samples*alpha)||x||_1 -- so the L1 weight
    # in the proximal step is scaled by n_samples to match the alpha grid.
    thr = float(alpha) * n_samples * step
    if penalty_weights is not None:
        thr = thr * np.asarray(penalty_weights, dtype=np.float64).ravel()

    z = x.copy()          # momentum point y_0 = x_0
    t = 1.0
    x_prev = x.copy()
    n_iter = 0
    for it in range(int(max_iter)):
        n_iter = it + 1
        if gram is not None:
            # [FIX P34] gradient from the precomputed Gram: A^T(A z - y) = G z - b
            grad = np.asarray(G @ z, dtype=np.float64).ravel() - b
        else:
            Az = np.asarray(A @ z, dtype=np.float64).ravel()
            grad = np.asarray(A.T @ (Az - y64), dtype=np.float64).ravel()
        x_new = _soft_threshold(z - step * grad, thr)

        # FISTA with adaptive restart (O'Donoghue & Candes 2015): reset the
        # momentum when it points against the last step, which restores (near)
        # linear convergence on ill-conditioned problems like the ALASSO
        # adaptive-scaled matrix.
        if float(np.dot(z - x_new, x_new - x)) > 0.0:
            z = x_new
            t = 1.0
        else:
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            z = x_new + ((t - 1.0) / t_new) * (x_new - x)
            t = t_new
        x = x_new

        if it % 20 == 19:
            dx = float(np.linalg.norm(x - x_prev))
            if dx <= tol * max(1.0, float(np.linalg.norm(x))):
                break
            x_prev = x.copy()
    if _info is not None:
        _info["n_iter"] = n_iter
    return x


def _scale_operator(A, w):
    """LinearOperator for the column-scaled A[:, j] / w[j], no materialization."""
    inv_w = (1.0 / np.asarray(w, dtype=np.float64)).ravel()

    def mv(v):
        v = np.asarray(v, dtype=np.float64).ravel()
        return np.asarray(A @ (v * inv_w), dtype=np.float64).ravel()

    def rmv(u):
        u = np.asarray(u, dtype=np.float64).ravel()
        return (np.asarray(A.T @ u, dtype=np.float64).ravel()) * inv_w

    return LinearOperator(A.shape, matvec=mv, rmatvec=rmv, dtype=np.float64)


def _row_slice_op(A, rows):
    """LinearOperator for A[rows, :] without materializing A (used by CV folds).

    [FIX P30] buffers are float64 so the FISTA gradient stays in float64 even
    when the underlying operator is float32 (the matvec itself still follows the
    operator's dtype, which is PHEASY_SM_DTYPE).
    """
    n = A.shape[1]
    dt = np.dtype(np.float64)
    rows = np.asarray(rows, dtype=np.intp)

    def mv(v):
        v = np.asarray(v, dtype=dt).ravel()
        return np.asarray(A @ v, dtype=dt).ravel()[rows]

    def rmv(u):
        u = np.asarray(u, dtype=dt).ravel()
        u_full = np.zeros(A.shape[0], dtype=dt)
        u_full[rows] = u
        return np.asarray(A.T @ u_full, dtype=dt).ravel()

    return LinearOperator((len(rows), n), matvec=mv, rmatvec=rmv, dtype=dt)


def _row_slice(A, rows):
    """A[rows, :] for dense / sparse / LinearOperator.

    [FIX P30] TwoLevelSM gets a true row-slice (slices SM_prime) so CV-fold
    matvecs cost O(nnz(SM_prime[rows])) instead of the full O(nnz(SM_prime)).
    """
    if hasattr(A, "row_slice"):     # TwoLevelSM
        return A.row_slice(rows)
    if _is_linear_operator(A):
        return _row_slice_op(A, rows)
    return A[rows]


def _ridge_solve(A, y, alpha, x0=None):
    """min ||A x - y||^2 + alpha||x||^2 for dense / sparse / LinearOperator.

    ``x0`` is a warm-start for the iterative (LinearOperator) branch: the alpha
    path walks large->small, so neighbouring alphas have nearly identical
    solutions and LSMR converges in a fraction of the cold-start iterations.
    """
    y64 = np.asarray(y, dtype=np.float64).ravel()
    if _is_linear_operator(A):
        n = A.shape[1]
        sqrt_a = float(np.sqrt(alpha)) if alpha > 0 else 0.0
        if sqrt_a > 0:
            def mv_aug(v):
                v = np.asarray(v, dtype=np.float64).ravel()
                return np.concatenate(
                    [np.asarray(A @ v, dtype=np.float64).ravel(), sqrt_a * v])

            def rmv_aug(u):
                u = np.asarray(u, dtype=np.float64).ravel()
                return (np.asarray(A.T @ u[: A.shape[0]], dtype=np.float64).ravel()
                        + sqrt_a * u[A.shape[0]:])

            op = LinearOperator((A.shape[0] + n, n), matvec=mv_aug,
                                rmatvec=rmv_aug, dtype=np.float64)
            y_aug = np.concatenate([y64, np.zeros(n)])
        else:
            op, y_aug = A, y64
        # [FIX] dtype from the DATA-carrying operator A, not the wrapper
        # shell: augmented/masked/sliced LinearOperator shells declare dtype
        # = the OUTPUT array type (often float64), not the computation
        # precision (A may be float32 TwoLevelSM) -- shell dtype would give
        # an unreachable strict tolerance (LSMR always hits maxiter).
        _eps = np.finfo(np.dtype(getattr(A, "dtype", np.float64))).eps
        _dflt = "1e-8" if _eps < 1e-10 else "1e-6"
        atol = float(os.environ.get("PHEASY_LSQR_ATOL", _dflt))
        btol = float(os.environ.get("PHEASY_LSQR_BTOL", _dflt))
        maxiter = int(os.environ.get("PHEASY_LSQR_MAXITER", "5000"))
        res = _lsmr(op, y_aug, atol=atol, btol=btol, maxiter=maxiter, x0=x0)
        print("[LSMR] istop={} itn={} r1norm={:.3e} (atol={:.0e}, lim={})".format(
            res[1], res[2], res[3], atol, maxiter), flush=True)
        return np.asarray(res[0], dtype=np.float64)
    if alpha > 0:
        ridge = Ridge(alpha=alpha, fit_intercept=False, solver="auto")
        ridge.fit(A, y64)
        return ridge.coef_
    return _solve_lstsq(A, y64)



def _solve_subset(A, y, row_idx, col_idx, ridge_alpha=0.0, qr=False,
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
        op_base = _make_masked_op(A, row_idx, col_idx)
        op = op_base          # no ridge: solve the masked operator directly
        n = len(col_idx)
        # [FIX] dtype from the DATA-carrying operator A, not the wrapper
        # shell: augmented/masked/sliced LinearOperator shells declare dtype
        # = the OUTPUT array type (often float64), not the computation
        # precision (A may be float32 TwoLevelSM) -- shell dtype would give
        # an unreachable strict tolerance (LSMR always hits maxiter).
        _eps = np.finfo(np.dtype(getattr(A, "dtype", np.float64))).eps
        _dflt = "1e-8" if _eps < 1e-10 else "1e-6"
        atol = float(os.environ.get("PHEASY_LSQR_ATOL", str(
            lsmr_atol if lsmr_atol is not None else _dflt)))
        btol = float(os.environ.get("PHEASY_LSQR_BTOL", str(
            lsmr_btol if lsmr_btol is not None else _dflt)))
        maxiter = int(os.environ.get("PHEASY_LSQR_MAXITER", str(
            lsmr_maxiter if lsmr_maxiter is not None else 5000)))
        if ridge_alpha > 0:
            sqrt_a = float(np.sqrt(ridge_alpha))

            def mv_aug(v):
                v = np.asarray(v, dtype=np.float64).ravel()
                return np.concatenate([np.asarray(op_base @ v).ravel(),
                                       sqrt_a * v])

            def rmv_aug(u):
                u = np.asarray(u, dtype=np.float64).ravel()
                return (np.asarray(op_base.T @ u[: op_base.shape[0]]).ravel()
                        + sqrt_a * u[op_base.shape[0]:])

            op = LinearOperator((op_base.shape[0] + n, n), matvec=mv_aug,
                                rmatvec=rmv_aug, dtype=np.float64)
            y_sub = np.concatenate([y_sub, np.zeros(n)])
        res = _lsmr(op, y_sub, atol=atol, btol=btol, maxiter=maxiter)
        print("[LSMR] istop={} itn={} r1norm={:.3e} (atol={:.0e}, lim={})".format(
            res[1], res[2], res[3], atol, maxiter), flush=True)
        return np.asarray(res[0], dtype=np.float64)

    A_sub = A[:, col_idx]
    if row_idx is not None:
        A_sub = A_sub[row_idx]
    if ridge_alpha > 0:
        ridge = Ridge(alpha=ridge_alpha, fit_intercept=False, solver="lsqr")
        ridge.fit(A_sub, y_sub)
        return ridge.coef_
    if qr:
        # [FIX P45] TSQR only pays off when the dense block does not fit in
        # memory. On a small matrix (the common case) the whole thing is one
        # block, so the tree-reduction pays all of its bookkeeping overhead for
        # zero gain AND parallelizes worse than a single LAPACK gelsd. Judge by
        # BYTES (not rows): gelsd must materialize A_sub plus its workspace, so
        # the same row count is a very different footprint at p=200 vs p=4000.
        # PHEASY_TSQR_FORCE=1 keeps the QR path for A/B cross-checks.
        _bytes = A_sub.shape[0] * A_sub.shape[1] * 8
        _thr = float(os.environ.get("PHEASY_TSQR_MIN_BYTES", "8e9"))
        if _bytes <= _thr and os.environ.get("PHEASY_TSQR_FORCE", "0") != "1":
            return _solve_lstsq(A_sub, y_sub)
        return _solve_qr(A_sub, y_sub, block_rows=block_rows,
                         diag_floor=diag_floor)
    return _solve_lstsq(A_sub, y_sub)


def _predict_subset(A, col_idx, row_idx, coef):
    """A[row_idx][:, col_idx] @ coef."""
    if _is_linear_operator(A):
        return np.asarray(_make_masked_op(A, row_idx, col_idx) @ coef).ravel()
    A_sub = A[:, col_idx]
    if row_idx is not None:
        A_sub = A_sub[row_idx]
    return np.asarray(A_sub @ coef).ravel()


def _cv_rmse(A, y, idx, solve, splits, n_jobs=1):
    """K-fold CV RMSE (mean, standard error, per-fold) for active columns idx.

    [FIX P45] folds are independent, so parallelize them with THREADS (not the
    default loky processes): each solve slices A[:, col_idx][row_idx] and a
    process worker would copy that block N ways (OOM on many-core hosts);
    threads share A, and the time is spent in LAPACK/scipy where the GIL is
    released anyway.
    """
    y64 = np.asarray(y, dtype=np.float64).ravel()

    def _one(tr, va):
        coef = solve(idx, tr)
        err = _predict_subset(A, idx, va, coef) - y64[va]
        return float(np.sqrt(np.mean(err * err)))

    if n_jobs > 1 and len(splits) > 1:
        from joblib import Parallel, delayed
        with _blas_limit(min(n_jobs, len(splits))):
            fold = np.asarray(Parallel(n_jobs=min(n_jobs, len(splits)),
                                       prefer="threads")(
                delayed(_one)(tr, va) for tr, va in splits), dtype=np.float64)
    else:
        fold = np.asarray([_one(tr, va) for tr, va in splits], dtype=np.float64)

    mean = float(fold.mean())
    se = float(fold.std(ddof=1) / np.sqrt(len(fold))) if len(fold) > 1 else 0.0
    return mean, se, fold


def _select_1se(history):
    """Select (n_active, mean, se) by the one-standard-error rule (or argmin)."""
    use_1se = os.environ.get("PHEASY_RFE_1SE", "1").lower() in ("1", "true", "yes")
    if not history:
        raise ValueError("empty RFE history")
    best_idx = int(np.argmin([h[1] for h in history]))
    if not use_1se:
        return history[best_idx]
    thr = history[best_idx][1] + history[best_idx][2]
    cands = [h for h in history if h[1] <= thr]
    return min(cands, key=lambda h: h[0])


class TwoLevelSM(LinearOperator):
    """Behave like SM = SM_prime @ NS without materializing the product.

    matvec:  SM @ v   = SM_prime @ (NS @ v)
    rmatvec: SM.T @ u = NS.T @ (SM_prime.T @ u)
    """

    def __init__(self, SM_prime, NS, dtype=None):
        self.SM_prime = SM_prime
        self.NS = NS
        dt = dtype if dtype is not None else SM_prime.dtype
        self._dt = dt
        self._matvec_nthr = min(
            int(os.environ.get("PHEASY_SM_MATVEC_THREADS", "32")),
            max(1, _avail_cores()))
        # [FIX] TRUE zero-copy row blocks: CSR slicing is a COPY (45GB per
        # slice; slicing inside matvec was 8.3x the matvec cost). Hand-
        # assemble each block so data/indices are numpy VIEWS sharing memory
        # with SM_prime (verified shares_memory=True); only indptr (MBs) is
        # new. NOTE: assign attrs directly -- passing arrays to the csr
        # constructor would copy them (the constructor copies anything it is
        # given). Also pre-transpose to CSC once: csr.T is ANOTHER O(nnz)
        # copy per call (csc data is reordered), so doing b.T inside
        # _sm_rmatvec would pay a 45GB transpose EVERY iteration.
        n = SM_prime.shape[0]
        self._blocks = None
        self._blocksT = None
        self._blk = n
        if self._matvec_nthr > 1 and n >= 10000:
            self._blk = int(np.ceil(n / self._matvec_nthr))
            self._blocks = []
            for i in range(self._matvec_nthr):
                r0 = i * self._blk
                r1 = min(r0 + self._blk, n)
                p0, p1 = SM_prime.indptr[r0], SM_prime.indptr[r1]
                b = sp.csr_matrix((1, 1), dtype=SM_prime.dtype)  # empty shell
                b.data = SM_prime.data[p0:p1]        # numpy VIEW, no copy
                b.indices = SM_prime.indices[p0:p1]  # numpy VIEW, no copy
                b.indptr = (SM_prime.indptr[r0:r1 + 1] - p0).astype(SM_prime.indptr.dtype)
                b._shape = (r1 - r0, SM_prime.shape[1])
                self._blocks.append(b)
            # CSC transpose once (construction-time cost only).
            self._blocksT = [b.T for b in self._blocks]
        super().__init__(np.dtype(dt), (SM_prime.shape[0], NS.shape[1]))

    def _sm_matvec(self, x):
        """Row-blocked threaded SM_prime @ x (blocks are zero-copy views)."""
        if self._blocks is None:
            return self.SM_prime @ x
        out = [None] * self._matvec_nthr
        def _do(i):
            out[i] = self._blocks[i] @ x
        th = [threading.Thread(target=_do, args=(i,)) for i in range(self._matvec_nthr)]
        for t in th: t.start()
        for t in th: t.join()
        return np.concatenate(out)

    def _sm_rmatvec(self, y):
        """Row-blocked threaded SM_prime.T @ y on pre-transposed CSC blocks;
        float64 cross-block sum (a float32 sum floors at ~1e-5 and would make
        atol=1e-6 unreachable), then cast back to the operator dtype so
        matvec and rmatvec outputs agree with the declared A.dtype (BUG3)."""
        if self._blocksT is None:
            return self.SM_prime.T @ y
        out = [None] * self._matvec_nthr
        def _do(i):
            r0 = i * self._blk
            r1 = min(r0 + self._blk, self.SM_prime.shape[0])
            out[i] = self._blocksT[i] @ y[r0:r1]
        th = [threading.Thread(target=_do, args=(i,)) for i in range(self._matvec_nthr)]
        for t in th: t.start()
        for t in th: t.join()
        acc = np.zeros(self.SM_prime.shape[1], dtype=np.float64)
        for b in out:
            acc += np.asarray(b, dtype=np.float64)
        return np.ascontiguousarray(acc, dtype=self._dt)

    def _matvec(self, v):
        v = np.ascontiguousarray(v, dtype=self._dt)
        t = self.NS @ v
        return self._sm_matvec(np.ascontiguousarray(t, dtype=self._dt))

    def _rmatvec(self, u):
        u = np.ascontiguousarray(u, dtype=self._dt)
        t = self._sm_rmatvec(u)
        return self.NS.T @ np.ascontiguousarray(t, dtype=self._dt)

    def col_norms(self):
        """Exact ||SM[:, j]|| (the true sensing-matrix column norms)."""
        return _col_norms(self)

    def row_slice(self, rows):
        """[FIX P30] TwoLevelSM for A[rows, :] by slicing SM_prime only.

        This is O(nnz(SM_prime[rows])) per matvec instead of the full
        O(nnz(SM_prime)) that the generic _row_slice_op wrapper pays, so a
        K-fold CV costs ~1x the full problem rather than ~Kx.
        """
        return TwoLevelSM(self.SM_prime[rows], self.NS, dtype=self._dt)

    def to_dense(self):
        return _to_dense_f64(self)



def _reselect_alpha(model, A, y, sample_weight=None, grid_diag=None):
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
    # [FIX P43] a flat CV tail is only a CONVERGENCE problem if coordinate
    # descent actually hit its iteration cap; otherwise it converged and the tail
    # is genuinely flat (the alphas simply do not separate).
    # [FIX P43/P44] sklearn's LassoCV.n_iter_ is the FINAL refit's iteration
    # count at the chosen alpha*, NOT the CV-path fits that produced the tail --
    # so it cannot certify that the tail is genuine. Only a hit cap
    # (n_iter >= max_iter) is assertable; otherwise we state the path is not
    # ruled out and ask for a tighter --tol to confirm (the flat tail often
    # disappears once tol is tightened).
    _n_iter = int(np.max(np.atleast_1d(model.n_iter_)))
    _hit_cap = _n_iter >= int(model.max_iter)
    if tied.size > 1:
        if _hit_cap:
            print("[CV] WARNING: %d alphas tie at CV MSE %.6e (%.3e ... %.3e). "
                  "Coordinate descent hit max_iter (%d); lower --tol (1e-6) "
                  "and/or raise --max_iter, otherwise the fit is over-regularized."
                  % (tied.size, best, float(alphas[tied].min()),
                     float(alphas[tied].max()), _n_iter), flush=True)
        else:
            print("[CV] WARNING: %d alphas tie at CV MSE %.6e (%.3e ... %.3e). "
                  "The final refit converged in %d iters, but sklearn does not "
                  "expose the CV-path iteration counts, so a tolerance-limited "
                  "path is not ruled out -- re-run with a tighter --tol to "
                  "confirm." % (tied.size, best, float(alphas[tied].min()),
                                float(alphas[tied].max()), _n_iter), flush=True)
    # [FIX P39/P41/P42/P43] alpha* pinned to the grid MINIMUM has FOUR causes, in
    # priority order: (1) the whole grid is below the weighted KKT threshold -- a
    # GRID-scale problem (manual --no-alpha_auto); (2) a flat tail AND CD hit
    # max_iter -- a CONVERGENCE problem; (3) a flat tail but CD converged -- alpha*
    # is just not well-determined; (4) the curve genuinely still falls -- a model-
    # density conclusion (treat as unregularized / compare OLS). Stash the flags so
    # run_pheasy can defer instead of re-asserting a cause it cannot see.
    model._alpha_at_min = new_alpha <= float(alphas.min()) * (1.0 + 1e-12)
    model._alpha_at_min_flat = (
        model._alpha_at_min and tied.size > 1
        and float(alphas[tied].min()) <= float(alphas.min()) * (1.0 + 1e-12))
    model._alpha_at_min_hitcap = model._alpha_at_min_flat and _hit_cap
    if model._alpha_at_min:
        if grid_diag:
            print("[CV] WARNING: alpha* %.3e sits at the grid MINIMUM; %s"
                  % (new_alpha, grid_diag), flush=True)
        elif model._alpha_at_min_hitcap:
            print("[CV] WARNING: alpha* %.3e sits at the grid MINIMUM via the "
                  "tie-break on a FLAT CV tail AND CD hit max_iter; this is a "
                  "CONVERGENCE problem (lower --tol (1e-6) and/or raise "
                  "--max_iter), not a model-density conclusion." % new_alpha,
                  flush=True)
        elif model._alpha_at_min_flat:
            print("[CV] WARNING: alpha* %.3e sits at the grid MINIMUM on a flat CV "
                  "tail; the final refit converged in %d iters, but sklearn does "
                  "not expose the CV-path iteration counts, so a tolerance-limited "
                  "path is not ruled out -- re-run with a tighter --tol to confirm."
                  % (new_alpha, _n_iter), flush=True)
        else:
            print("[CV] WARNING: alpha* %.3e sits at the grid MINIMUM; the CV curve "
                  "is still falling at the low end, so widening the grid only pushes "
                  "alpha* toward OLS. Treat this fit as effectively unregularized "
                  "(compare with OLS/RFE)." % new_alpha, flush=True)
    if new_alpha == float(model.alpha_):
        return
    print("[CV] alpha reselected: %.6e -> %.6e" % (float(model.alpha_), new_alpha),
          flush=True)
    from sklearn.linear_model import Lasso as _Lasso
    est = _Lasso(alpha=new_alpha, fit_intercept=model.fit_intercept,
                 max_iter=model.max_iter, tol=model.tol,
                 selection=getattr(model, "selection", "cyclic"),
                 random_state=getattr(model, "random_state", None))
    est.fit(A, y, sample_weight=sample_weight)  # [FIX] keep sample_weight in the refit
    model.alpha_ = new_alpha
    model.coef_ = est.coef_
    model.intercept_ = est.intercept_
    model.n_iter_ = int(np.max(np.atleast_1d(est.n_iter_)))


class _OLSModel:
    def __init__(self, coef, n_iter=None, alpha=None):
        self.coef_ = np.asarray(coef)
        self.intercept_ = 0.0
        self.n_features_in_ = self.coef_.shape[0]
        self.n_iter_ = n_iter
        # [FIX P47] RIDGE stores an _OLSModel (no sklearn model), but
        # fit.py / holdout_eval.py read alpha_ for the grid-edge flag. Expose it
        # here as a defensive mirror of o.results["alpha"] (the formal exit).
        self.alpha_ = alpha

    def predict(self, A):
        return np.asarray(A @ self.coef_).ravel()


def _lasso_n_jobs(A):
    """[FIX P24] Cap sklearn LassoCV process parallelism by matrix size.

    LassoCV(n_jobs=N) fans the (alpha, fold) grid out to N loky *processes*;
    each worker copies the centered training fold of the dense design matrix,
    so N=-1 (one worker per core) on a many-core host blows up memory and the
    OOM killer SIGTERMs the run. Default to a memory-aware, bounded worker count.
    """
    n = _resolve_n_jobs("LASSO")
    if n == 1:
        return 1
    n_cpu = _avail_cores()
    n = min(n, n_cpu)
    try:
        per_worker = int(A.shape[0]) * int(A.shape[1]) * 8
        budget = int(float(os.environ.get("PHEASY_LASSO_MEM_GB", "6"))) * 2 ** 30
        cap = max(1, int(budget // max(per_worker, 1)))
    except Exception:
        cap = 4
    # [FIX P24b] the 16-worker ceiling was hardcoded; it underuses a many-core
    # box. Keep a memory cap but let the ceiling be raised explicitly.
    _hi = int(os.environ.get("PHEASY_LASSO_MAX_WORKERS", "16"))
    return max(1, min(n, cap, _hi))


def _predict_rows(A, coef, rows):
    """A[rows] @ coef for dense / sparse / LinearOperator (one matvec)."""
    return np.asarray(A @ coef, dtype=np.float64).ravel()[rows]


class _LassoCVIterative:
    """LASSO over an alpha grid with grouped CV via the matvec-only FISTA solver.

    [FIX P26] Drop-in for _LassoCVModel when the design matrix is a
    LinearOperator or a sparse matrix too large to densify (see
    _lasso_backend). Never materializes the sensing matrix; peak memory is
    ~O(n_features). The alpha path is walked large->small and warm-started.

    NOTE: this is a memory-for-time trade-off. FISTA converges O(1/k^2) and each
    TwoLevelSM matvec is two sparse multiplies, so it can be 10-100x slower than
    the dense sklearn path. It is auto-selected only when the dense SM does not
    fit in memory (_lasso_backend); PHEASY_LASSO_TWOLEVEL stays off by default.
    """
    def __init__(self, alphas, cv, tol, max_iter, rand_seed, n_jobs=None,
                 fit_intercept=False, group_size=None, selection="cyclic",
                 penalty_weights=None, grid_diag=None):
        # [FIX P40] sort ascending: the alpha walk assumes smallest-first and
        # the grid-MINIMUM edge check (best_i == 0) depends on it. Callers pass
        # logspace/derive grids that are ascending, but be explicit rather than
        # relying on that contract.
        self.alphas = np.sort(np.asarray(alphas, dtype=np.float64))
        self.cv = cv
        self.tol = tol
        self.max_iter = int(max_iter)
        self.rand_seed = rand_seed
        # [FIX P45] thread-fold parallelism, resolved independently of the
        # dense sklearn _lasso_n_jobs memory cap (threads share A, so there is
        # no per-worker matrix copy to budget against).
        self.n_jobs = _resolve_n_jobs("LASSO", n_jobs if n_jobs is not None else -1)
        self.fit_intercept = fit_intercept
        self.group_size = group_size
        self.selection = selection
        self.penalty_weights = penalty_weights
        # [FIX P44] manual-grid scale-mismatch diagnosis (message string from
        # _AdaptiveLassoCV.fit); used to give the GRID, not the data, as the
        # cause of a pinned alpha*.
        self.grid_diag = grid_diag
        self._lipschitz = None

    def fit(self, A, y, sample_weight=None):
        y64 = np.asarray(y, dtype=np.float64).ravel()
        if sample_weight is not None and not np.allclose(sample_weight, sample_weight[0]):
            # [FIX P26] the matvec-only FISTA path does not yet support weighted
            # data; the dense sklearn path does.  sample_weight is always None in
            # the pheasy CLI, so this only guards direct Optimizer API use.
            print("[optimizer] WARNING: iterative LASSO ignores non-uniform "
                  "sample_weight (dense path supports it).", flush=True)
        splits = _make_cv_splits(A.shape[0], self.cv, self.rand_seed,
                                 self.group_size)
        n_alphas = len(self.alphas)

        # [FIX P34] build the Gram A^T A (and per-fold variants) when it fits the
        # memory budget, then FISTA runs a dense n_features x n_features matvec
        # per iteration instead of two sparse multiplies over the full SM. On a
        # budget miss (or a build error) fall back to the matvec path.
        use_gram = False
        gram_full = None
        gram_folds = None
        lip_full = None
        lip_folds = None
        A_va_list = None
        if _gram_budget_ok(A):
            try:
                G_full, b_full, _P = _compute_gram(A, y64)
                gram_full = (G_full, b_full)
                lip_full = _top_eigval(G_full)
                gram_folds = []
                lip_folds = []
                # [FIX P34] A_va_list holds the K disjoint VALIDATION slices
                # (~1x SM_prime total, not the (K-1)x of the P33 train-fold
                # cache); it pays for the per-fold prediction A[va] @ coef.
                # G_tr = G_full - G_va: safe for K>=3; for K=2 / LOOCV the
                # cancellation (G_va ~ G_full) can leave G_tr slightly non-PSD
                # (harmless to eigvalsh lambda_max, but noted).
                A_va_list = []
                for _tr, va in splits:
                    A_va = _row_slice(A, va)
                    G_va, b_va, _ = _compute_gram(A_va, y64[va])
                    G_tr = G_full - G_va
                    gram_folds.append((G_tr, b_full - b_va))
                    lip_folds.append(_top_eigval(G_tr))
                    A_va_list.append(A_va)
                use_gram = True
                print("[optimizer] Gram path: G=%dx%d built (PHEASY_GRAM_MAX_GB=%s); "
                      "per-iter cost is a dense matvec, no full-SM multiply."
                      % (G_full.shape[0], G_full.shape[1],
                         os.environ.get("PHEASY_GRAM_MAX_GB", "4")), flush=True)
            except Exception as _e:
                print("[optimizer] WARNING: Gram build failed (%s); falling back "
                      "to matvec FISTA." % _e, flush=True)
                use_gram = False
        if use_gram:
            self._lipschitz = lip_full
            self._gram = gram_full
        else:
            self._lipschitz = _estimate_lipschitz(A)
            self._gram = None

        # CV folds only need to RANK the alphas, so a loose tolerance and a low
        # iteration budget suffice; the final refit uses the caller's tight tol.
        # [FIX P32] the knobs are exposed because a flat CV tail (see the tie
        # warning below) is fixed by tightening these, not by --tol (which only
        # affects the final refit).
        cv_tol = float(os.environ.get(
            "PHEASY_CV_TOL", str(max(float(self.tol), 1e-3))))
        cv_max_iter = int(os.environ.get(
            "PHEASY_CV_MAX_ITER", str(min(self.max_iter, 800))))

        # [FIX P33] hoist the per-fold row slices out of the alpha loop so the
        # CSR / TwoLevelSM slicing is done once instead of n_alphas times.  Off
        # by default: caching K folds keeps ~(K-1)x SM_prime resident; enable on
        # big-but-not-huge systems via PHEASY_CV_CACHE_FOLDS=1.
        _cache_folds = os.environ.get("PHEASY_CV_CACHE_FOLDS", "0").lower() in ("1", "true", "yes")
        A_folds = None
        if _cache_folds:
            A_folds = [_row_slice(A, tr) if _is_linear_operator(A) else A[tr]
                       for tr, _ in splits]

        mse_path = np.zeros((n_alphas, len(splits)))
        x_folds = [None] * len(splits)   # [FIX P28] per-fold warm-start
        x_full = None                    # full-data warm-start for the alpha path
        best_i = 0
        best_mean = float("inf")
        best_x = None
        # [FIX P43] track the max FISTA iterations used by the CV solves so a flat
        # tail can be attributed to hitting cv_max_iter (convergence) vs a genuinely
        # flat curve (converged).
        _cv_info = {}
        _cv_max_n_iter = 0

        n_jobs = min(self.n_jobs, len(splits))
        for a_i in range(n_alphas - 1, -1, -1):  # descending: large alpha first
            alpha = float(self.alphas[a_i])

            # [FIX P45] the alpha path is a warm-start chain (each fold's
            # solution seeds the next alpha), so only FOLDS -- never alphas --
            # may run in parallel. Each fold reads/writes its own x_folds[k] and
            # its own _info dict, so the workers share no mutable state.
            def _fold_fit(k):
                tr, va = splits[k]
                info = {}
                if use_gram:
                    # [FIX P34] the Gram encodes A[tr], so pass the TRAIN fold's
                    # row count: the L1 threshold is (n_tr * alpha), not
                    # (n_full * alpha) -- otherwise the fold fits at ~(K/(K-1))x
                    # the intended effective alpha.
                    coef = _fista_lasso(A, y64, alpha, x0=x_folds[k],
                                        max_iter=cv_max_iter, tol=cv_tol,
                                        lipschitz=lip_folds[k],
                                        penalty_weights=self.penalty_weights,
                                        gram=gram_folds[k],
                                        n_samples=len(tr), _info=info)
                    pred = np.asarray(A_va_list[k] @ coef, dtype=np.float64).ravel()
                else:
                    if A_folds is not None:
                        A_tr = A_folds[k]
                    else:
                        A_tr = _row_slice(A, tr) if _is_linear_operator(A) else A[tr]
                    # [FIX P28] warm-start from THIS fold's previous-alpha solution,
                    # not the full-data solution: with a loose CV budget the warm
                    # start does not fully wash out, so x_full would leak the
                    # validation rows into the fold fit and bias CV low.
                    coef = _fista_lasso(A_tr, y64[tr], alpha, x0=x_folds[k],
                                        max_iter=cv_max_iter, tol=cv_tol,
                                        lipschitz=self._lipschitz,
                                        penalty_weights=self.penalty_weights,
                                        _info=info)
                    pred = _predict_rows(A, coef, va)
                err = pred - y64[va]
                return float(np.mean(err * err)), coef, int(info.get("n_iter", 0))

            if n_jobs > 1:
                from joblib import Parallel, delayed
                with _blas_limit(n_jobs):
                    results = Parallel(n_jobs=n_jobs, prefer="threads")(
                        delayed(_fold_fit)(k) for k in range(len(splits)))
            else:
                results = [_fold_fit(k) for k in range(len(splits))]
            fold_mse = np.zeros(len(splits))
            for k, (mse_k, coef, n_it) in enumerate(results):
                fold_mse[k] = mse_k
                x_folds[k] = coef
                _cv_max_n_iter = max(_cv_max_n_iter, n_it)
            mse_path[a_i] = fold_mse
            mean = float(fold_mse.mean())
            # warm-start the next (smaller) alpha from this alpha's full fit
            if use_gram:
                x_full = _fista_lasso(A, y64, alpha, x0=x_full,
                                      max_iter=cv_max_iter, tol=cv_tol,
                                      lipschitz=lip_full,
                                      penalty_weights=self.penalty_weights,
                                      gram=gram_full, _info=_cv_info)
            else:
                x_full = _fista_lasso(A, y64, alpha, x0=x_full,
                                      max_iter=cv_max_iter, tol=cv_tol,
                                      lipschitz=self._lipschitz,
                                      penalty_weights=self.penalty_weights,
                                      _info=_cv_info)
            _cv_max_n_iter = max(_cv_max_n_iter, _cv_info.get("n_iter", 0))
            # [FIX P27] <= (not <) so a tie picks the SMALLEST alpha (the one
            # seen LAST in the descending walk), matching _reselect_alpha's
            # tie-break toward the least-regularized member.
            if mean <= best_mean:
                best_mean = mean
                best_i = a_i
                best_x = x_full.copy()

        # [FIX P27] port _reselect_alpha's tie warning: a flat CV tail means the
        # loose CV solver did not separate the alphas and the choice is suspect.
        mean_path = mse_path.mean(axis=1)
        rtol = float(os.environ.get("PHEASY_LASSO_TIE_RTOL", "1e-9"))
        tied = np.flatnonzero(mean_path <= best_mean * (1.0 + rtol) + 1e-300)
        _cv_hit_cap = _cv_max_n_iter >= cv_max_iter
        if tied.size > 1:
            if _cv_hit_cap:
                print("[CV] WARNING: %d alphas tie at CV MSE %.6e (%.3e ... %.3e). "
                      "The CV solver (FISTA, tol=%.0e) hit cv_max_iter (%d) and is "
                      "too loose to separate them -- lower PHEASY_CV_TOL / raise "
                      "PHEASY_CV_MAX_ITER."
                      % (tied.size, best_mean, float(self.alphas[tied].min()),
                         float(self.alphas[tied].max()), cv_tol, cv_max_iter),
                      flush=True)
            else:
                print("[CV] WARNING: %d alphas tie at CV MSE %.6e (%.3e ... %.3e); "
                      "FISTA already converged (max %d iters < %d), so the CV tail "
                      "is genuinely flat."
                      % (tied.size, best_mean, float(self.alphas[tied].min()),
                         float(self.alphas[tied].max()), _cv_max_n_iter,
                         cv_max_iter), flush=True)
        # [FIX P39/P41/P42/P43/P44] best_i == 0 is the SMALLEST alpha (the grid
        # walks descending). Four causes, in priority order: (1) manual-grid scale
        # mismatch -- a GRID-scale problem; (2) flat tail AND FISTA hit cv_max_iter
        # -- CONVERGENCE; (3) flat tail but FISTA converged -- alpha* not
        # well-determined; (4) the curve still falls -- model-density (treat as
        # unregularized / compare OLS). NOTE: unlike the sklearn path (whose n_iter_
        # is the final refit count and cannot certify convergence), _cv_max_n_iter
        # HERE measures the actual CV-path FISTA iterations, so the 'FISTA already
        # converged' branch is a real assertion.
        self._alpha_at_min = (best_i == 0)
        self._alpha_at_min_flat = (
            self._alpha_at_min and tied.size > 1
            and float(self.alphas[tied].min()) <= float(self.alphas.min()) * (1.0 + 1e-12))
        self._alpha_at_min_hitcap = self._alpha_at_min_flat and _cv_hit_cap
        if self._alpha_at_min:
            if self.grid_diag:
                print("[CV] WARNING: alpha* %.3e sits at the grid MINIMUM; %s"
                      % (float(self.alphas[0]), self.grid_diag), flush=True)
            elif self._alpha_at_min_hitcap:
                print("[CV] WARNING: alpha* %.3e sits at the grid MINIMUM via the "
                      "tie-break on a FLAT CV tail AND FISTA hit cv_max_iter; this "
                      "is a CONVERGENCE problem (lower PHEASY_CV_TOL / raise "
                      "PHEASY_CV_MAX_ITER), not a model-density conclusion."
                      % float(self.alphas[0]), flush=True)
            elif self._alpha_at_min_flat:
                print("[CV] WARNING: alpha* %.3e sits at the grid MINIMUM on a flat "
                      "CV tail, but FISTA already converged (max %d iters): alpha* "
                      "is not well-determined by CV (not a convergence problem)."
                      % (float(self.alphas[0]), _cv_max_n_iter), flush=True)
            else:
                print("[CV] WARNING: alpha* %.3e sits at the grid MINIMUM; the CV "
                      "curve is still falling at the low end, so widening the grid "
                      "only pushes alpha* toward OLS. Treat this fit as effectively "
                      "unregularized (compare with OLS/RFE)."
                      % float(self.alphas[0]), flush=True)

        self.alpha_ = float(self.alphas[best_i])
        # final refit at the chosen alpha, warm-started from the path. Cap the
        # iteration budget (FISTA is O(1/k^2), and the scaled ALASSO matrix is
        # more ill-conditioned); 5000 warm-started steps already reach ~1e-5.
        _finfo = {"n_iter": 0}
        self.coef_ = _fista_lasso(A, y64, self.alpha_, x0=best_x,
                                  max_iter=min(self.max_iter, 5000),
                                  tol=max(float(self.tol), 1e-7),
                                  lipschitz=lip_full if use_gram else self._lipschitz,
                                  penalty_weights=self.penalty_weights,
                                  _info=_finfo,
                                  gram=gram_full if use_gram else None)
        self.intercept_ = 0.0
        self.alphas_ = self.alphas
        self.mse_path_ = mse_path
        self.n_iter_ = int(_finfo.get("n_iter", 0))
        self.n_features_in_ = A.shape[1]
        return self

    def predict(self, A):
        pred = np.asarray(A @ self.coef_).ravel()
        if self.fit_intercept:
            pred = pred + self.intercept_
        return pred

class _LassoCVModel:
    """Thin wrapper around sklearn LassoCV with correct alpha grid and grouped CV."""

    def __init__(self, alphas, cv, tol, max_iter, rand_seed, n_jobs,
                 fit_intercept=False, group_size=None, selection="cyclic"):
        self.alphas = np.asarray(alphas, dtype=np.float64)
        self.cv = cv
        self.tol = tol
        self.max_iter = max_iter
        self.rand_seed = rand_seed
        self.n_jobs = n_jobs
        self.fit_intercept = fit_intercept
        self.group_size = group_size
        self.selection = selection

    def fit(self, A, y, sample_weight=None):
        if _lasso_backend(A) == "iterative":
            it = _LassoCVIterative(
                self.alphas, self.cv, self.tol, self.max_iter, self.rand_seed,
                None, fit_intercept=self.fit_intercept,
                group_size=self.group_size, selection=self.selection)
            it.fit(A, y, sample_weight=sample_weight)
            self.model_ = it
            self.coef_ = it.coef_
            self.intercept_ = it.intercept_
            self.alpha_ = it.alpha_
            self.alphas_ = it.alphas_
            self.mse_path_ = it.mse_path_
            self.n_iter_ = it.n_iter_
            self.n_features_in_ = it.n_features_in_
            self._alpha_at_min = getattr(it, "_alpha_at_min", False)
            self._alpha_at_min_flat = getattr(it, "_alpha_at_min_flat", False)
            self._alpha_at_min_hitcap = getattr(it, "_alpha_at_min_hitcap", False)
            return self

        n_samples = A.shape[0]
        splits = _make_cv_splits(n_samples, self.cv, self.rand_seed, self.group_size)
        model = LassoCV(
            alphas=self.alphas,
            cv=splits,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
            random_state=self.rand_seed,
            selection=self.selection,
            n_jobs=self.n_jobs,
        )
        model.fit(A, y, sample_weight=sample_weight)
        self.model_ = model
        _reselect_alpha(model, A, y, sample_weight=sample_weight)  # [FIX P10]
        self.coef_ = model.coef_
        self.intercept_ = model.intercept_
        self.alpha_ = model.alpha_
        self.alphas_ = np.asarray(model.alphas_)
        self.mse_path_ = np.asarray(model.mse_path_)
        self.n_iter_ = int(model.n_iter_)
        self.n_features_in_ = A.shape[1]
        self._alpha_at_min = getattr(model, "_alpha_at_min", False)
        self._alpha_at_min_flat = getattr(model, "_alpha_at_min_flat", False)
        self._alpha_at_min_hitcap = getattr(model, "_alpha_at_min_hitcap", False)
        return self

    def predict(self, A):
        pred = np.asarray(A @ self.coef_).ravel()
        if self.fit_intercept:
            pred = pred + self.intercept_
        return pred


class _AdaptiveLassoCV(_LassoCVModel):
    """Adaptive LASSO (Zou 2006).

    Stage 1: ridge initial estimate beta0.
    Stage 2: weights w_j = 1/(|beta0_j| + eps)^gamma, then column-scaled LASSO.
    """

    def __init__(self, *args, gamma=1.0, init_alpha=1e-3, eps=1e-8,
                 nalpha=None, decades=4.0, alpha_auto=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = float(gamma)
        self.init_alpha = float(init_alpha)
        self.eps = float(eps)
        self.nalpha = int(nalpha) if nalpha else None
        self.decades = float(decades)
        # [FIX P38] the weighted-space grid is an AUTO-mode behavior: it must
        # not clobber a user-supplied manual --mu_min/--mu_max grid. run_pheasy
        # only derives the LASSO grid when --alpha_auto is on; ALASSO must
        # respect the same flag instead of unconditionally overriding.
        self.alpha_auto = bool(alpha_auto)
        self._weights = None

    def _initial_estimate(self, A, y):
        # [FIX P26] _ridge_solve handles dense / sparse / LinearOperator, so the
        # adaptive weights are available on the two-level operator too (LSMR on
        # the augmented system for operators, cholesky/svd for dense).
        return _ridge_solve(A, y, self.init_alpha)

    def fit(self, A, y, sample_weight=None):
        n_samples = A.shape[0]
        beta0 = self._initial_estimate(A, y)
        self._weights = 1.0 / (np.abs(beta0) + self.eps) ** self.gamma
        # [FIX] eps-floor fraction: the fraction of pilot coefficients at/below
        # eps. Underdetermined ridge pilots can either FLATTEN (weights ~uniform)
        # or SATURATE at the 1/eps ceiling -- two opposite failure modes that a
        # single weight-dispersion number cannot separate. 0.0 = flattened,
        # ~1.0 = most weights pinned at the 1/eps ceiling.
        self._beta0_floor = float(np.mean(np.abs(beta0) < self.eps))

        # [FIX P35] derive the alpha grid in the WEIGHTED space:
        # (A/w)^T y = (A^T y)/w, so alpha_max = max_j |(A^T y)_j / w_j| / n.
        # This replaces the unweighted grid + hardcoded mu_shift=-2 and is the
        # right grid for BOTH the scaled (LassoCV on A/w) and the penalized
        # (FISTA with per-coordinate penalty w_j) ALASSO forms.
        _wgrid = os.environ.get("PHEASY_ALASSO_WEIGHTED_GRID", "1").lower() in ("1", "true", "yes")
        # [FIX P38/P39/P40] three grid modes, so the logging and the matvec count
        # are each honest:
        #   weighted-auto: _wgrid & alpha_auto      -> derive the weighted KKT grid
        #                  here (one rmatvec).
        #   mu_shift-auto: (not _wgrid) & alpha_auto -> run_pheasy already derived a
        #                  mu_shift grid (one rmatvec THERE); do NOT redo it here.
        #   manual:        not alpha_auto           -> user --mu_min/--mu_max grid,
        #                  used AS-IS (diagnostic rmatvec only).
        _override = bool(self.nalpha and _wgrid and self.alpha_auto)
        _manual = bool(self.nalpha and not self.alpha_auto)
        _mu_shift = bool(self.nalpha and self.alpha_auto and not _wgrid)
        _a_uw = _a_max = 0.0
        _grid_diag = None   # [FIX P44] manual-grid scale-mismatch diagnosis string
        if _override:
            # [FIX P39/P40] one rmatvec for BOTH the weighted KKT threshold and the
            # unweighted one, kept in A's dtype so a float32 sensing matrix is not
            # silently promoted to float64 (which doubles peak memory). NOTE: a
            # float32 matvec accumulates in float32 (~1e-5 relative at n~1e5 rows);
            # acceptable because these thresholds only set grid ENDPOINTS.
            _A_dt = A.dtype if hasattr(A, "dtype") else np.float64
            _g_raw = np.abs(np.asarray(
                A.T @ np.asarray(y, dtype=np.float64).ravel().astype(_A_dt, copy=False),
                dtype=np.float64)).ravel()
            _a_uw = float(_g_raw.max()) / n_samples
            _g = _g_raw / np.maximum(self._weights, 1e-300)
            _a_max = float(_g.max()) / n_samples
        elif _manual and self.alphas.size > 1:
            # [FIX P40] manual grid diagnostic: one rmatvec (guarded by
            # PHEASY_ALASSO_GRID_DIAG) to quantify how far the UNWEIGHTED scale is
            # from the weighted KKT threshold. Default ON here because this is the
            # one non-auto path most likely to be mis-scaled.
            _diag = os.environ.get("PHEASY_ALASSO_GRID_DIAG", "1").lower() in ("1", "true", "yes")
            if _diag:
                _A_dt = A.dtype if hasattr(A, "dtype") else np.float64
                _g_raw = np.abs(np.asarray(
                    A.T @ np.asarray(y, dtype=np.float64).ravel().astype(_A_dt, copy=False),
                    dtype=np.float64)).ravel()
                _a_max = float((_g_raw / np.maximum(self._weights, 1e-300)).max()) / n_samples

        if _override:
            if _a_max > 0 and np.isfinite(_a_max) and _a_uw > 0 and np.isfinite(_a_uw):
                # [FIX P37] the weighted grid must also reach the UNDER-regularized
                # regime. The 4-decade span below the weighted KKT threshold clips
                # the CV optimum whenever the true model is dense (measured on
                # MnIn2Se4 c3=7.0 n=45: nnz 546 / rel_err 6.5% with the 4-decade
                # span vs nnz 3317 / rel_err 0.55% once the bottom reaches the
                # unweighted-threshold scale; the CV optimum for a dense model sits
                # far below the weighted threshold). Anchor the top at the weighted
                # KKT threshold and the bottom at 10^-max(decades, 6) below the
                # MIN of the two thresholds.
                # Only extend for overdetermined problems (n_rows > n_cols): an
                # underdetermined system genuinely needs regularization, its CV
                # optimum is interior, and the extra low-alpha tail just makes
                # coordinate descent crawl (MnIn2Se4 c3=7.0 n=4: 11s -> 649s for a
                # result within 5% of the 4-decade one).
                if n_samples > A.shape[1]:
                    _hi = _a_max
                    _lo = min(_a_max, _a_uw) * 10.0 ** -max(self.decades, 6.0)
                else:
                    _hi = _a_max
                    _lo = _a_max * 10.0 ** -self.decades
                if _lo > 0 and np.isfinite(_lo) and _hi > _lo:
                    # [FIX P38/P39/P40] density is anchored to the user's per-decade
                    # count (PHEASY_ALPHA_PER_DECADE, default (nmu-1)/4 matching the
                    # historical 4-decade grid), NOT to --alpha_decades: decades
                    # controls the SPAN while density stays fixed, so widening the
                    # grid (to chase a low alpha*) no longer coarsens the step.
                    _span = np.log10(_hi / _lo)
                    _per_dec = float(os.environ.get(
                        "PHEASY_ALPHA_PER_DECADE", str((self.nalpha - 1) / 4.0)))
                    _n = 1 + int(np.ceil(_span * _per_dec))
                    _n = max(_n, self.nalpha)
                    # [FIX P40] hard safety cap (PHEASY_ALPHA_NMAX); warn if it
                    # clamps so --nmu is not silently ignored.
                    _nmax = int(os.environ.get("PHEASY_ALPHA_NMAX", "200"))
                    if _n > _nmax:
                        print("[ALASSO] grid density capped at %d alphas "
                              "(PHEASY_ALPHA_NMAX; computed %d)."
                              % (_nmax, _n), flush=True)
                        _n = _nmax
                    self.alphas = np.logspace(np.log10(_lo), np.log10(_hi), _n)
                    print("[ALASSO] weighted-space alpha grid: [%.3e .. %.3e] "
                          "(%d alphas, %.2f decades, step %.2fx)"
                          % (_lo, _hi, _n, _span,
                             10.0 ** (_span / max(_n - 1, 1))), flush=True)
        elif _manual and self.alphas.size > 1:
            # [FIX P40/P44/P45] manual grid (--no-alpha_auto): used AS-IS, but
            # ALWAYS report the weighted scale so even a well-scaled manual grid
            # leaves an auditable record, and diagnose the two real mismatch modes:
            # (a) the ENTIRE grid is below the weighted KKT threshold (no point
            # regularizes); (b) the grid BOTTOM is far above where the auto grid
            # starts (never reaches the under-regularized regime). Both pin alpha*
            # to the bottom; whether the fit is actually over-regularized depends
            # on the data (the [1e-2,1e2] case can recover the true support).
            _lo_m = float(self.alphas.min())
            _hi_m = float(self.alphas.max())
            if _a_max > 0:
                _a_uw = float(_g_raw.max()) / n_samples
                _auto_lo = min(_a_max, _a_uw) * 10.0 ** -max(self.decades, 6.0)
                if _hi_m < _a_max:
                    _grid_diag = ("the ENTIRE grid lies below the weighted KKT "
                                  "threshold (%.3e): no grid point regularizes at "
                                  "all -- the GRID, not the data, pins alpha*. Drop "
                                  "--no-alpha_auto and use the weighted auto grid."
                                  % _a_max)
                elif _lo_m > _auto_lo * 10.0:
                    _grid_diag = ("the grid BOTTOM (%.3e) is %.1f decades ABOVE "
                                  "where the auto grid starts (%.3e): alpha* is "
                                  "pinned by the grid bottom rather than chosen by "
                                  "CV; the fit MAY be over-regularized (MnIn2Se4 "
                                  "c3=5.2 n45 on this path: nnz 375 vs 1224 on the "
                                  "auto grid). Drop --no-alpha_auto and use the "
                                  "weighted auto grid."
                                  % (_lo_m, np.log10(_lo_m / _auto_lo), _auto_lo))
                else:
                    _grid_diag = None
                print("[ALASSO] manual alpha grid [%.3e .. %.3e] used AS-IS; "
                      "weighted KKT threshold %.3e, auto grid would start at "
                      "%.3e.%s" % (_lo_m, _hi_m, _a_max, _auto_lo,
                                   (" " + _grid_diag) if _grid_diag else ""),
                      flush=True)
            else:
                print("[ALASSO] manual alpha grid [%.3e .. %.3e] used AS-IS "
                      "(scale diagnostic off: PHEASY_ALASSO_GRID_DIAG=0)."
                      % (_lo_m, _hi_m), flush=True)
        elif _mu_shift and self.alphas.size > 1:
            # [FIX P40] mu_shift-auto grid (PHEASY_ALASSO_WEIGHTED_GRID=0 with
            # alpha_auto): run_pheasy already did the rmatvec to derive it, so no
            # matvec here -- a terse note that this is the fallback path.
            print("[ALASSO] mu_shift-auto alpha grid [%.3e .. %.3e] (fallback "
                  "path, PHEASY_ALASSO_WEIGHTED_GRID=0)."
                  % (float(self.alphas.min()), float(self.alphas.max())), flush=True)

        if _lasso_backend(A) == "iterative":
            # [FIX P26] penalized form: pass per-coordinate weights w_j and
            # fit on the ORIGINAL columns (no column-scaling), so the FISTA
            # Lipschitz stays ||A||^2 and convergence is as fast as plain
            # LASSO. Column-scaling by 1/w would inflate the Lipschitz to
            # ||A / w||^2 and make FISTA crawl on the ill-conditioned matrix.
            it = _LassoCVIterative(
                self.alphas, self.cv, self.tol, self.max_iter, self.rand_seed,
                None, fit_intercept=self.fit_intercept,
                group_size=self.group_size, selection=self.selection,
                penalty_weights=self._weights, grid_diag=_grid_diag)
            it.fit(A, y, sample_weight=sample_weight)
            self.model_ = it
            self.coef_ = it.coef_
            self.intercept_ = it.intercept_ if self.fit_intercept else 0.0
            self.alpha_ = it.alpha_
            self.alphas_ = it.alphas_
            self.mse_path_ = it.mse_path_
            self.n_iter_ = it.n_iter_
            self.n_features_in_ = A.shape[1]
            self._alpha_at_min = getattr(it, "_alpha_at_min", False)
            self._alpha_at_min_flat = getattr(it, "_alpha_at_min_flat", False)
            self._alpha_at_min_hitcap = getattr(it, "_alpha_at_min_hitcap", False)
            return self

        A_scaled = _scale_columns(A, self._weights)
        splits = _make_cv_splits(n_samples, self.cv, self.rand_seed, self.group_size)
        model = LassoCV(
            alphas=self.alphas,
            cv=splits,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
            random_state=self.rand_seed,
            selection=self.selection,
            n_jobs=self.n_jobs,
        )
        model.fit(A_scaled, y, sample_weight=sample_weight)
        _reselect_alpha(model, A_scaled, y, sample_weight=sample_weight,
                       grid_diag=_grid_diag)  # [FIX P23/P43/P44]
        self.model_ = model
        self.coef_ = model.coef_ / self._weights
        self.intercept_ = model.intercept_ if self.fit_intercept else 0.0
        self.alpha_ = model.alpha_
        self.alphas_ = np.asarray(model.alphas_)
        self.mse_path_ = np.asarray(model.mse_path_)
        self.n_iter_ = int(model.n_iter_)
        self.n_features_in_ = A.shape[1]
        self._alpha_at_min = getattr(model, "_alpha_at_min", False)
        self._alpha_at_min_flat = getattr(model, "_alpha_at_min_flat", False)
        self._alpha_at_min_hitcap = getattr(model, "_alpha_at_min_hitcap", False)
        return self

    def predict(self, A):
        pred = np.asarray(A @ self.coef_).ravel()
        if self.fit_intercept:
            pred = pred + self.intercept_
        return pred


def _scale_columns(A, w):
    """Column-scaled copy of A: A[:, j] / w[j] (dense or sparse).

    [FIX P26] A LinearOperator (e.g. TwoLevelSM) is wrapped by _scale_operator
    instead of being materialized, so LASSO / ALASSO can run the matvec-only
    FISTA solver and keep the two-level memory optimization (sklearn's
    coordinate descent cannot consume an operator, so _LassoCVModel routes
    it to the iterative backend).
    """
    inv_w = 1.0 / np.asarray(w, dtype=np.float64)
    if sp.issparse(A):
        return A.astype(np.float64).multiply(inv_w[None, :]).tocsr()
    if _is_linear_operator(A):
        # [FIX P26] wrap instead of materializing, so LASSO/ALASSO run the
        # matvec-only FISTA solver and keep the two-level memory optimization.
        return _scale_operator(A, w)
    A64 = _to_dense_f64(A)
    # A[:, j] * (1/w[j]) == A[:, j] / w[j]
    return A64 * inv_w[None, :]


class _RFECVBase:
    """Recursive feature elimination with cross-validated feature count.

    Uses scale-invariant importance |coef_j| * ||A[:, j]|| so that 2nd and 3rd
    order force-constant columns (different physical units) are ranked by their
    actual contribution to the fit.
    """

    def __init__(self, step=0.1, cv=5, min_features=1, n_jobs=None,
                 verbose=False, random_state=None, solver="lstsq", ridge_alpha=0.0,
                 patience=5, lsmr_maxiter=5000, lsmr_atol=1e-8, lsmr_btol=1e-8,
                 block_rows=None, diag_floor=1e-12):
        self.step = float(step)
        self.cv = int(cv)
        self.min_features = int(min_features)
        self.n_jobs = _resolve_n_jobs("RFE", n_jobs if n_jobs is not None else -1)
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
        self.diag_floor = float(diag_floor)
        # [FIX P35] feature-count selection criterion. Default "cv" (CV + 1-SE).
        # RFE-OLS-TSQR overrides this from PHEASY_TSQR_CRITERION=bic|cv so it
        # becomes a genuinely independent method (BIC) instead of a numerically
        # identical twin of RFE.
        self._criterion = "cv"

    def _cv_group_size(self, n_samples):
        gs = int(os.environ.get("PHEASY_CV_GROUP_SIZE", "0") or "0")
        if gs > 1 and n_samples % gs == 0:
            return gs
        return None

    def _bic_n_eff(self, n_samples):
        """[FIX P35] effective independent observations for BIC.

        Force components in one configuration are highly correlated (3*natoms
        rows per config) -- the same reason the CV is grouped.  BIC's n should
        therefore be the CONFIGURATION count, not the raw row count, or the
        k*ln(n) penalty is off by ln(3*natoms) and the fit term is inflated by
        3*natoms. PHEASY_BIC_N_EFF=samples falls back to raw rows.
        """
        gs = self._cv_group_size(n_samples)
        if (os.environ.get("PHEASY_BIC_N_EFF", "groups").lower() != "samples"
                and gs and gs > 1 and n_samples % gs == 0):
            return n_samples // gs, gs
        return n_samples, gs

    def fit(self, A, y, sample_weight=None):
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = A.shape
        self.n_features_in_ = n_features

        col_norms = _col_norms(A)
        _qr = self._solver_name == "qr"

        def solve(col_idx, row_idx=None):
            return _solve_subset(A, y, row_idx, col_idx,
                                 self.ridge_alpha, _qr,
                                 lsmr_atol=self.lsmr_atol,
                                 lsmr_btol=self.lsmr_btol,
                                 lsmr_maxiter=self.lsmr_maxiter,
                                 block_rows=self.block_rows,
                                 diag_floor=self.diag_floor)

        splits = _make_cv_splits(n_samples, self.cv, self.random_state,
                                 self._cv_group_size(n_samples))

        # [FIX P09] constructor value is the default; env var is an override
        patience = int(os.environ.get("PHEASY_RFE_PATIENCE", str(self.patience)))
        criterion = getattr(self, "_criterion", "cv")
        active = np.ones(n_features, dtype=bool)
        history = []      # (n_active, cv_mean, cv_se, support)
        history_bic = []  # [FIX P35] (n_active, bic, support) for TSQR BIC

        if self.verbose:
            print(f"[RFE] START n_features={n_features}, step={self.step:.2f}, "
                  f"cv={self.cv}, min_features={self.min_features}, "
                  f"patience={patience}, "
                  f"solver={self._solver_name}, ridge_alpha={self.ridge_alpha:.2e}",
                  flush=True)

        round_num = 0
        best_cv = float("inf")
        best_bic = float("inf")
        no_improve = 0
        while True:
            idx = np.where(active)[0]
            n_active = len(idx)
            if n_active <= self.min_features:
                break

            coef_active = solve(idx)
            if criterion in ("bic", "aic"):
                # [FIX P35] IC mode: no CV needed for stopping (skips the K-fold
                # sub-solves -> ~5x faster); CV is computed once at the end for
                # the report only. Loop driven by the criterion's own patience.
                _pred = _predict_subset(A, idx, None, coef_active)
                _rss = max(float(np.sum((_pred - y) ** 2)),
                           np.finfo(float).tiny * n_samples)
                _n_eff, _gs = self._bic_n_eff(n_samples)
                # dimension-consistent: fit term and penalty share n_eff
                # [FIX P36] k+1 counts sigma^2 as a parameter (constant offset;
                # argmin over k is unchanged but the reported absolute value is
                # the textbook AIC/BIC).
                _loglik = _n_eff * np.log(_rss / _n_eff)
                _pen = ((n_active + 1) * np.log(_n_eff) if criterion == "bic"
                        else 2 * (n_active + 1))
                _ic = _loglik + _pen
                history_bic.append((n_active, _ic, active.copy()))
                cv_mean, cv_se = 0.0, 0.0
                if self.verbose:
                    print(f"[RFE] Round {round_num:3d}: n_active={n_active:5d}  "
                          f"{criterion.upper()}={_ic:.2e} (n_eff={_n_eff})  "
                          f"nonzero={int(np.count_nonzero(coef_active))}", flush=True)
                if _ic < best_bic:
                    best_bic = _ic
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                cv_mean, cv_se, _ = _cv_rmse(A, y, idx, solve, splits,
                                                n_jobs=self.n_jobs)
                history.append((n_active, cv_mean, cv_se, active.copy()))
                if self.verbose:
                    print(f"[RFE] Round {round_num:3d}: n_active={n_active:5d}  "
                          f"CV_RMSE={cv_mean:.6e} (+-{cv_se:.2e})  "
                          f"nonzero={int(np.count_nonzero(coef_active))}", flush=True)
                if cv_mean < best_cv:
                    best_cv = cv_mean
                    no_improve = 0
                else:
                    no_improve += 1
            if no_improve >= patience:
                if self.verbose:
                    print(f"[RFE] {criterion.upper()} 连续 {patience} 轮无改善, 提前停止.",
                          flush=True)
                break

            if n_active <= self.min_features:
                break

            imp = np.abs(coef_active) * col_norms[idx]
            n_remove = max(1, int(round(n_active * self.step)))
            n_remove = min(n_remove, n_active - self.min_features)
            remove_local = np.argsort(imp)[:n_remove]
            active[idx[remove_local]] = False
            round_num += 1

        # --- selection (each criterion prints its own summary, once) ---
        if criterion in ("bic", "aic") and history_bic:
            best_round = int(np.argmin([h[1] for h in history_bic]))
            n_best = history_bic[best_round][0]
            best_support = history_bic[best_round][2]
            # CV once for the selected round (report only)
            _sel = np.where(best_support)[0]
            best_mean, best_se, _ = _cv_rmse(A, y, _sel, solve, splits,
                                                n_jobs=self.n_jobs)
            if self.verbose:
                print("[RFE] %s min: n_active=%d, %s=%.2e" % (criterion.upper(), n_best, criterion.upper(), history_bic[best_round][1]), flush=True)
                print("[RFE] selected: n_active=%d, CV_RMSE=%.6e (+-%.2e)" % (n_best, best_mean, best_se), flush=True)
        elif history:
            n_best, best_mean, best_se, best_support = _select_1se(history)
            if self.verbose:
                argmin = history[int(np.argmin([h[1] for h in history]))]
                print(f"[RFE] argmin: n_active={argmin[0]}, CV={argmin[1]:.6e}", flush=True)
                print(f"[RFE] selected: n_active={n_best}, CV={best_mean:.6e} (+-{best_se:.2e})", flush=True)
        else:
            # min_features >= n_features: nothing was eliminated, keep all.
            n_best, best_mean, best_se = n_features, 0.0, 0.0
            best_support = np.ones(n_features, dtype=bool)
        best_idx = np.where(best_support)[0]

        coef_final = solve(best_idx)
        coef_full = np.zeros(n_features, dtype=np.float64)
        coef_full[best_idx] = coef_final

        self.coef_ = coef_full
        self.intercept_ = 0.0
        self.support_ = best_support
        self.n_iter_ = round_num
        self.best_rmse_cv_ = best_mean
        self.ridge_alpha = self.ridge_alpha
        self.alphas_ = np.array([self.ridge_alpha])
        self.mse_path_ = np.array([[best_mean ** 2]])
        return self

    def predict(self, A):
        return np.asarray(A @ self.coef_).ravel()


class PheasyRFECV(_RFECVBase):
    """RFE with an OLS (optionally ridge-regularized) base estimator."""

    def __init__(self, step=0.05, cv=5, ridge_alpha=0.0, lsmr_maxiter=3000,   # [FIX P21]
                 lsmr_atol=1e-8, lsmr_btol=1e-8, n_jobs=None, min_features=1,
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
                 random_state=None, n_jobs=None):
        super().__init__(step=step, cv=cv, min_features=min_features, n_jobs=n_jobs,
                         verbose=verbose, random_state=random_state,
                         solver="qr", ridge_alpha=0.0, patience=patience,
                         block_rows=block_rows, diag_floor=diag_floor)
        # [FIX P35] BIC/AIC are genuinely independent stopping rules (the
        # CV+1-SE path makes RFE and RFE-OLS-TSQR numerically identical).
        # NOTE: with the grouped-CV n_eff (configuration count, often tens),
        # BIC's k*ln(n_eff) penalty is heavy and picks over-sparse models
        # (e.g. MnIn2Se4 n_eff=45 -> 21 features, CV_RMSE 78x worse than CV's
        # 1238). CV remains the recommended default; BIC/AIC are sensible only
        # when n_eff is large (hundreds+).
        _crit = os.environ.get("PHEASY_TSQR_CRITERION", "cv").lower()
        self._criterion = _crit if _crit in ("bic", "aic") else "cv"


# backward-compatible aliases
CelerLassoCV = _LassoCVModel
CelerALassoCV = _AdaptiveLassoCV
_LsmrOLSResult = _OLSModel


class Optimizer(object):
    """Interatomic force constant optimizer.

    Supported methods: ols, lasso, alasso, rfe, rfe-ols-tsqr (rfe_tsqr) and
    the legacy ridge.
    """

    def __init__(
        self,
        method="ols",
        nalpha=100,
        alpha_min=-6,
        alpha_max=-2,
        alpha=None,
        cv=5,
        tol=1e-4,
        max_iter=20000,
        rand_seed=None,
        standardize=False,
        fit_intercept=False,
        alpha_auto=True,
        decades=4.0,
    ):
        self._method = method
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._nalpha = nalpha
        self._cv = cv
        self._tol = tol
        self._max_iter = max_iter
        self._rand_seed = rand_seed
        self._standardize = standardize
        self._fit_intercept = fit_intercept
        self._alpha_auto = bool(alpha_auto)
        self._decades = float(decades)

        if alpha is not None:
            self._alpha = np.asarray(alpha, dtype=np.float64)
        else:
            # alpha_min/alpha_max are POWERS OF 10 (exponents), matching the
            # pheasy CLI (--mu_min/--alpha_min, default -6/-2):
            #   alphas = 10^alpha_min ... 10^alpha_max
            self._alpha = np.logspace(alpha_min, alpha_max, nalpha)

        self._group_size = None
        self._results = {}
        self._metrics = {}

    def _ols_lsmr(self, X, y, atol=None, btol=None, maxiter=None):
        """OLS via LSMR (iterative; sparse and LinearOperator safe).

        [FIX] tolerance must follow operator dtype: float32 (eps ~1.2e-7)
        cannot converge to atol=1e-8 (unreachable -> LSMR always hits
        maxiter). Defaults: 1e-8 for float64, 1e-6 for float32, from the
        DATA-carrying operator X. show=PHEASY_LSQR_SHOW prints the per-iter
        LS column (the real stopping criterion). PHEASY_LSQR_MAXITER caps
        iterations like the LSQR path (probe runs set it to 200).
        """
        _eps = np.finfo(np.dtype(getattr(X, "dtype", np.float64))).eps
        _dflt = "1e-8" if _eps < 1e-10 else "1e-6"
        atol = float(os.environ.get("PHEASY_OLS_ATOL",
                                    os.environ.get("PHEASY_LSQR_ATOL", _dflt)))
        btol = float(os.environ.get("PHEASY_OLS_BTOL",
                                    os.environ.get("PHEASY_LSQR_BTOL", _dflt)))
        maxiter = int(os.environ.get("PHEASY_OLS_MAXITER",
                                     os.environ.get("PHEASY_LSQR_MAXITER", "5000")))
        ridge = float(os.environ.get("PHEASY_OLS_RIDGE", "0"))
        n_samples = X.shape[0]
        damp = float(np.sqrt(ridge * n_samples)) if ridge > 0 else 0.0
        y_in = np.asarray(y, dtype=np.float64).ravel()
        _show = (os.environ.get("PHEASY_LSQR_SHOW", "0") == "1")
        result = _lsmr(X, y_in, damp=damp, atol=atol, btol=btol,
                       maxiter=maxiter, show=_show)
        coef = np.asarray(result[0], dtype=np.float64)
        print("[LSMR] istop={} itn={} normr={:.3e} (atol={:.0e}, lim={})".format(
            result[1], result[2], result[3], atol, maxiter), flush=True)
        self._ols_lsmr_info = {"istop": result[1], "itn": result[2],
                               "normr": result[3], "normar": result[4]}
        return coef

    def fit(self, A, F, weights=None):
        method = self._method.upper().replace("_", "-")
        if method in ("RFE-OLS-TSQR", "RFE-TSQR"):
            method = "RFE-OLS-TSQR"
        elif method == "RFECV":
            method = "RFE"

        F = np.asarray(F)
        if F.ndim == 2:
            F = F.ravel()
        F64 = np.asarray(F, dtype=np.float64).ravel()

        self._group_size = self._detect_group_size(A.shape[0])

        # Column standardization (unit L2 norm) for the scale-sensitive
        # penalized methods; coefficients are un-scaled after fitting.
        col_scale = None
        A_fit = A
        if self._standardize and method in ("LASSO", "ALASSO", "RIDGE"):
            col_scale = _col_norms(A)
            col_scale = np.where(col_scale < 1e-30, 1.0, col_scale)
            A_fit = _scale_columns(A, col_scale)

        if method == "OLS":
            coef, n_iter = self._fit_ols(A, F64)
            self._model = _OLSModel(coef, n_iter=n_iter)
        elif method == "LASSO":
            self._model = _LassoCVModel(
                self._alpha, self._cv, self._tol, self._max_iter, self._rand_seed,
                _lasso_n_jobs(A),
                fit_intercept=self._fit_intercept, group_size=self._group_size)
            self._model.fit(A_fit, F64, sample_weight=weights)
            coef = self._model.coef_
        elif method == "ALASSO":
            self._model = _AdaptiveLassoCV(
                self._alpha, self._cv, self._tol, self._max_iter, self._rand_seed,
                _lasso_n_jobs(A),
                fit_intercept=self._fit_intercept, group_size=self._group_size,
                gamma=float(os.environ.get("PHEASY_ALASSO_GAMMA", "1.0")),
                init_alpha=float(os.environ.get("PHEASY_ALASSO_RIDGE_ALPHA", "1e-3")),
                eps=float(os.environ.get("PHEASY_ALASSO_EPS", "1e-8")),
                nalpha=self._nalpha,
                decades=self._decades,
                alpha_auto=self._alpha_auto)
            self._model.fit(A_fit, F64, sample_weight=weights)
            coef = self._model.coef_
        elif method == "RFE":
            self._model = PheasyRFECV(
                step=float(os.environ.get("PHEASY_RFE_STEP", "0.05")),  # [FIX P21]
                cv=self._cv,
                ridge_alpha=float(os.environ.get("PHEASY_RFE_RIDGE_ALPHA", "0")),
                n_jobs=None,  # resolved via PHEASY_RFE_N_JOBS / PHEASY_N_JOBS
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
                # [FIX P21] PHEASY_TSQR_STEP overrides, else PHEASY_RFE_STEP
                step=float(os.environ.get(
                    "PHEASY_TSQR_STEP",
                    os.environ.get("PHEASY_RFE_STEP", "0.05"))),
                cv=self._cv,
                min_features=int(os.environ.get("PHEASY_TSQR_MIN_FEATURES", "100")),
                block_rows=int(os.environ.get("PHEASY_TSQR_BLOCK_ROWS", "40000")),
                diag_floor=float(os.environ.get("PHEASY_TSQR_DIAG_FLOOR", "1e-12")),
                patience=int(os.environ.get("PHEASY_RFE_PATIENCE", "5")),
                verbose=True, random_state=self._rand_seed)
            self._model.fit(A, F64, sample_weight=weights)
            coef = self._model.coef_
        elif method == "RIDGE":
            alphas = np.sort(np.asarray(self._alpha, dtype=np.float64))[::-1]
            if _is_linear_operator(A_fit):
                # [FIX P26/P45] ridge CV over the alpha grid on the two-level
                # operator via _ridge_solve (LSMR on the augmented system).
                # (1) row slices are hoisted out of the alpha loop; (2) the
                # alpha path walks large->small with LSMR warm-start; (3) folds
                # are parallelized with threads (alpha stays serial).
                splits = _make_cv_splits(A.shape[0], self._cv, self._rand_seed,
                                         self._group_size)
                n_jobs = _resolve_n_jobs("RIDGE")
                A_tr_list = [_row_slice(A_fit, tr) for tr, _ in splits]
                y_tr_list = [F64[tr] for tr, _ in splits]
                warm = [None] * len(splits)

                def _fold_rmse(a, k):
                    c = _ridge_solve(A_tr_list[k], y_tr_list[k], a, x0=warm[k])
                    warm[k] = c
                    va = splits[k][1]
                    return float(np.mean(
                        (_predict_rows(A_fit, c, va) - F64[va]) ** 2))

                mse_path = np.zeros((len(alphas), len(splits)))
                parallel = n_jobs > 1 and len(splits) > 1
                n_workers = min(n_jobs, len(splits))
                if parallel:
                    from joblib import Parallel, delayed
                # [FIX P45b] enter _blas_limit ONCE for the whole alpha sweep
                # instead of per alpha (threadpool_limits walks the loaded BLAS
                # libraries on every entry/exit).
                ctx = _blas_limit(n_workers) if parallel else contextlib.nullcontext()
                with ctx:
                    for j, a in enumerate(alphas):
                        if parallel:
                            errs = Parallel(n_jobs=n_workers, prefer="threads")(
                                delayed(_fold_rmse)(a, k)
                                for k in range(len(splits)))
                        else:
                            errs = [_fold_rmse(a, k) for k in range(len(splits))]
                        mse_path[j] = errs
                best_alpha = float(alphas[int(np.argmin(mse_path.mean(axis=1)))])
                coef = _ridge_solve(A_fit, F64, best_alpha)
                self._model = _OLSModel(coef, alpha=best_alpha)
                self._results["alpha"] = best_alpha
                self._results["mse_path"] = mse_path
            else:
                # [FIX P46] dense ridge CV: one economic SVD per fold evaluates
                # the WHOLE alpha grid in closed form
                # (w = V diag(s/(s^2+a)) U^T y), and the splits are the same
                # grouped CV every other method uses. The old RidgeCV(alphas=,
                # no cv=) ran leave-one-out GCV: it ignored --cv and
                # PHEASY_CV_GROUP_SIZE, and row-wise LOO leaks the other 3N-1
                # rows of the same configuration into training (biasing alpha*
                # toward 0).
                if sp.issparse(A_fit):
                    A_dense = _to_dense_f64(A_fit)
                else:
                    A_dense = np.ascontiguousarray(A_fit, dtype=np.float64)
                splits = _make_cv_splits(A_dense.shape[0], self._cv,
                                         self._rand_seed, self._group_size)
                if self._fit_intercept:
                    # grouped CV still fixes the leak; RidgeCV handles the
                    # intercept the closed form below does not (pheasy CLI
                    # leaves fit_intercept=False).
                    self._model = RidgeCV(alphas=self._alpha, fit_intercept=True,
                                          cv=splits)
                    self._model.fit(A_dense, F64, sample_weight=weights)
                    coef = self._model.coef_
                else:
                    # [FIX P47] weighted ridge == unweighted ridge on
                    # sqrt(weights)-scaled rows; exact, and the CV folds then
                    # need no per-fold reweighting.
                    y_dense = F64
                    if weights is not None:
                        sw = np.sqrt(np.asarray(weights, dtype=np.float64).ravel())
                        A_dense = A_dense * sw[:, None]
                        y_dense = y_dense * sw
                    # [FIX P47] mse_path keeps the SAME (n_alphas, n_folds)
                    # shape as the operator branch (per-fold values), so the two
                    # paths agree and downstream can read per-fold RMSE.
                    mse_path = np.zeros((len(alphas), len(splits)), dtype=np.float64)
                    for k, (tr, va) in enumerate(splits):
                        U, s, Vt = np.linalg.svd(A_dense[tr], full_matrices=False)
                        Uty = U.T @ y_dense[tr]
                        AvV = A_dense[va] @ Vt.T
                        for j, a in enumerate(alphas):
                            pred = AvV @ ((s / (s ** 2 + a)) * Uty)
                            mse_path[j, k] = float(np.mean((pred - y_dense[va]) ** 2))
                    best_alpha = float(alphas[int(np.argmin(mse_path.mean(axis=1)))])
                    # final refit at the selected alpha (closed form, full data)
                    U, s, Vt = np.linalg.svd(A_dense, full_matrices=False)
                    coef = Vt.T @ ((s / (s ** 2 + best_alpha)) * (U.T @ y_dense))
                    self._model = _OLSModel(coef, alpha=best_alpha)
                    self._results["alpha"] = best_alpha
                    self._results["mse_path"] = mse_path
        else:
            raise ValueError(
                "Unknown linear model for fitting force constants: {} ".format(self._method)
                + "(expected OLS, LASSO, ALASSO, RFE, RFE-OLS-TSQR, RIDGE)")

        coef = np.asarray(coef, dtype=np.float64)

        # Relaxed LASSO / debias: L1 selects the support, then an unbiased OLS
        # refit on that support removes the L1 shrinkage bias.  This is the
        # standard practical way to obtain physical force constants from a
        # LASSO fit (Meinshausen 2007; used by ALAMODE/phono3py).  Default on;
        # disable with PHEASY_LASSO_DEBIAS=0.
        if method in ("LASSO", "ALASSO") and self._debias_enabled():
            # [FIX P34] propagate the model's Gram (built on A_fit) so _debias
            # can solve G[sup,sup] x = b[sup] instead of re-solving the OLS.
            self._gram = getattr(self._model, "_gram", None)
            coef = self._debias(A_fit, F64, coef)

        # un-scale standardized coefficients back to the original column scale
        if col_scale is not None:
            coef = coef / col_scale

        # [FIX P11] the hard threshold used to hit every method, including OLS
        # and RFE, where zeroing tiny-but-real coefficients is not wanted.
        # Default: only the L1 methods (whose exact zeros are the point).
        _default_tol = "1e-12" if method in ("LASSO", "ALASSO") else "0"
        _zero_tol = float(os.environ.get("PHEASY_COEF_ZERO_TOL", _default_tol))
        if _zero_tol > 0:
            coef = np.where(np.abs(coef) < _zero_tol, 0.0, coef)
        self._results["coef"] = coef
        self._model.coef_ = self._results["coef"]

        if method in ("LASSO", "ALASSO"):
            self._results["alpha"] = float(self._model.alpha_)
            self._results["n_iter"] = int(self._model.n_iter_)
            alpha_idx = int(np.argmin(np.abs(self._model.alphas_ - self._model.alpha_)))
            self._metrics["mse_path"] = np.asarray(self._model.mse_path_[alpha_idx])
            self._metrics["mse_path_mean"] = float(np.mean(self._metrics["mse_path"]))
            self._metrics["rmse_path"] = np.sqrt(self._metrics["mse_path"])
            self._metrics["rmse_path_mean"] = float(np.mean(self._metrics["rmse_path"]))
            self._metrics["n_features"] = self._model.n_features_in_
            self._metrics["n_featrues"] = self._metrics["n_features"]  # [FIX P12] deprecated alias
        elif method in ("RFE", "RFE-OLS-TSQR"):
            self._results["alpha"] = float(getattr(self._model, "ridge_alpha", 0.0))
            self._results["n_iter"] = int(getattr(self._model, "n_iter_", 0))
            bcv = float(getattr(self._model, "best_rmse_cv_", 0.0))
            self._metrics["mse_path"] = np.array([bcv ** 2])
            self._metrics["mse_path_mean"] = bcv ** 2
            self._metrics["rmse_path"] = np.array([bcv])
            self._metrics["rmse_path_mean"] = bcv
            self._metrics["n_features"] = self._model.n_features_in_
            self._metrics["n_featrues"] = self._metrics["n_features"]  # [FIX P12] deprecated alias
        elif method == "RIDGE":
            # [FIX P26] the operator path stores a plain _OLSModel (no alpha_),
            # so fall back to the alpha already recorded during the ridge CV.
            self._results["alpha"] = float(getattr(
                self._model, "alpha_", self._results.get("alpha", 0.0)))
        elif method == "OLS":
            self._results["n_iter"] = getattr(self._model, "n_iter_", None)

        F_pred = np.asarray(self.predict(A)).ravel()
        eps = np.finfo(F64.dtype).eps
        F_err = np.abs(F_pred - F64)
        F_re = F_err / np.maximum(np.abs(F64), eps)

        self._metrics["re"] = float(np.sqrt(np.dot(F_err, F_err) / np.dot(F64, F64)))
        self._metrics["r2_score"] = float(r2_score(F64, F_pred, sample_weight=weights))
        self._metrics["mae"] = float(mean_absolute_error(F64, F_pred, sample_weight=weights))
        self._metrics["mape"] = float(mean_absolute_percentage_error(F64, F_pred, sample_weight=weights))
        self._metrics["mse"] = float(mean_squared_error(F64, F_pred, sample_weight=weights))
        self._metrics["rmse"] = float(np.sqrt(self._metrics["mse"]))
        self._metrics["mspe"] = float(np.average(np.square(F_re), weights=weights, axis=0))
        self._metrics["rmspe"] = float(np.sqrt(self._metrics["mspe"]))
        return self

    def _fit_ols(self, A, F):
        if _is_linear_operator(A):
            # LSMR only needs matvec/rmatvec; the two-level operator stays sparse.
            coef = self._ols_lsmr(A, F)
            n_iter = self._ols_lsmr_info.get("itn")
            return coef, n_iter
        # dense, or a sparse container: _solve_lstsq densifies small/sparse-but-
        # dense matrices (fast SVD) and only uses iterative LSQR for genuinely
        # huge sparse matrices (FIX: previously everything sparse went to LSMR).
        coef = _solve_lstsq(A, F, driver="gelsd")
        return coef, None

    @staticmethod
    def _debias_enabled():
        return os.environ.get("PHEASY_LASSO_DEBIAS", "1").lower() in ("1", "true", "yes")

    def _debias(self, A, y, coef):
        """OLS refit on the nonzero support (relaxed LASSO)."""
        sup = np.flatnonzero(np.abs(coef) > 0)
        if not (0 < sup.size < coef.size):
            return coef
        gram = getattr(self, "_gram", None)
        if gram is not None:
            # [FIX P34] OLS on the support via the Gram: G[sup,sup] x = b[sup]
            # is a |sup| x |sup| dense solve, far cheaper than re-solving the
            # full least-squares problem against the operator.
            G, b = gram
            Gss = G[np.ix_(sup, sup)]
            bs = b[sup]
            try:
                coef_sub = np.linalg.solve(Gss, bs)
            except np.linalg.LinAlgError:
                coef_sub = _solve_lstsq(Gss, bs)
            new = np.zeros_like(coef)
            new[sup] = coef_sub
            r_new = float(np.linalg.norm(np.asarray(A @ new).ravel() - y))
            r_old = float(np.linalg.norm(np.asarray(A @ coef).ravel() - y))
            if r_new <= r_old:
                return new
            return coef
        if _is_linear_operator(A):
            # [FIX P26] column-slice via a masked operator + LSMR, so the
            # relaxed-LASSO debias is no longer skipped on the two-level
            # operator (the L1 shrinkage bias is removed there too).
            op = _make_masked_op(A, None, sup)
            coef_sub = _solve_sparse_lsqr(op, y)
            new = np.zeros_like(coef)
            new[sup] = coef_sub
            r_new = float(np.linalg.norm(np.asarray(A @ new).ravel() - y))
            r_old = float(np.linalg.norm(np.asarray(A @ coef).ravel() - y))
            if r_new <= r_old:
                return new
            return coef
        A_sub = A[:, sup]
        coef_sub = _solve_lstsq(A_sub, y)
        new = np.zeros_like(coef)
        new[sup] = coef_sub
        # keep only if the residual does not increase (guards against a
        # support that is inconsistent with the data scale)
        r_new = float(np.linalg.norm(np.asarray(A @ new).ravel() - y))
        r_old = float(np.linalg.norm(np.asarray(A @ coef).ravel() - y))
        if r_new <= r_old:
            return new
        return coef

    @staticmethod
    def _detect_group_size(n_samples):
        gs = int(os.environ.get("PHEASY_CV_GROUP_SIZE", "0") or "0")
        if gs > 1 and n_samples % gs == 0:
            return gs
        return None

    def predict(self, A):
        return self._model.predict(A)

    @property
    def results(self):
        return self._results

    @property
    def metrics(self):
        return self._metrics

    @property
    def model(self):
        return self._model

    def get_paras(self):
        return self._model.coef_

    def __repr__(self):
        return "<Optimizer method={}>".format(self._method)
