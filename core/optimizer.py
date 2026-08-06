"""Classes and functions for force constant regression."""
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2023 Changpeng Lin
# All rights reserved.
import os
__all__ = ["Optimizer"]

import numpy as np
import scipy.sparse as _sp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression,
    LassoCV,
    RidgeCV,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

# ===== celer fast LASSO support =====
try:
    from celer import Lasso as _CelerLasso
    from celer.homotopy import celer_path as _celer_path
    _HAS_CELER = True
except ImportError:
    _HAS_CELER = False



class CelerLassoCV:
    """LassoCV-compatible wrapper using celer.Lasso for fast sparse fitting.

    Attributes after fit (matching sklearn LassoCV):
      coef_, alpha_, alphas_, mse_path_, n_iter_, n_features_in_
    """

    def __init__(self, alphas, max_iter=20000, tol=1e-4, cv=5,
                 fit_intercept=False, random_state=None, n_jobs=-1,
                 verbose=False):
        self.alphas = np.asarray(alphas, dtype=np.float64)
        # celer expects descending alphas for warm-start path; sort desc
        self._alpha_order = np.argsort(self.alphas)[::-1]
        self.alphas_sorted_desc = self.alphas[self._alpha_order]
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.cv = int(cv)
        self.fit_intercept = bool(fit_intercept)
        self.random_state = random_state
        self.n_jobs = int(n_jobs) if n_jobs is not None else -1
        self.verbose = verbose

    def _fit_one_fold(self, X_train, y_train, X_val, y_val, sw_train, sw_val):
        """Fit path on one fold; return MSE for each alpha (sorted desc).

        NOTE: DEAD CODE. Replaced by _fit_one_fold_inplace. Kept for
        reference; do not call (X, val_idx not in scope -> NameError).
        """
        n_alphas = len(self.alphas_sorted_desc)
        mse = np.zeros(n_alphas)
        prev_coef = None
        for i, alpha in enumerate(self.alphas_sorted_desc):
            est = _CelerLasso(
                alpha=alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                fit_intercept=self.fit_intercept,
                warm_start=(prev_coef is not None),
                verbose=0,
            )
            if prev_coef is not None:
                # manually inject warm start
                est.coef_ = prev_coef.copy()
            if sw_train is not None:
                # celer doesn't support sample_weight directly; rescale
                w = np.sqrt(sw_train)
                if _sp.issparse(X_train):
                    Xw = X_train.multiply(w[:, None]).tocsc()
                else:
                    Xw = X_train * w[:, None]
                yw = y_train * w
                est.fit(Xw, yw)
            else:
                est.fit(X_train, y_train)
            prev_coef = est.coef_
            # NOTE: undefined names X, val_idx — dead method, never called.
            y_pred_full = X @ est.coef_
            if self.fit_intercept:
                y_pred_full = y_pred_full + est.intercept_
            err = y_pred_full[val_idx] - y_val
            if sw_val is not None:
                mse[i] = np.average(err ** 2, weights=sw_val)
            else:
                mse[i] = float(np.mean(err ** 2))
        return mse
    def _fit_one_fold_inplace(self, X, y, train_idx, val_idx, sample_weight):
        """Fit on X[train_idx] using celer_path low-level API.

        celer_path accepts Fortran-contiguous X without copying (per docs).
        We materialize ONE F-order copy of X[train_idx] (~133 GB), pass to
        celer_path which solves the FULL alpha path with warm-start in one
        call (much faster than looping over alphas).
        """
        # FIX OOM: avoid double copy. X[train_idx] is already a fresh
        # C-order array from fancy indexing (~209 GB for 548k x 102k float32).
        # Calling np.asfortranarray on it would allocate ANOTHER 209 GB.
        # celer_path accepts C-order; it will F-convert internally only if
        # needed (single allocation, slight overhead). Net saving: ~209 GB.
        if not _sp.issparse(X):
            X_tr = X[train_idx]                         # 1 copy (fancy index)
            if X_tr.dtype != np.float32:
                X_tr = X_tr.astype(np.float32, copy=False)
        else:
            X_tr = X[train_idx]                         # sparse stays sparse
        y_tr = np.ascontiguousarray(y[train_idx], dtype=np.float32)
        y_va = y[val_idx]

        # celer_path handles warm-start internally and skips sklearn's copy.
        # alphas must be sorted descending for warm-start.
        _, coefs, _ = _celer_path(
            X_tr, y_tr,
            pb='lasso',
            alphas=self.alphas_sorted_desc.astype(np.float32),
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=0,
        )
        # coefs shape: (n_features, n_alphas)
        del X_tr, y_tr

        # Score on val: full X @ coef (no X_val copy)
        n_alphas = coefs.shape[1]
        mse = np.zeros(n_alphas)
        for i in range(n_alphas):
            y_pred_full = X @ coefs[:, i]
            err = y_pred_full[val_idx] - y_va
            mse[i] = float(np.mean(err ** 2))
        return mse


    def fit(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # K-fold split
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        splits = list(kf.split(np.arange(n_samples)))

        def _fold_job(train_idx, val_idx):
            # FIX: avoid X_val copy (133 GB). Pass full X + indices,
            # _fit_one_fold uses train_idx for fitting and val_idx for scoring.
            return self._fit_one_fold_inplace(X, y, train_idx, val_idx, sample_weight)

        # Parallel over folds (each fold internally is multi-threaded by celer)
        n_par = min(self.cv, abs(self.n_jobs) if self.n_jobs != -1 else self.cv)
        if self.verbose:
            print(f'[CelerLassoCV] {self.cv}-fold CV, {len(self.alphas)} alphas, '
                  f'fold-parallel={n_par}', flush=True)
        mse_per_fold = Parallel(n_jobs=n_par, prefer="threads", verbose=10 if self.verbose else 0)(
            delayed(_fold_job)(tr, va) for tr, va in splits
        )
        mse_path_sorted = np.array(mse_per_fold).T  # (n_alphas, n_folds)
        mse_mean = mse_path_sorted.mean(axis=1)
        best_sorted_idx = int(np.argmin(mse_mean))
        best_alpha = float(self.alphas_sorted_desc[best_sorted_idx])

        # Final fit on full data with best alpha
        if self.verbose:
            print(f'[CelerLassoCV] best alpha={best_alpha:.4e}, '
                  f'CV MSE={mse_mean[best_sorted_idx]:.4e}; refitting on full data',
                  flush=True)
        # FIX: detect ALASSO's penalty weights (self._weights). When set,
        # use celer_path's native `weights` parameter instead of multiplying
        # X by sqrt(sample_weight) (saves 280 GB memory copy AND fixes bug
        # where ALASSO final fit reverted to plain LASSO).
        alasso_weights = getattr(self, '_weights', None)

        if alasso_weights is not None:
            # ALASSO: pass penalty weights directly to celer_path (correct math)
            if _sp.issparse(X):
                X_in = X.tocsc().astype(np.float32) if X.dtype != np.float32 else X.tocsc()
                _need_del = X_in is not X
            else:
                # Same relaxed F-order rule as below: only convert if dtype off
                if X.dtype == np.float32:
                    X_in = X
                    _need_del = False
                else:
                    X_in = X.astype(np.float32, copy=False)
                    _need_del = (X_in is not X)
            y_in = np.ascontiguousarray(y, dtype=np.float32)
            print(f'[CelerALassoCV] Final fit with penalty weights '
                  f'(no X*w copy)', flush=True)
            _, _coefs_final, _ = _celer_path(
                X_in, y_in, pb='lasso',
                alphas=np.array([best_alpha], dtype=np.float32),
                weights=alasso_weights.astype(np.float32),
                max_iter=self.max_iter, tol=self.tol, verbose=0,
            )
            if _need_del:
                del X_in
        elif sample_weight is not None:
            # Plain sample weighting (legacy path, rarely used)
            w = np.sqrt(sample_weight)
            if _sp.issparse(X):
                Xw = X.multiply(w[:, None]).tocsc().astype(np.float32)
            else:
                # Single copy: X*w creates fresh array, astype reuses it
                Xw = (X * w[:, None]).astype(np.float32, copy=False)
            yw = (y * w).astype(np.float32)
            _, _coefs_final, _ = _celer_path(
                Xw, yw, pb='lasso',
                alphas=np.array([best_alpha], dtype=np.float32),
                max_iter=self.max_iter, tol=self.tol, verbose=0,
            )
            del Xw, yw
        else:
            # Plain LASSO
            if _sp.issparse(X):
                X_in = X.tocsc().astype(np.float32) if X.dtype != np.float32 else X.tocsc()
                _need_del = X_in is not X
            else:
                # Relaxed F-order: only convert dtype if needed
                if X.dtype == np.float32:
                    X_in = X
                    _need_del = False
                else:
                    X_in = X.astype(np.float32, copy=False)
                    _need_del = (X_in is not X)
            y_in = np.ascontiguousarray(y, dtype=np.float32)
            _, _coefs_final, _ = _celer_path(
                X_in, y_in, pb='lasso',
                alphas=np.array([best_alpha], dtype=np.float32),
                max_iter=self.max_iter, tol=self.tol, verbose=0,
            )
            if _need_del:
                del X_in

        # Store sklearn-compatible attributes
        self.coef_ = _coefs_final[:, 0].astype(np.float32)
        self.intercept_ = 0.0
        self.alpha_ = best_alpha
        self.alphas_ = self.alphas_sorted_desc
        self.mse_path_ = mse_path_sorted
        self.n_iter_ = self.max_iter
        return self

    def predict(self, X):
        pred = X @ self.coef_
        if self.fit_intercept:
            pred = pred + self.intercept_
        return pred
# ===== end celer wrapper =====
# ===== begin ALASSO wrapper =====
class CelerALassoCV(CelerLassoCV):
    """Adaptive LASSO with CV (subclass of CelerLassoCV).

    Two-stage:
        Stage 1: ridge regression via lsmr (memory-friendly conjugate gradient)
                 to get initial estimate β̂^(0).
        Stage 2: weighted LASSO with w_j = 1 / (|β̂^(0)_j| + eps)^gamma.

    Reuses parent class's CV path (_fit_one_fold_inplace and final fit) by
    overriding the alpha grid and passing weights to celer_path.

    Env vars:
        PHEASY_ALASSO_GAMMA: weight exponent gamma (default 1.0)
        PHEASY_ALASSO_INIT_TYPE: 'ridge' or 'lasso_warm' (default 'ridge')
        PHEASY_ALASSO_RIDGE_ALPHA: ridge regularization (default 1e-3)
        PHEASY_ALASSO_EPS: weight floor to avoid division by zero (default 1e-8)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import os as _os_a
        self.gamma = float(_os_a.environ.get('PHEASY_ALASSO_GAMMA', '1.0'))
        self.init_type = _os_a.environ.get('PHEASY_ALASSO_INIT_TYPE', 'ridge')
        self.ridge_alpha = float(_os_a.environ.get('PHEASY_ALASSO_RIDGE_ALPHA', '1e-3'))
        self.weight_eps = float(_os_a.environ.get('PHEASY_ALASSO_EPS', '1e-8'))
        self._weights = None  # filled in fit()

    def _initial_estimate(self, X, y):
        """Ridge regression via lsmr — works for sparse OR dense X without
        materializing X^T X."""
        from scipy.sparse.linalg import lsmr
        from scipy.sparse import issparse, vstack as sp_vstack, eye as sp_eye, csr_matrix
        n_samples, n_features = X.shape
        sqrt_alpha = np.sqrt(self.ridge_alpha * n_samples)
        # Augmented system: minimize ||[X; sqrt(α)·I] β - [y; 0]||²
        if issparse(X):
            X_aug_top = X
            X_aug_bot = sqrt_alpha * sp_eye(n_features, format='csr')
            X_aug = sp_vstack([X_aug_top, X_aug_bot], format='csr')
        else:
            # dense X: avoid building dense augmented (huge memory).
            # Use LinearOperator wrapper instead.
            from scipy.sparse.linalg import LinearOperator
            X_dtype = X.dtype  # capture (likely float32)
            def matvec(v):
                # FIX OOM: lsmr passes float64 v by default; X@v with
                # mismatched dtypes triggers upcast of X to float64,
                # allocating ~2x of X (533 GB for 280 GB float32 X).
                # Cast v to X's dtype to keep matmul in float32.
                v_cast = v.astype(X_dtype, copy=False) if v.dtype != X_dtype else v
                Xv = X @ v_cast
                Iv = (sqrt_alpha * v).astype(X_dtype, copy=False)
                return np.concatenate([Xv, Iv])
            def rmatvec(u):
                u_top = u[:n_samples]
                u_bot = u[n_samples:]
                u_top_cast = u_top.astype(X_dtype, copy=False) if u_top.dtype != X_dtype else u_top
                XTu = X.T @ u_top_cast
                return XTu + (sqrt_alpha * u_bot).astype(X_dtype, copy=False)
            X_aug = LinearOperator(
                (n_samples + n_features, n_features),
                matvec=matvec, rmatvec=rmatvec, dtype=X.dtype,
            )
        y_aug = np.concatenate([y, np.zeros(n_features, dtype=y.dtype)])
        if self.verbose:
            print(f'[CelerALassoCV] Stage 1: ridge regression '
                  f'(α={self.ridge_alpha}, lsmr)', flush=True)
        result = lsmr(X_aug, y_aug, atol=1e-6, btol=1e-6, maxiter=2000, conlim=1e16)
        beta0 = result[0]
        if self.verbose:
            n_iters = result[2]
            print(f'[CelerALassoCV] Stage 1 done: {n_iters} iters, '
                  f'||β0||={np.linalg.norm(beta0):.4e}', flush=True)
        return beta0.astype(np.float32)

    def fit(self, X, y, sample_weight=None):
        if self.verbose:
            print(f'[CelerALassoCV] gamma={self.gamma}, init={self.init_type}, '
                  f'ridge_alpha={self.ridge_alpha}, eps={self.weight_eps}',
                  flush=True)
        # Stage 1: initial estimate
        beta0 = self._initial_estimate(X, y)
        # Compute weights (small β → big penalty)
        self._weights = 1.0 / (np.abs(beta0) + self.weight_eps) ** self.gamma
        self._weights = self._weights.astype(np.float32)
        if self.verbose:
            wmin, wmax = self._weights.min(), self._weights.max()
            print(f'[CelerALassoCV] weights: min={wmin:.3e} max={wmax:.3e} '
                  f'(ratio={wmax/wmin:.2e})', flush=True)
        # Stage 2: delegate to parent's fit() but with weights set
        return super().fit(X, y, sample_weight=sample_weight)

# Monkey-patch parent's _fit_one_fold_inplace and final fit to use weights
# IF self._weights is set. This avoids duplicating ~80 lines.
_orig_fold = CelerLassoCV._fit_one_fold_inplace
def _fold_with_weights(self, X, y, train_idx, val_idx, sample_weight):
    # Only ALASSO subclass sets _weights
    weights = getattr(self, '_weights', None)
    if weights is None:
        return _orig_fold(self, X, y, train_idx, val_idx, sample_weight)
    # Otherwise: same as parent but pass weights to celer_path
    # FIX OOM: same as parent, avoid asfortranarray double copy
    if not _sp.issparse(X):
        X_tr = X[train_idx]
        if X_tr.dtype != np.float32:
            X_tr = X_tr.astype(np.float32, copy=False)
    else:
        X_tr = X[train_idx]
    y_tr = np.ascontiguousarray(y[train_idx], dtype=X.dtype if not _sp.issparse(X) else None)
    y_va = y[val_idx]
    _, coefs, _ = _celer_path(
        X_tr, y_tr, pb='lasso',
        alphas=self.alphas_sorted_desc.astype(np.float32),
        weights=weights,                # ← ALASSO key
        max_iter=self.max_iter, tol=self.tol, verbose=0,
    )
    del X_tr, y_tr
    n_alphas = coefs.shape[1]
    mse = np.zeros(n_alphas)
    for i in range(n_alphas):
        y_pred_full = X @ coefs[:, i]
        err = y_pred_full[val_idx] - y_va
        mse[i] = float(np.mean(err ** 2))
    return mse
CelerLassoCV._fit_one_fold_inplace = _fold_with_weights
# ===== end ALASSO wrapper =====




# ===== begin [PATCH tsqr] RFE-OLS-TSQR =====
import scipy.linalg as _tsqr_la

class PheasyRFE_OLS_TSQR:
    """严格 OLS (alpha=0) + RFE, 用 Q-less 增广 Tall-Skinny QR.

    核心: 对增广矩阵 [A_active | b] 做 Q-less TSQR 得 R_aug ((na+1)x(na+1)).
      - R_aug[:na,:na] = R,  R_aug[:na,na] = Q^T b,  R_aug[na,na] = 残差 rho
      - 解 R x = Q^T b 得 coef;  BIC = m*ln(rho^2/m) + k*ln(m) 用于判停
    继承: 删列时对 R_aug[:, keep+bcol] 重新三角化 (n空间, 不碰原始行).
    周期性重算: 每 recalibrate 轮重做完整 TSQR; R 对角过小也强制重算.
    """
    def __init__(self, step=0.05, patience=5, min_features=100,
                 block_rows=40000, recalibrate=4, diag_floor=1e-12,
                 verbose=True):
        self.step = float(step)
        self.patience = int(patience)
        self.min_features = int(min_features)
        self.block_rows = int(block_rows)
        self.recalibrate = max(1, int(recalibrate))
        self.diag_floor = float(diag_floor)
        self.verbose = verbose
        # sklearn-style attrs
        self.coef_ = None
        self.support_ = None
        self.n_features_in_ = None
        self.best_bic_ = None
        self.ridge_alpha = 0.0
        self.n_iter_ = None

    # ---- Q-less 增广 TSQR (完整, 扫全部行) ----
    def _aug_tsqr_full(self, SM, active_idx, y):
        m = SM.shape[0]
        n = len(active_idx)
        br = self.block_rows
        R_aug = None
        is_sparse = _sp.issparse(SM)
        for s in range(0, m, br):
            e = min(s + br, m)
            blk = SM[s:e]
            if is_sparse:
                A_blk = np.asarray(blk[:, active_idx].todense(), dtype=np.float64)
            else:
                A_blk = np.asarray(blk[:, active_idx], dtype=np.float64)
            b_blk = np.asarray(y[s:e], dtype=np.float64).reshape(-1, 1)
            Ab = np.concatenate([A_blk, b_blk], axis=1)
            del A_blk
            stacked = Ab if R_aug is None else np.concatenate([R_aug, Ab], axis=0)
            del Ab
            R_full = _tsqr_la.qr(stacked, mode='r', check_finite=False)[0]
            del stacked
            R_aug = np.ascontiguousarray(R_full[:n+1, :n+1])
            del R_full
        return R_aug

    # ---- R 空间继承: 删列后重新三角化 ----
    def _inherit(self, R_aug_prev, keep_local):
        na_prev = R_aug_prev.shape[0] - 1
        cols = list(keep_local) + [na_prev]   # 保留 b 列
        sub = R_aug_prev[:, cols]
        R_full = _tsqr_la.qr(sub, mode='r', check_finite=False)[0]
        na_new = len(keep_local)
        return np.ascontiguousarray(R_full[:na_new+1, :na_new+1])

    # ---- 从 R_aug 解 coef + 残差 ----
    def _solve(self, R_aug):
        na = R_aug.shape[0] - 1
        R = R_aug[:na, :na]
        z = R_aug[:na, na]
        diag = np.abs(np.diag(R))
        x = _tsqr_la.solve_triangular(R, z, check_finite=False)
        rho = abs(float(R_aug[na, na]))
        return x, rho, diag.min(), diag.max()

    def fit(self, SM, y, sample_weight=None):
        m = SM.shape[0]
        n_full = SM.shape[1]
        self.n_features_in_ = n_full
        active = np.ones(n_full, dtype=bool)
        cur_idx = np.arange(n_full)              # active 的全局列索引
        R_aug = None
        rounds_since_recal = 0

        best_bic = np.inf
        best_support = active.copy()
        best_coef_full = None
        no_improve = 0
        rnd = 0

        if self.verbose:
            print(f"[RFE-OLS-TSQR] START n_features={n_full}, step={self.step}, "
                  f"recalibrate={self.recalibrate}, block_rows={self.block_rows}",
                  flush=True)

        import time as _t
        while True:
            na = len(cur_idx)
            t0 = _t.time()
            # 决定本轮重算还是继承
            need_full = (R_aug is None) or (rounds_since_recal >= self.recalibrate)
            if need_full:
                R_aug = self._aug_tsqr_full(SM, cur_idx, y)
                rounds_since_recal = 0
                mode = "FULL"
            else:
                mode = "inherit"

            x, rho, dmin, dmax = self._solve(R_aug)

            # 病态保护: R 对角过小 -> 本轮结果不可信, 强制重算一次
            if dmin < self.diag_floor and mode == "inherit":
                if self.verbose:
                    print(f"[RFE-OLS-TSQR] Round {rnd}: diag_min={dmin:.2e} "
                          f"< floor, 强制重算", flush=True)
                R_aug = self._aug_tsqr_full(SM, cur_idx, y)
                rounds_since_recal = 0
                mode = "FULL(forced)"
                x, rho, dmin, dmax = self._solve(R_aug)

            # BIC (m 个残差平方, k 个有效参数)
            k = na
            mse = (rho * rho) / m
            bic = m * np.log(mse + 1e-300) + k * np.log(m)
            dt = _t.time() - t0

            if self.verbose:
                print(f"[RFE-OLS-TSQR] Round {rnd:3d} [{mode:12s}]: "
                      f"n_active={na:7d}  rho={rho:.4e}  BIC={bic:.4e}  "
                      f"diag[min={dmin:.2e},max={dmax:.2e}]  cond~{dmax/max(dmin,1e-300):.2e}  "
                      f"t={dt:.1f}s", flush=True)

            # 记录最优
            if (not np.isfinite(best_bic)) or bic < best_bic - abs(best_bic) * 1e-9:
                best_bic = bic
                best_support = np.zeros(n_full, dtype=bool)
                best_support[cur_idx] = True
                best_coef_full = np.zeros(n_full, dtype=np.float64)
                best_coef_full[cur_idx] = x
                no_improve = 0
            else:
                no_improve += 1
                if self.verbose:
                    print(f"[RFE-OLS-TSQR]   无改善 ({no_improve}/{self.patience}): "
                          f"BIC={bic:.4e} vs best={best_bic:.4e}", flush=True)
                if no_improve >= self.patience:
                    if self.verbose:
                        print(f"[RFE-OLS-TSQR] 连续 {self.patience} 轮无改善, 停止.",
                              flush=True)
                    break

            # 删特征
            if na <= self.min_features:
                if self.verbose:
                    print(f"[RFE-OLS-TSQR] 达到 min_features={self.min_features}, 停止.",
                          flush=True)
                break
            n_rm = max(1, int(na * self.step))
            n_keep = max(self.min_features, na - n_rm)
            n_rm = na - n_keep
            importance = np.abs(x)               # OLS coef 直接作 importance
            rm_local = np.argpartition(importance, n_rm)[:n_rm]
            keep_mask = np.ones(na, dtype=bool)
            keep_mask[rm_local] = False
            keep_local = np.where(keep_mask)[0]

            # 继承更新 R_aug (除非下轮要重算)
            R_aug = self._inherit(R_aug, keep_local)
            cur_idx = cur_idx[keep_local]
            rounds_since_recal += 1
            rnd += 1

        # 最终: 用 best_support 重新做一次完整 TSQR (确保最优解精确)
        if self.verbose:
            print(f"[RFE-OLS-TSQR] 最终重算 best: n_active={int(best_support.sum())}",
                  flush=True)
        final_idx = np.where(best_support)[0]
        R_aug_final = self._aug_tsqr_full(SM, final_idx, y)
        x_final, rho_final, _, _ = self._solve(R_aug_final)
        coef_full = np.zeros(n_full, dtype=SM.dtype)
        coef_full[final_idx] = x_final.astype(SM.dtype)

        self.coef_ = coef_full
        self.support_ = best_support
        self.best_bic_ = best_bic
        self.n_iter_ = rnd
        # 供下游汇总
        self.best_rmse_cv_ = float(rho_final / np.sqrt(m))
        return self

    def predict(self, X):
        return X @ self.coef_
# ===== end [PATCH tsqr] =====

# ===== begin PheasyRFECV =====
from scipy.sparse.linalg import lsmr as _rfe_lsmr, LinearOperator as _RFE_LO
import scipy.sparse as _spm_rfe
# [PATCH mkl] MKL 多线程稀疏 matvec (实测 43x vs scipy 单线程, 192 线程)
_RFE_USE_MKL = os.environ.get("PHEASY_RFE_MKL", "1").lower() in ("1", "true", "yes")
try:
    import sparse_dot_mkl as _sdm_rfe
    from sparse_dot_mkl import dot_product_mkl as _mkl_dot_rfe
    # [PATCH ilp64] 大矩阵 nnz > 2**31 需 ILP64 (int64) 接口, 否则 MKL 拒绝.
    os.environ.setdefault("MKL_INTERFACE_LAYER", "ILP64")
    try:
        _sdm_rfe.mkl_set_interface_layer(1)  # 1 = ILP64
        print(f"[optimizer/mkl] ILP64 接口已设, int_dtype="
              f"{_sdm_rfe.mkl_interface_integer_dtype()}", flush=True)
    except Exception as _e_ilp:
        print(f"[optimizer/mkl] WARNING: set ILP64 failed: {_e_ilp}", flush=True)
except Exception:
    _RFE_USE_MKL = False

def _rfe_spmv(SM, v, dtype):
    """SM @ v, MKL 多线程优先, 回退 scipy. v 强制连续 + dtype."""
    if _RFE_USE_MKL and _spm_rfe.issparse(SM):
        return _mkl_dot_rfe(SM, np.ascontiguousarray(v, dtype=dtype))
    return SM @ v

def _rfe_spmv_T(SM, u, dtype):
    """SM.T @ u, 写成 (u_row @ SM).ravel() 避开 CSR 转置开销."""
    if _RFE_USE_MKL and _spm_rfe.issparse(SM):
        _r = _mkl_dot_rfe(np.ascontiguousarray(u, dtype=dtype).reshape(1, -1), SM)
        return np.asarray(_r).ravel()
    return SM.T @ u

# [PATCH twolevel] 两级 matvec 包装: 不生成 SM = SM_prime @ NS,
# 而是 (SM_prime @ (NS @ v)). 每次 matvec 两次 MKL 稀疏乘, 内存仅 SM_prime+NS.
from scipy.sparse.linalg import LinearOperator as _TL_LO

class TwoLevelSM(_TL_LO):
    """LinearOperator: 行为等价于 SM = SM_prime @ NS, 但从不显式相乘.
    matvec:  SM @ v   = SM_prime @ (NS @ v)
    rmatvec: SM.T @ u = NS.T @ (SM_prime.T @ u)
    两级均走 MKL (_rfe_spmv / _rfe_spmv_T).
    """
    def __init__(self, SM_prime, NS, dtype=None):
        self.SM_prime = SM_prime
        self.NS = NS
        _dt = dtype if dtype is not None else SM_prime.dtype
        self._dt = _dt
        super().__init__(_dt, (SM_prime.shape[0], NS.shape[1]))

    def _matvec(self, v):
        # NS @ v  (n_ns_rows,) ; 再 SM_prime @ 结果
        t = _rfe_spmv(self.NS, np.ascontiguousarray(v, dtype=self._dt), self._dt)
        return _rfe_spmv(self.SM_prime, np.ascontiguousarray(t, dtype=self._dt), self._dt)

    def _rmatvec(self, u):
        # SM_prime.T @ u ; 再 NS.T @ 结果
        t = _rfe_spmv_T(self.SM_prime, np.ascontiguousarray(u, dtype=self._dt), self._dt)
        return _rfe_spmv_T(self.NS, np.ascontiguousarray(t, dtype=self._dt), self._dt)

    def col_norms(self, block=4096):
        """RFE 特征排序用列范数 = 真实 ||SM[:, j]||, SM = SM_prime @ NS.

        [FIX colnorm 2026-08] 旧实现返回 ||NS[:, j]||. NS 是对称性约束零空间的
        SVD 正交基, 列范数恒等于 1 -> min=max=1.00e+00, 归一化完全失效:
          * _ridge_lsmr 里的 _inv_d 全为 1, 列缩放没起作用;
          * fit() 里 |coef|*col_norms 退化成裸 |coef|, 而 2 阶列量纲 eV/A^2、
            3 阶 eV/A^3, 位移 RMS ~0.017 A 使两者尺度差约两个数量级
            -> RFE 变成"按量纲排序", 一剪 CV 就崩.

        精确算法要对全部 m 行做 SpMM (旧注释实测 27min/块). 这里改为按行子采样:
        ||SM[:,j]||^2 的行求和是可无偏估计的, 抽 f 帧后乘 sqrt(m/m_sub) 还原.
        相对误差 ~1/sqrt(m_sub); 24 帧 x 1488 行 = 3.6e4 行 -> ~0.5%,
        对排序绰绰有余, 耗时降到全量的 ~1/29.

        env:
            PHEASY_COLNORM_FRAMES  抽样帧数 (default 24; <=0 或 >=总帧数 = 精确)
            PHEASY_COLNORM_EXACT   =1 强制全量精确
            PHEASY_CV_GROUP_SIZE   每帧行数 = 3 x 超胞原子数 (用于按帧抽样)
        """
        import numpy as _np
        import time as _t
        _t0 = _t.time()

        m, p = self.SM_prime.shape
        n = self.NS.shape[1]
        _exact = os.environ.get("PHEASY_COLNORM_EXACT", "0").lower() in ("1", "true", "yes")
        _nf = int(os.environ.get("PHEASY_COLNORM_FRAMES", "24"))
        _gs = int(os.environ.get("PHEASY_CV_GROUP_SIZE", "0"))

        # ── 选行 ────────────────────────────────────────────────────────────
        _rng = _np.random.default_rng(12345)
        if _exact or _nf <= 0:
            _A, _scale, _how = self.SM_prime, 1.0, f"exact ({m} 行)"
        elif _gs > 0 and m % _gs == 0 and _nf < m // _gs:
            # 按整帧抽样: 同帧内 3N 行强相关, 整帧取更能代表真实行分布
            _ntot = m // _gs
            _f = _np.sort(_rng.choice(_ntot, size=_nf, replace=False))
            _rows = (_f[:, None] * _gs + _np.arange(_gs)[None, :]).ravel()
            _A = self.SM_prime[_rows]
            _scale = _np.sqrt(_ntot / float(_nf))
            _how = f"{_nf}/{_ntot} 帧抽样 ({len(_rows)} 行)"
        else:
            # 无 group 信息 (或帧数不足): 退回随机行抽样, 同样无偏
            _nr = min(m, max(20000, _nf * 1500))
            _rows = _np.sort(_rng.choice(m, size=_nr, replace=False))
            _A = self.SM_prime[_rows]
            _scale = _np.sqrt(m / float(_nr))
            _how = f"{_nr}/{m} 随机行抽样"
            if _gs > 0 and m % _gs != 0:
                print(f"[TwoLevelSM.col_norms] WARNING: m={m} 不能被 "
                      f"PHEASY_CV_GROUP_SIZE={_gs} 整除, 退回随机行抽样", flush=True)

        if _A.format != 'csr':
            _A = _A.tocsr()
        _NS = self.NS.tocsc() if self.NS.format != 'csc' else self.NS

        # ── 分块 SpMM: SM_sub[:, blk] = A @ NS[:, blk], 只留列平方和 ────────
        _sq = _np.zeros(n, dtype=_np.float64)
        for _j0 in range(0, n, block):
            _j1 = min(_j0 + block, n)
            _B = _NS[:, _j0:_j1]
            _B = _np.ascontiguousarray(_B.toarray(), dtype=self._dt)
            if _RFE_USE_MKL:
                _S = _mkl_dot_rfe(_A, _B)
            else:
                _S = _A @ _B
            _S = _np.asarray(_S)
            _sq[_j0:_j1] = _np.einsum('ij,ij->j', _S, _S, dtype=_np.float64)
            del _S, _B

        cn = _scale * _np.sqrt(_sq)
        _nz = int((cn > 1e-30).sum())
        _pos = cn[cn > 1e-30]
        print(f"[TwoLevelSM.col_norms] 真实 SM 列范数 [{_how}] "
              f"{_t.time()-_t0:.1f}s, nz={_nz}/{n}", flush=True)
        if _nz:
            _rng_ratio = float(_pos.max() / _pos.min())
            print(f"[TwoLevelSM.col_norms] min={_pos.min():.4e}, "
                  f"max={_pos.max():.4e}, median={_np.median(_pos):.4e}, "
                  f"max/min={_rng_ratio:.2e}", flush=True)
            if _rng_ratio < 1.5:
                print("[TwoLevelSM.col_norms] WARNING: 列范数动态范围 < 1.5. "
                      "真实设计矩阵不应如此均匀 —— 极可能仍在对 NS 求范数, "
                      "请检查 SM_prime/NS 是否传反.", flush=True)
        return cn


class PheasyRFECV:
    """Recursive Feature Elimination with Cross-Validation.

    接受 dense float32 SM (141 GB)，全程不复制子矩阵。
    用 LinearOperator 做列掩码: matvec = SM @ v_full (全列), 无 slice 复制。
    Ridge via lsmr 作为 base estimator。
    K-fold CV 并行 (joblib threads)。

    并行注意:
      每个 fold 内部调用 BLAS GEMV (受 OPENBLAS_NUM_THREADS 控制).
      若 cv=5 且 OPENBLAS_NUM_THREADS=32, 总线程数 = 160.
      建议: 192 核以上节点可保持默认; 小节点应设 OPENBLAS_NUM_THREADS=核数/cv.

    env vars:
        PHEASY_RFE_STEP          每轮消除比例    (default 0.1)
        PHEASY_RFE_RIDGE_ALPHA   Ridge 正则化    (default 1e-5)
        PHEASY_RFE_MIN_FEATURES  最少保留特征数  (default 1)
        PHEASY_RFE_LSMR_MAXITER  lsmr 最大迭代  (default 3000)
        PHEASY_RFE_LSMR_ATOL     lsmr atol      (default 1e-6)
        PHEASY_RFE_LSMR_BTOL     lsmr btol      (default 1e-6)
        PHEASY_N_JOBS            CV fold 并行数  (default -1)
    """

    def __init__(self, step=0.1, cv=5, ridge_alpha=1e-5,
                 lsmr_maxiter=3000, lsmr_atol=1e-6, lsmr_btol=1e-6,
                 n_jobs=-1, min_features=1,
                 verbose=True, random_state=None):
        _e = os.environ.get
        self.step         = float(_e("PHEASY_RFE_STEP",         str(step)))
        self.cv           = int(cv)
        self.ridge_alpha  = float(_e("PHEASY_RFE_RIDGE_ALPHA",  str(ridge_alpha)))
        self.lsmr_maxiter = int(  _e("PHEASY_RFE_LSMR_MAXITER", str(lsmr_maxiter)))
        self.lsmr_atol    = float(_e("PHEASY_RFE_LSMR_ATOL",    str(lsmr_atol)))
        self.lsmr_btol    = float(_e("PHEASY_RFE_LSMR_BTOL",    str(lsmr_btol)))
        self.n_jobs       = int(  _e("PHEASY_N_JOBS",
                                     str(n_jobs if n_jobs is not None else -1)))
        self.min_features = int(  _e("PHEASY_RFE_MIN_FEATURES", str(min_features)))
        self.verbose      = verbose
        self.random_state = 42 if random_state is None else random_state

        # 拟合后填充
        self.coef_         = None
        self.intercept_    = 0.0
        self.n_features_in_= None
        self.support_      = None
        self.best_rmse_cv_ = float('inf')
        self.n_iter_       = 0
        self.alpha_        = self.ridge_alpha
        self.alphas_       = np.array([self.ridge_alpha])
        self.mse_path_     = None

        # [PATCH warmstart] warm start 配置
        self._last_coef_full = None
        self._warm_start = (os.environ.get("PHEASY_RFE_WARM_START", "0") == "1")

    # ── LinearOperator 构建 (零复制) ─────────────────────────────────────────
    def _make_op(self, SM, active_idx, row_idx=None):
        """SM[row_idx, :][:, active_idx] 的 LinearOperator, 不复制 SM."""
        n_rows_full, n_cols_full = SM.shape
        n_active = len(active_idx)
        dtype = SM.dtype

        if row_idx is None:
            n_rows_op = n_rows_full

            def _mv(v):
                v_full = np.zeros(n_cols_full, dtype=dtype)
                v_full[active_idx] = v.astype(dtype, copy=False)
                return _rfe_spmv(SM, v_full, dtype)

            def _rmv(u):
                return _rfe_spmv_T(SM, u, dtype)[active_idx]
        else:
            n_rows_op = len(row_idx)

            def _mv(v):
                v_full = np.zeros(n_cols_full, dtype=dtype)
                v_full[active_idx] = v.astype(dtype, copy=False)
                return _rfe_spmv(SM, v_full, dtype)[row_idx]

            def _rmv(u):
                u_full = np.zeros(n_rows_full, dtype=dtype)
                u_full[row_idx] = u.astype(dtype, copy=False)
                return _rfe_spmv_T(SM, u_full, dtype)[active_idx]

        return _RFE_LO(
            (n_rows_op, n_active),
            matvec=_mv, rmatvec=_rmv, dtype=dtype,
        )

    # ── lsmr Ridge ────────────────────────────────────────────────────────────
    def _ridge_lsmr(self, SM, active_idx, y, row_idx=None):
        """Ridge via lsmr 增广系统法, 无复制求解."""
        if len(active_idx) == 0:
            raise ValueError("_ridge_lsmr: empty active_idx")

        dtype = SM.dtype
        n_active = len(active_idx)
        n_eq = len(row_idx) if row_idx is not None else SM.shape[0]
        sqrt_alpha = float(np.sqrt(self.ridge_alpha * n_eq))

        # [PATCH colnorm] 嵌入列缩放: 等效 A' = A * diag(1/d) 但不复制 SM.
        # 内部 mv/rmv 闭包捕获 inv_d (局部变量), 多次调用各自独立, 无 alias.
        if getattr(self, '_col_norms', None) is None:
            # 兼容路径: 若 fit() 未先于此调用 (单元测试场景), 用 1.0 占位
            _inv_d = np.ones(n_active, dtype=dtype)
        else:
            _inv_d = (1.0 / self._col_norms[active_idx]).astype(dtype)
        n_rows_full, n_cols_full = SM.shape

        if row_idx is None:
            n_rows_op = n_rows_full

            def _mv_scaled(v, _idx=active_idx, _id=_inv_d):
                v_full = np.zeros(n_cols_full, dtype=dtype)
                v_full[_idx] = v.astype(dtype, copy=False) * _id
                return _rfe_spmv(SM, v_full, dtype)

            def _rmv_scaled(u, _idx=active_idx, _id=_inv_d):
                return _rfe_spmv_T(SM, u, dtype)[_idx] * _id
        else:
            n_rows_op = len(row_idx)
            _row_idx_local = row_idx  # 闭包绑定

            def _mv_scaled(v, _idx=active_idx, _id=_inv_d, _ri=_row_idx_local):
                v_full = np.zeros(n_cols_full, dtype=dtype)
                v_full[_idx] = v.astype(dtype, copy=False) * _id
                return _rfe_spmv(SM, v_full, dtype)[_ri]

            def _rmv_scaled(u, _idx=active_idx, _id=_inv_d,
                            _ri=_row_idx_local, _nrf=n_rows_full):
                u_full = np.zeros(_nrf, dtype=dtype)
                u_full[_ri] = u.astype(dtype, copy=False)
                return _rfe_spmv_T(SM, u_full, dtype)[_idx] * _id

        data_op = _RFE_LO(
            (n_rows_op, n_active),
            matvec=_mv_scaled, rmatvec=_rmv_scaled, dtype=dtype,
        )
        # [/PATCH colnorm]

        def _mv_aug(v):
            v = v.astype(dtype, copy=False)
            return np.concatenate([data_op.matvec(v), sqrt_alpha * v])

        def _rmv_aug(u):
            u = u.astype(dtype, copy=False)
            return (data_op.rmatvec(u[:n_eq])
                    + sqrt_alpha * u[n_eq:])

        aug_op = _RFE_LO(
            (n_eq + n_active, n_active),
            matvec=_mv_aug, rmatvec=_rmv_aug, dtype=dtype,
        )
        y_aug = np.concatenate([
            y.astype(dtype),
            np.zeros(n_active, dtype=dtype),
        ])
        # [PATCH warmstart] 构造 x0 (归一化空间)
        # 触发条件: warm_start=ON, row_idx=None (主循环, 非 CV fold), 有上轮 coef
        _x0 = None
        if (self._warm_start
                and row_idx is None
                and self._last_coef_full is not None):
            _x_prev_active = self._last_coef_full[active_idx]
            if getattr(self, '_col_norms', None) is not None:
                # 归一化空间: y = x * d  (反变换是 x = y / d)
                _x0 = (_x_prev_active * self._col_norms[active_idx]).astype(dtype)
            else:
                _x0 = _x_prev_active.astype(dtype)

        res = _rfe_lsmr(
            aug_op, y_aug,
            atol=self.lsmr_atol, btol=self.lsmr_btol,
            maxiter=self.lsmr_maxiter, conlim=1e16, show=False,
            x0=_x0,
        )
        # [/PATCH warmstart]
        # istop legend: 1/2/4/5 = converged; 3=conlim exceeded,
        # 6=cond too large at machine precision, 7=maxiter reached.
        if res[1] in (3, 6, 7):
            print(f'[_ridge_lsmr] WARNING: lsmr did NOT converge cleanly '
                  f'(istop={res[1]}, itn={res[2]}/{self.lsmr_maxiter}). '
                  f'Consider raising PHEASY_RFE_LSMR_MAXITER.', flush=True)
        # [PATCH colnorm] 反变换: 归一化空间 y → 原始空间 x = y / d = y * inv_d
        return (res[0].astype(dtype) * _inv_d)

    # ── K-fold CV RMSE ────────────────────────────────────────────────────────
    def _cv_rmse(self, SM, active_idx, y):
        from sklearn.model_selection import KFold, GroupKFold
        n_rows = SM.shape[0]
        dtype = SM.dtype

        # group_size = 每个配置的行数 = 3 × 超胞原子数 (run_pheasy 自动注入)
        group_size = int(os.environ.get("PHEASY_CV_GROUP_SIZE", "0"))
        if group_size > 0:
            if n_rows % group_size != 0:
                raise ValueError(
                    "[PheasyRFECV] n_rows=%d 不能被 PHEASY_CV_GROUP_SIZE=%d 整除; "
                    "group_size 应 = 3×超胞原子数." % (n_rows, group_size))
            groups = np.arange(n_rows) // group_size
            gkf = GroupKFold(n_splits=self.cv)
            splits = list(gkf.split(np.arange(n_rows), groups=groups))
            # [FIX cvlog] 显式确认按构型分折已生效 (原来只有失败时才打印)
            if self.verbose and active_idx.size == self.n_features_in_:
                print(f"[PheasyRFECV/cv] GroupKFold 按构型分折: "
                      f"{n_rows // group_size} 帧 × {group_size} 行 → "
                      f"{self.cv} 折, 无配置内泄漏.", flush=True)
        else:
            if self.verbose and active_idx.size == self.n_features_in_:
                print("[PheasyRFECV] WARNING: PHEASY_CV_GROUP_SIZE 未设置, "
                      "退回按行 KFold (存在配置内信息泄漏, CV 偏乐观).", flush=True)
            kf = KFold(n_splits=self.cv, shuffle=True,
                       random_state=self.random_state)
            splits = list(kf.split(np.arange(n_rows)))

        def _fold(tr_idx, va_idx):
            coef = self._ridge_lsmr(SM, active_idx, y[tr_idx], row_idx=tr_idx)
            v_full = np.zeros(SM.shape[1], dtype=dtype)
            v_full[active_idx] = coef
            y_pred_va = _rfe_spmv(SM, v_full, dtype)[va_idx]
            err = y_pred_va - y[va_idx]
            return float(np.mean(err * err))

        n_par = min(self.cv,
                    abs(self.n_jobs) if self.n_jobs != -1 else self.cv)
        mse_list = Parallel(n_jobs=n_par, prefer="threads")(
            delayed(_fold)(tr, va) for tr, va in splits
        )
        # [FIX 1se] 同时返回逐折 RMSE, 供 fit() 计算标准误做 1-SE 选型
        fold_rmse = np.sqrt(np.asarray(mse_list, dtype=np.float64))
        return float(np.sqrt(np.mean(mse_list))), fold_rmse

    def fit(self, SM, y, sample_weight=None):
        import time as _t
        import scipy.sparse as _spm
        # [PATCH mkl] 兜底设 MKL 线程数 (探测确认计算节点可设到 192)
        if _RFE_USE_MKL:
            try:
                _sdm_rfe.mkl_set_num_threads(
                    int(os.environ.get("MKL_NUM_THREADS", "192")))
                if self.verbose:
                    print(f"[PheasyRFECV/mkl] MKL matvec ON, "
                          f"threads={_sdm_rfe.mkl_get_max_threads()}", flush=True)
            except Exception as _e:
                print(f"[PheasyRFECV/mkl] set threads failed: {_e}", flush=True)
        elif self.verbose:
            print("[PheasyRFECV/mkl] MKL OFF, 回退 scipy 单线程 matvec", flush=True)
        # [PATCH warmstart] 每次 fit 重置 warm start 状态, 避免跨调用污染
        self._last_coef_full = None
        n_rows, n_features = SM.shape
        self.n_features_in_ = n_features

        # Scale-invariant RFE importance: |coef_j| * ||SM[:,j]||
        if isinstance(SM, TwoLevelSM):
            # [PATCH twolevel-colnorm] 两级 matvec 无完整 SM, 分块算列范数
            _col_norms = SM.col_norms()
            if self.verbose:
                print(f"[PheasyRFECV/colnorm] TwoLevelSM 分块列范数 done", flush=True)
        elif _spm.issparse(SM):
            _ss = np.asarray(SM.multiply(SM).sum(axis=0)).ravel()
            _col_norms = np.sqrt(_ss.astype(np.float64))
        else:
            _col_norms = np.zeros(n_features, dtype=np.float64)
            _CHUNK = 50000
            for _r0 in range(0, n_rows, _CHUNK):
                _blk = SM[_r0:_r0 + _CHUNK]
                _col_norms += np.einsum('ij,ij->j', _blk, _blk, dtype=np.float64)
            _col_norms = np.sqrt(_col_norms)

        # [PATCH colnorm] 保存 col_norms 为 self 属性, 供 _ridge_lsmr 使用.
        # 列范数极小 (常数零列, 或近零列) 用 1.0 占位避免除零;
        # 这些列对应的 SM 列为 0, 经反变换后 coef 仍为 0, 数值无影响.
        self._col_norms = np.where(_col_norms > 1e-30,
                                   _col_norms, 1.0).astype(np.float64)
        if self.verbose:
            _nz = int(np.sum(_col_norms > 1e-30))
            print(f"[PheasyRFECV/colnorm] col_norms ready: "
                  f"min={_col_norms[_col_norms > 1e-30].min():.2e}, "
                  f"max={_col_norms.max():.2e}, "
                  f"nz_cols={_nz}/{n_features}", flush=True)
        # [/PATCH colnorm]

        active = np.ones(n_features, dtype=bool)
        n_active = n_features
        best_rmse = float('inf')
        best_active = active.copy()
        round_num = 0

        # ── 判停参数 (env 可调) ──────────────────────────────────────
        # patience: 连续多少轮无"有效改善"才停 (对标 ALAMODE STOP_CRITERION)
        # rel_tol : 改善需超过 prev_best*(1-rel_tol) 才算有效 (吸收 CV 噪声)
        _patience = int(os.environ.get("PHEASY_RFE_PATIENCE", "5"))
        _rel_tol  = float(os.environ.get("PHEASY_RFE_TOL", "0.005"))
        n_no_improve = 0
        # [FIX 1se] 选型规则: argmin 会把噪声级"改善"(如 0.14% < rel_tol)当成最优,
        # 与 rel_tol 判停自相矛盾. 1-SE 规则 = 在 CV 落入 (min + 1 标准误) 的模型里
        # 取最简者, 是 RFECV/glmnet 的标准做法, 也让 rel_tol 真正生效.
        _one_se = os.environ.get("PHEASY_RFE_ONE_SE", "1").lower() in ("1", "true", "yes")
        _history = []

        if self.verbose:
            print(
                f"[PheasyRFECV] START  n_features={n_features}, "
                f"step={self.step:.2f}, cv={self.cv}, "
                f"ridge_alpha={self.ridge_alpha:.2e}, "
                f"n_jobs={self.n_jobs}, lsmr_maxiter={self.lsmr_maxiter}",
                flush=True,
            )
            print(f"[PheasyRFECV] 判停: patience={_patience}, "
                  f"rel_tol={_rel_tol:.3g} | 选型: "
                  f"{'1-SE 规则' if _one_se else '历史 argmin'}", flush=True)

        while n_active > self.min_features:
            t0 = _t.time()
            active_idx = np.where(active)[0]

            coef_active = self._ridge_lsmr(SM, active_idx, y)

            # [PATCH warmstart] 保存本轮原始空间 full coef 供下一轮 lsmr 用作 x0
            if self._warm_start:
                _cf = np.zeros(n_features, dtype=SM.dtype)
                _cf[active_idx] = coef_active
                self._last_coef_full = _cf
                if self.verbose and round_num == 0:
                    print(f"[PheasyRFECV/warmstart] enabled, "
                          f"Round 1+ will use Round 0 coef as x0", flush=True)

            cv_rmse, _fold_rmse = self._cv_rmse(SM, active_idx, y)

            # [FIX 1se] 记录历史: (n_active, CV_RMSE, SE, active mask)
            _se = (float(_fold_rmse.std(ddof=1) / np.sqrt(len(_fold_rmse)))
                   if len(_fold_rmse) > 1 else 0.0)
            _history.append((n_active, cv_rmse, _se, active.copy()))

            elapsed = _t.time() - t0
            if self.verbose:
                print(
                    f"[PheasyRFECV] Round {round_num:3d}: "
                    f"n_active={n_active:6d}  "
                    f"CV_RMSE={cv_rmse:.6e} ± {_se:.2e}  "
                    f"fold=[{_fold_rmse.min():.3e},{_fold_rmse.max():.3e}]  "
                    f"nonzero={int(np.count_nonzero(coef_active)):6d}  "
                    f"t={elapsed:.0f}s",
                    flush=True,
                )

            # 始终用 prev_best 判断"有效改善", 再更新 best (保证最终为真 argmin)
            prev_best = best_rmse
            if cv_rmse < best_rmse:
                best_rmse = cv_rmse
                best_active = active.copy()

            if cv_rmse < prev_best * (1.0 - _rel_tol):
                n_no_improve = 0          # 有效改善, 重置计数
            else:
                n_no_improve += 1
                if self.verbose:
                    print(
                        f"[PheasyRFECV] 无有效改善 "
                        f"({n_no_improve}/{_patience}): "
                        f"CV_RMSE={cv_rmse:.4e} vs best={best_rmse:.4e}",
                        flush=True,
                    )
                if n_no_improve >= _patience:
                    if self.verbose:
                        print(
                            f"[PheasyRFECV] 连续 {_patience} 轮无有效改善, 停止. "
                            f"best: n_active={int(best_active.sum())}, "
                            f"best_CV_RMSE={best_rmse:.6e}",
                            flush=True,
                        )
                    break

            n_remove = max(1, int(n_active * self.step))
            n_keep = max(self.min_features, n_active - n_remove)
            n_remove = n_active - n_keep
            _importance = np.abs(coef_active).astype(np.float64) * _col_norms[active_idx]
            remove_local = np.argpartition(_importance, n_remove)[:n_remove]
            active[active_idx[remove_local]] = False
            n_active = int(active.sum())
            round_num += 1

            if n_active <= self.min_features:
                break

        self.n_iter_ = round_num

        # [FIX 1se] 用 1-SE 规则从历史中选型 (默认开启)
        if _one_se and _history:
            _i_min = int(np.argmin([h[1] for h in _history]))
            _n_min, _m_min, _se_min, _act_min = _history[_i_min]
            _thr = _m_min + _se_min
            _cands = [h for h in _history if h[1] <= _thr]
            _chosen = min(_cands, key=lambda h: h[0])
            if self.verbose:
                print(f"[PheasyRFECV] argmin : n_active={_n_min}, "
                      f"CV={_m_min:.6e} ± {_se_min:.2e}", flush=True)
                print(f"[PheasyRFECV] 1-SE   : n_active={_chosen[0]}, "
                      f"CV={_chosen[1]:.6e}  (阈值 {_thr:.6e})", flush=True)
                _n_max = max(h[0] for h in _history)
                if _chosen[0] == _n_max:
                    print("[PheasyRFECV] NOTE: 1-SE 选中全特征集 —— "
                          "该体系超定充分, 特征选择无收益; "
                          "直接用 OLS/ridge 可省掉整轮 RFE 开销.", flush=True)
            best_active = _chosen[3]
            best_rmse   = _chosen[1]

        # 最终拟合: 最佳活跃集 + 全数据
        best_idx = np.where(best_active)[0]
        if self.verbose:
            print(
                f"[PheasyRFECV] 最终拟合: n_active={len(best_idx)}, "
                f"best_CV_RMSE={best_rmse:.6e}",
                flush=True,
            )
        coef_final = self._ridge_lsmr(SM, best_idx, y)

        coef_full = np.zeros(n_features, dtype=SM.dtype)
        coef_full[best_idx] = coef_final

        self.coef_         = coef_full
        self.support_      = best_active
        self.best_rmse_cv_ = best_rmse
        self.mse_path_     = np.array([[best_rmse ** 2]])
        self.alpha_        = self.ridge_alpha
        self.alphas_       = np.array([self.ridge_alpha])

        return self

    def predict(self, X):
        return X @ self.coef_

# ===== end PheasyRFECV =====


# ===== FIX OLS: lightweight sklearn-compatible wrapper =====
class _LsmrOLSResult:
    """sklearn LinearRegression-compatible result object backed by lsmr.

    Provides only what Optimizer.fit() and run_pheasy's reporting need:
    coef_, intercept_, n_features_in_, predict(X). No sklearn dependency.
    """
    def __init__(self, coef, n_iter=None, istop=None, normr=None):
        self.coef_ = coef
        self.intercept_ = 0.0
        self.n_features_in_ = len(coef)
        # Bookkeeping (not used by sklearn API but handy for logs)
        self.n_iter_ = n_iter
        self.istop_ = istop
        self.normr_ = normr

    def predict(self, X):
        return X @ self.coef_
# ===== end OLS wrapper =====



class Optimizer(object):
    """Intertatomic force constant optimizer.
    
    This class defines the regression methods to extract interatomic
    force constants of a lattice potential. It contains all details
    including parameters of a regressor to fit force constants. The
    implementation is based on scikit-learn package.
    
    """

    def __init__(
        self,
        method="ols",
        nalpha=100,
        alpha_min=1e-6,
        alpha_max=1e-2,
        alpha=None,
        cv=5,
        tol=1e-4,
        max_iter=20000,
        rand_seed=None,
        standardize=False,
        fit_intercept=False,
    ):
        """Initialization function.

        Parameters
        ----------
        method : str
           Fit method to get interatomic force constants. It can be
           'ols' for ordinary least-square, 'lasso' for least absolute
           shrinkage and selection operator in case of compressive sensing
           problem, and 'ridge' for ridge regression.
        nalpha : int
            The number of alpha parameters to be generated. A list of alpha
            parameters spaced evenly on a log scale will be generated.
        alpha_min : float
            The minimum value of alpha. A list of alpha parameters spaced
            evenly on a log scale will be generated.
        alpha_max : float
            The maximum value of alpha. A list of alpha parameters spaced
            evenly on a log scale will be generated.
        alpha : numpy.ndarray
            A list of parameters to control the sparseness of fitting results.
            Cross validation will be used to determine the optimal value. It 
            cannot be set together with nalpha, alpha_min and alpha_max.
        cv : int
            The fold of cross-validation splitting strategy.
        tol : float
            The tolerance for the optimization.
        max_iter: int
            The maximum number of iterations to find the solution.
        rand_seed : int
            The seed for random number generator.
        standardize : bool
            If True, the fit matrix and target values are standardized before fitting.
        fit_intercept : bool
            If True, calculate the intercept for the model.

        """
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

        if alpha is not None:
            self._alpha = np.asarray(alpha)
        else:
            # FIX: original code used `np.logspace(alpha_min, alpha_max, ...)`,
            # but logspace takes EXPONENTS not values. e.g. alpha_min=1e-6 gave
            # 10**1e-6 ≈ 1.0000023, not 1e-6. Use np.log10 to convert values.
            self._alpha = np.logspace(alpha_min, alpha_max, nalpha)

        # FIX: env switch for fast celer-based LassoCV
        self._use_celer = (
            os.environ.get("PHEASY_USE_CELER", "1") == "1" and _HAS_CELER
        )

        self._results = {}
        self._metrics = {}

    # ===== FIX OLS: memory-efficient OLS via lsmr =====
    def _ols_lsmr(self, X, y, atol=1e-8, btol=1e-8, maxiter=5000):
        """Solve OLS via scipy.sparse.linalg.lsmr (Krylov subspace method).

        Why not sklearn LinearRegression? It densifies X and does SVD,
        which for our (685968, 51590) sensing matrix needs ~500+ GB of
        workspace. lsmr only needs matvec/rmatvec, so peak memory is
        ~1× of X (sparse: ~26 GB, dense float32: ~141 GB) plus a few
        N-dim vectors.

        Handles both sparse and dense X. For dense X, wraps in a
        LinearOperator with explicit dtype-cast in matvec/rmatvec so
        scipy's internal float64 vectors don't trigger an X upcast
        (same dtype-safety trick as CelerALassoCV._initial_estimate).

        Tuning (via env vars, mostly leave defaults):
            PHEASY_OLS_ATOL   : lsmr atol (default 1e-8)
            PHEASY_OLS_BTOL   : lsmr btol (default 1e-8)
            PHEASY_OLS_MAXITER: lsmr maxiter (default 5000)
        """
        from scipy.sparse.linalg import lsmr, LinearOperator
        from scipy.sparse import issparse

        # env overrides
        atol    = float(os.environ.get("PHEASY_OLS_ATOL",    str(atol)))
        btol    = float(os.environ.get("PHEASY_OLS_BTOL",    str(btol)))
        maxiter = int(  os.environ.get("PHEASY_OLS_MAXITER", str(maxiter)))

        n_samples, n_features = X.shape
        is_sparse = issparse(X)
        X_dtype = X.dtype

        # Coerce X to a known float32/64 to avoid surprising upcasts
        if X_dtype not in (np.float32, np.float64):
            if is_sparse:
                X = X.astype(np.float32)
            else:
                X = X.astype(np.float32, copy=False)
            X_dtype = X.dtype

        print(f'[Optimizer] OLS via lsmr (Krylov, no SVD, no X^T X). '
              f'shape={X.shape}, sparse={is_sparse}, dtype={X_dtype}, '
              f'atol={atol:.0e}, btol={btol:.0e}, maxiter={maxiter}',
              flush=True)

        # Wrap as LinearOperator. This serves two purposes:
        #   1) For DENSE X: prevents scipy/lsmr from upcasting float32 X
        #      to float64 internally (would double memory).
        #   2) For SPARSE X: works identically, plus we keep dtype control.
        def matvec(v):
            v_cast = v.astype(X_dtype, copy=False) if v.dtype != X_dtype else v
            return X @ v_cast

        def rmatvec(u):
            u_cast = u.astype(X_dtype, copy=False) if u.dtype != X_dtype else u
            return X.T @ u_cast

        A_op = LinearOperator(
            (n_samples, n_features),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=X_dtype,
        )

        y_in = np.ascontiguousarray(y, dtype=X_dtype)

        import time as _t
        _t0 = _t.time()
        # [PATCH ols-ridge] 小 ridge (damp) 防病态; PHEASY_OLS_RIDGE 默认 1e-8
        _ridge = float(os.environ.get("PHEASY_OLS_RIDGE", "1e-8"))
        _damp = float(np.sqrt(_ridge * n_samples)) if _ridge > 0 else 0.0
        print(f'[Optimizer] OLS lsmr damp(ridge)={_damp:.3e} '
              f'(PHEASY_OLS_RIDGE={_ridge:.1e})', flush=True)
        result = lsmr(A_op, y_in, damp=_damp, atol=atol, btol=btol,
                      maxiter=maxiter, show=False)
        _elapsed = _t.time() - _t0

        coef   = result[0]
        istop  = result[1]
        n_iter = result[2]
        normr  = result[3]
        normar = result[4]

        # istop legend:
        #   0  x = 0 is solution
        #   1  approximate solution to A x = b
        #   2  approximate least-squares solution
        #   3  COND(A) too large (conlim)
        #   4  same as 1 (atol/btol hit)
        #   5  same as 2
        #   6  same as 3
        #   7  itn == maxiter (did NOT converge in maxiter)
        STOP_REASONS = {
            0: 'x=0 is solution',
            1: 'A x = b approximated (atol/btol)',
            2: 'least-squares solution (atol)',
            3: 'condition number too large',
            4: 'A x = b approximated (machine precision)',
            5: 'least-squares solution (machine precision)',
            6: 'condition number > conlim',
            7: 'reached maxiter — did NOT converge',
        }
        msg = STOP_REASONS.get(istop, f'unknown ({istop})')
        print(f'[Optimizer] lsmr done in {_elapsed:.1f}s: '
              f'iters={n_iter}, istop={istop} ({msg}), '
              f'||r||={normr:.4e}, ||A^T r||={normar:.4e}',
              flush=True)

        if istop in (3, 6, 7):
            print(f'[Optimizer] WARNING: lsmr did NOT converge cleanly '
                  f'(istop={istop}). Consider raising PHEASY_OLS_MAXITER '
                  f'or loosening atol/btol.', flush=True)

        # Cast coef to match the pipeline's expected dtype
        return coef.astype(np.float32 if X_dtype == np.float32 else np.float64)
    # ===== end OLS lsmr =====

    def fit(self, A, F, weights=None):

        """Fit regression model with the specified method and parameters.

        Parameters
        ----------
        A : numpy.ndarray
            2D sensing matrix.
        F : numpy.ndarray
            1D interatomic force array.
        weights : numpy.ndarray
            Weights for each sample in sensing matrix A.

        """
        """Initialization"""
        # FIX OLS: do NOT instantiate sklearn.LinearRegression here.
        # That would call scipy.linalg.lstsq -> SVD on 141 GB dense matrix
        # -> ~500 GB workspace -> OOM. We solve OLS via lsmr in the fit
        # block below, then wrap the coef in _LsmrOLSResult to keep the
        # sklearn-style API expected by downstream code.
        if self._method.upper() == "OLS":
            self._model = None   # set below in the Fit block
            print(f'[Optimizer] Using lsmr-based OLS (memory-efficient, '
                  f'no SVD, supports sparse).', flush=True)
        elif self._method.upper() == "ALASSO":
            if not self._use_celer:
                raise RuntimeError(
                    "ALASSO requires celer. Set PHEASY_USE_CELER=1 and ensure "
                    "celer is installed."
                )
            _n_jobs = int(os.environ.get('PHEASY_N_JOBS', '-1'))
            print(f'[Optimizer] Using CelerALassoCV (Adaptive LASSO). '
                  f'alphas: {self._alpha.min():.2e} to {self._alpha.max():.2e}, '
                  f'n={len(self._alpha)}, cv={self._cv}, n_jobs={_n_jobs}',
                  flush=True)
            self._model = CelerALassoCV(
                alphas=self._alpha,
                max_iter=self._max_iter,
                tol=self._tol,
                cv=self._cv,
                fit_intercept=self._fit_intercept,
                random_state=self._rand_seed,
                n_jobs=_n_jobs,
                verbose=True,
            )
        elif self._method.upper() == "LASSO":
            _n_jobs = int(os.environ.get('PHEASY_N_JOBS', '-1'))
            if self._use_celer:

                # FIX: celer is ~10-100x faster than sklearn on sparse LASSO,
                # supports multi-threaded coordinate descent natively.
                print(f'[Optimizer] Using CelerLassoCV (sparse-friendly, '
                      f'multi-threaded). alphas: {self._alpha.min():.2e} '
                      f'to {self._alpha.max():.2e}, n={len(self._alpha)}, '
                      f'cv={self._cv}, n_jobs={_n_jobs}', flush=True)
                self._model = CelerLassoCV(
                    alphas=self._alpha,
                    max_iter=self._max_iter,
                    tol=self._tol,
                    cv=self._cv,
                    fit_intercept=self._fit_intercept,
                    random_state=self._rand_seed,
                    n_jobs=_n_jobs,
                    verbose=True,
                )
            else:
                print(f'[Optimizer] Using sklearn LassoCV (set '
                      f'PHEASY_USE_CELER=1 for celer)', flush=True)
                self._model = LassoCV(
                    alphas=self._alpha,
                    max_iter=self._max_iter,
                    tol=self._tol,
                    cv=self._cv,
                    fit_intercept=self._fit_intercept,
                    random_state=self._rand_seed,
                    selection="random",
                    n_jobs=_n_jobs,
                )
        elif self._method.upper() == "RFE-OLS-TSQR":
            import os as _os_t
            self._model = PheasyRFE_OLS_TSQR(
                step=float(_os_t.environ.get('PHEASY_TSQR_STEP', '0.05')),
                patience=int(_os_t.environ.get('PHEASY_TSQR_PATIENCE', '5')),
                min_features=int(_os_t.environ.get('PHEASY_TSQR_MIN_FEATURES', '100')),
                block_rows=int(_os_t.environ.get('PHEASY_TSQR_BLOCK_ROWS', '40000')),
                recalibrate=int(_os_t.environ.get('PHEASY_TSQR_RECALIBRATE', '4')),
                diag_floor=float(_os_t.environ.get('PHEASY_TSQR_DIAG_FLOOR', '1e-12')),
                verbose=True,
            )
            print(f'[Optimizer] Using PheasyRFE_OLS_TSQR (strict OLS + RFE, '
                  f'Q-less aug-TSQR).', flush=True)
        elif self._method.upper() == "RFE":
            _rfe_n_jobs = int(os.environ.get('PHEASY_N_JOBS', '-1'))
            _rfe_step   = float(os.environ.get('PHEASY_RFE_STEP', '0.1'))
            _rfe_alpha  = float(os.environ.get('PHEASY_RFE_RIDGE_ALPHA', '1e-5'))
            _rfe_maxiter = int(os.environ.get('PHEASY_RFE_LSMR_MAXITER', '3000'))
            print(
                f'[Optimizer] Using PheasyRFECV '
                f'(Ridge+RFE, dense SM + LinearOperator masking). '
                f'step={_rfe_step:.2f}, ridge_alpha={_rfe_alpha:.2e}, '
                f'cv={self._cv}, n_jobs={_rfe_n_jobs}, lsmr_maxiter={_rfe_maxiter}',
                flush=True,
            )
            self._model = PheasyRFECV(
                step=_rfe_step,
                cv=self._cv,
                ridge_alpha=_rfe_alpha,
                lsmr_maxiter=_rfe_maxiter,
                n_jobs=_rfe_n_jobs,
                verbose=True,
                random_state=self._rand_seed,
            )
        elif self._method.upper() == "RIDGE":
            self._model = RidgeCV(
                alphas=self._alpha,
                fit_intercept=self._fit_intercept,
                store_cv_values=True,
            )

        """Fit"""
        # FIX OLS: dedicated branch — bypass sklearn LinearRegression.
        if self._method.upper() == "OLS":
            coef = self._ols_lsmr(A, F)
            # Wrap into sklearn-API-compatible object for downstream use
            # (run_pheasy reporting + the predict() call below in metrics).
            self._model = _LsmrOLSResult(coef)
        elif self._standardize:
            scaler = StandardScaler(copy=False, with_mean=False, with_std=True)
            scaler.fit_transform(A)
            F_scale = 1.0 / np.std(F)
            F_scaled = F * F_scale
            self._model.fit(A, F_scaled, sample_weight=weights)
            scaler.inverse_transform(A)
            coef = self._model.coef_ / F_scale
            scaler.transform(coef.reshape(1, -1)).reshape(-1,)
        else:
            self._model.fit(A, F, sample_weight=weights)
            coef = self._model.coef_
        self._results["coef"] = np.where(abs(coef) < 1e-6, 0, coef)
        self._model.coef_ = self._results["coef"]

        if self._method.upper() in ("LASSO", "ALASSO"):
            self._results["alpha"] = self._model.alpha_
            self._results["n_iter"] = self._model.n_iter_
            alpha_idx = np.argmin(np.abs(self._model.alphas_ - self._results["alpha"]))
            self._metrics["mse_path"] = self._model.mse_path_[alpha_idx]
            self._metrics["mse_path_mean"] = np.mean(self._model.mse_path_[alpha_idx])
            self._metrics["rmse_path"] = np.sqrt(self._metrics["mse_path"])
            self._metrics["rmse_path_mean"] = np.mean(self._metrics["rmse_path"])
            self._metrics["n_featrues"] = self._model.n_features_in_
        elif self._method.upper() == "RFE-OLS-TSQR":
            self._results["alpha"]  = 0.0
            self._results["n_iter"] = getattr(self._model, 'n_iter_', None)
            _bcv = float(getattr(self._model, 'best_rmse_cv_', 0.0))
            self._metrics["mse_path"]       = np.array([_bcv ** 2])
            self._metrics["mse_path_mean"]  = _bcv ** 2
            self._metrics["rmse_path"]      = np.array([_bcv])
            self._metrics["rmse_path_mean"] = _bcv
            self._metrics["n_featrues"]     = self._model.n_features_in_
        elif self._method.upper() == "RFE":
            self._results["alpha"]  = self._model.ridge_alpha
            self._results["n_iter"] = getattr(self._model, 'n_iter_', None)
            _best_cv = float(getattr(self._model, 'best_rmse_cv_', 0.0))
            self._metrics["mse_path"]       = np.array([_best_cv ** 2])
            self._metrics["mse_path_mean"]  = _best_cv ** 2
            self._metrics["rmse_path"]      = np.array([_best_cv])
            self._metrics["rmse_path_mean"] = _best_cv
            self._metrics["n_featrues"]     = self._model.n_features_in_
        elif self._method.upper() == "RIDGE":
            self._results["alpha"] = self._model.alpha_
        # FIX OLS: record lsmr iter count for visibility (optional)
        elif self._method.upper() == "OLS":
            self._results["n_iter"] = getattr(self._model, 'n_iter_', None)

        """Metrics"""
        eps = np.finfo(F.dtype).eps
        F_pred = self.predict(A)
        F_err = np.abs(F_pred - F)
        F_re = F_err / np.maximum(np.abs(F), eps)

        # Relative error
        self._metrics["re"] = np.sqrt(np.dot(F_err, F_err) / np.dot(F, F))
        # R^2 score
        self._metrics["r2_score"] = r2_score(F, F_pred, sample_weight=weights)
        # Mean absolute error
        self._metrics["mae"] = mean_absolute_error(F, F_pred, sample_weight=weights)
        # Mean absolute percentage error
        self._metrics["mape"] = mean_absolute_percentage_error(
            F, F_pred, sample_weight=weights
        )
        # Mean squared error
        self._metrics["mse"] = mean_squared_error(F, F_pred, sample_weight=weights)
        # Root mean squared error
        self._metrics["rmse"] = np.sqrt(self._metrics["mse"])

        # ===== Train/Test 误差对比 (只加打印, 不改计算) =====
        _train_rmse = float(self._metrics["rmse"])
        _train_re   = float(self._metrics["re"])
        # CV_RMSE (test, 留出验证) 仅 RFE 的 model 有 best_rmse_cv_
        _test_rmse = getattr(self._model, "best_rmse_cv_", None)
        print("", flush=True)
        print("========== 误差汇总 ==========", flush=True)
        print(f"  Train RMSE: {_train_rmse:.4e} eV/A   (拟合集自身残差)", flush=True)
        if _test_rmse is not None and _test_rmse != float("inf"):
            _ratio = _test_rmse / _train_rmse if _train_rmse > 0 else float("nan")
            print(f"  Test  RMSE: {float(_test_rmse):.4e} eV/A   "
                  f"(5-fold CV, 留出验证)", flush=True)
            print(f"  Ratio (test/train): {_ratio:.2f}   "
                  f"(接近1=好, 远大于1=过拟合)", flush=True)
        else:
            print("  Test  RMSE: N/A (非RFE方法, 无CV留出误差)", flush=True)
        print(f"  Train Relative error: {_train_re:.4e}", flush=True)
        print("==============================", flush=True)
        # Mean square percentage error
        self._metrics["mspe"] = np.average(np.square(F_re), weights=weights, axis=0)
        # Root mean square percentage error
        self._metrics["rmspe"] = np.sqrt(self._metrics["mspe"])

    def predict(self, A):
        """Predit using the trained model.

        Parameters
        ----------
        A : numpy.ndarray
            2D sensing matrix.
            
        Returns:
        -------
        numpy.ndarray
        Predicted interatomic forces.

        """
        return self._model.predict(A)

    @property
    def results(self):
        """Return fitting results."""
        return self._results

    @property
    def metrics(self):
        """Return multiple metrics of the model."""
        return self._metrics

    @property
    def model(self):
        """Return sklearn estimator instance."""
        return self._model

    def get_paras(self):
        """Return parameters used in fittings."""
        return self._model.coef_

    def __repr__(self):
        pass
