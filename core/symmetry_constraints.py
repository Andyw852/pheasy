"""Classes and functions for building symmetry constraints and null space."""
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2023 Changpeng Lin
# All rights reserved.

__all__ = ["SymmetryConstraints"]

import pickle
import datetime
import itertools
from copy import deepcopy
from collections import Counter, deque

import numpy as np
import scipy.sparse as spmat
from sklearn.preprocessing import normalize

from .utilities import (
    block_diag_sparse,
    get_permutation_tensor,
    kron_product,
    kron_product_einsum,
    null_space_dense,
)
from pheasy.basic_io import logger


class SymmetryConstraints(object):
    """Class for building symmetry constraints for interatomic force constants.
    
    The main routine that will be called when executing the code is
    'impose_symmetry_constraints' where all kinds of symmetry constraints will
    be applied, including space group (isotropy) symmetry, permutation symmetry
    for improper clusters, translational invariance and invariance conditions
    (including Born-Huang rotational invariance and Huang equilibrium conditions
    for second-order harmonic force constants only).

    """

    def __init__(
        self,
        scell,
        crys_basis=False,
        asr=True,
        rasr=None,
        do_rasr="FIT",
        nac=0,
        eps=1e-4,
    ):
        """Initialize a builder for imposing symmtery constraints.

        Parameters
        ----------
        scell : pheasy.Atoms
            Supercell structure.
        crys_basis : bool
            True to deal with rotation matrix in crystal coordinate,
            False for Cartesian coordinate.
        asr : bool
            True to apply constraints from translational invariance.
        rasr : str
            The type of rotational acoustic sum rules to be applied on IFCs.
            It can be 'BH' for imposing only Born-Huang rotational invariance,
            'H' for imposing only Huang equilibrium conditions, or 'BHH' for 
            imposing both Born-Huang rotational invariance and Huang equilibrium
            conditions. Currently, they are only applied on the second-order
            harmonic force constants.
        do_rasr : str
            The way to implement rotational acoustic sum rules (RASRs).
            'FIT' for imposing RASRs during force constant fitting, i.e.
            RASRs included in null space construction. 'PP' for imposing
            RASRs as an addtional post-processing.
        nac : int
            Nonanalytic correction due to long-range coulomb interactions.
            The allowed values are 2 and 3 for the 2D and 3D treatments,
            respectively. Set 0 to deactivate this option.
        eps : float
            Numeric tolerance used in null space construction.

        """
        self._eps = eps
        self._scell = scell
        self._crys_basis = crys_basis
        self._asr = asr
        if rasr is not None:
            self._rasr = rasr.upper()
        else:
            self._rasr = None
        self._do_rasr = do_rasr
        self._nac = nac

    @property
    def number_of_free_parameters_cluster(self):
        """Return number of free IFC parameters for each cluster.

        Returns:
        -------
        Dict
            It contains the keys of order with the values as a 
            list of free IFC parameters for each cluster.

        """
        return self._ifc_free_clus

    @property
    def number_of_free_parameters_order(self):
        """Return number of free IFC parameters of each order.

        Returns:
        -------
        Dict
            It contains the keys of order with the corresponding
            number of free IFC parameters of each order.

        """
        return self._ifc_free_order

    @property
    def total_number_of_free_parameters(self):
        """Return total number of free IFC parameters.
        """
        return self._ifc_free_tot

    def get_symmetry_constraints(self):
        """Return the symmetry constraints on each order IFCs.

        Returns:
        -------
        Dict
            The keys are order number of IFCs. The values are isotropy
            and permutation symmetry constraints for each represenative
            clusters, followed by the translational invariance. Each value
            thus has the length of number of representative clusters plus 1,
            except that harmonic IFCs can have additional invariance conditions
            at the end.

        """
        return self._constraints

    def impose_symmtery_constaints(self, cluster_space, rigid=None):
        """Build constraint matrix from all kinds of symmetry that IFCs obey.

        Parameters
        ----------
        cluster_space : ClusterSpace object
            Cluster space that contains all of symmetry-distinct representative
            clusters of each order followed by their orbits.
        rigid : TODO, rigid class to calculate long-range part of IFCs.
        
        Returns:
        -------
        scipy.sparse.coo_matrix
            Full null space in COO sparse matrix form. It consists of null
            space for each order IFCs.

        """
        max_order = cluster_space.max_order
        clusters = cluster_space.get_cluster_space()
        cons_mat_dict = {}
        null_space = {}
        null_space_list = []
        ifc_free_clus = {}
        ifc_free_order = {}

        for order in range(2, max_order + 1):
            start_time = datetime.datetime.now()
            logger.info("Symmetry constraints on {}-order IFCs:".format(order))
            logger.info("- Imposing space group and permuation symmetry.")
            cons_mat_dict[order] = deque()
            null_space[order] = deque()
            ifc_free_clus[order] = deque()

            """Isotropy symmetry constraints."""
            for orbit in clusters[order]:
                cons_list = orbit[0].build_isotropy_symmetry_constraints(
                    self._crys_basis
                )
                """Permutation symmetry constraints for improper clusters."""
                if not orbit[0].is_proper:
                    cons_list += orbit[
                        0
                    ].build_improper_permutation_symmetry_constraints()
                ns_mat = np.eye(3 ** order)
                for cons_mat in cons_list:
                    ns_mat_tmp = null_space_dense(cons_mat.dot(ns_mat), self._eps)
                    if ns_mat_tmp.shape == (0,):
                        ns_mat = ns_mat_tmp
                        break
                    else:
                        ns_mat = ns_mat.dot(ns_mat_tmp)
                        ns_mat = normalize(ns_mat, axis=0)
                ns_mat = spmat.coo_matrix(ns_mat)
                null_space[order].append(ns_mat)
                ifc_free_clus[order].append(ns_mat.shape[1])
                cons_mat_dict[order].append(cons_mat)

            """Translational invariance."""
            if self._asr:
                logger.info("- Imposing translational invariance.")
                cons_asr = self.build_translational_invariance(clusters, order)
                cons_mat_dict[order].append(cons_asr)

            """Invariance conditions for second-order IFCs."""
            if order == 2 and self._rasr is not None:
                if self._crys_basis:
                    logger.error(
                        "Rotational invariance is only implemented"
                        + "in Cartesian coordinate system."
                    )
                    raise ValueError
                if self._rasr == "BH":
                    logger.info("- Imposing rotational invariance.")
                elif self._rasr == "H":
                    logger.info("- Imposing equilibrium conditions.")
                elif self._rasr == "BHH":
                    logger.info(
                        "- Imposing rotational invariance and equilibrium conditions."
                    )
                if self._nac != 0 and self._do_rasr == "PP":
                    cons_rasr, nac_sum = self.build_rotational_invariance(clusters[2])
                else:
                    cons_rasr = self.build_rotational_invariance(clusters[2])
                cons_mat_dict[order].append(cons_rasr)
                # [DUMP] 临时: 导出完整 RASR 约束矩阵供离线诊断
                import os as _os_d
                if _os_d.environ.get("PHEASY_DUMP_RASR", "0") == "1":
                    import scipy.sparse as _spd, numpy as _npd
                    _parts = []
                    for _cm in cons_rasr:
                        _a = _cm.toarray() if _spd.issparse(_cm) else _npd.asarray(_cm)
                        _parts.append(_spd.csr_matrix(_a))
                    _Cfull = _spd.vstack(_parts, format='csr')
                    _spd.save_npz(_os_d.path.join(
                        _os_d.environ.get("PHEASY_DUMP_DIR", "."),
                        "cons_rasr_full.npz"), _Cfull)
                    print("[DUMP] cons_rasr_full.npz saved, shape={}, {} blocks".format(
                        _Cfull.shape, len(cons_rasr)), flush=True)

            """Construct whole null space matrix of each order."""
            logger.info("- Calculating null space.")
            ns_mat = block_diag_sparse(null_space[order], order)
            if self._asr:
                import os as _os
                import numpy as _np
                import scipy.sparse as _sp
                from scipy.sparse import csc_matrix as _csc, csr_matrix as _csr
                _use_combined = _os.environ.get("PHEASY_ASR_COMBINED", "1") == "1"
                _use_sparse = _os.environ.get("PHEASY_ASR_SPARSE", "1") == "1"
                _sparse_thr = float(_os.environ.get("PHEASY_ASR_SPARSE_THR", "1e-10"))
                _col_blk = int(_os.environ.get("PHEASY_ASR_COL_BLOCK", "5000"))
                _INT32_MAX = 2 ** 31 - 1
                # Keep ns_mat sparse (was 73 GB dense before; ~99% zeros).
                # Convert to CSC for fast column slicing in projection.
                if _use_sparse:
                    if _sp.issparse(ns_mat):
                        ns_mat = ns_mat.tocsc()
                    else:
                        ns_mat = _csc(ns_mat)
                    print(f'[ASR] sparse mode ON, ns_mat: {type(ns_mat).__name__} '
                          f'shape={ns_mat.shape}, nnz={ns_mat.nnz}, '
                          f'mem~{ns_mat.data.nbytes/1e6:.1f}MB '
                          f'(thr={_sparse_thr:.0e}, col_blk={_col_blk})', flush=True)
                else:
                    ns_mat = ns_mat.toarray() if _sp.issparse(ns_mat) else ns_mat
                _n_free = ns_mat.shape[1]
                _total_rows = sum(_cm.shape[0] for _cm in cons_asr)

                def _ns_apply(ns, B, eps):
                    """Apply null-space projection: ns := null(B @ ns) basis.

                    Sparse-aware. ns can be CSC or dense; B is dense (small after vstack).
                    For wide B (m<n) — common case — uses column-blocked projection
                    to avoid materializing full (n_orig, n_orig) dense intermediate.
                    """
                    m_b, n_b = B.shape
                    _is_sp = _sp.issparse(ns)
                    if m_b >= n_b:
                        # Tall B: explicit null-space basis path
                        ns_tmp = null_space_dense(B, eps)
                        if ns_tmp.size > 0:
                            ns = ns.dot(ns_tmp)  # sparse @ dense -> dense (small r),
                                                 # or dense @ dense
                            # If was sparse and became dense, sparsify back
                            if _is_sp and not _sp.issparse(ns):
                                ns = _csc(ns)
                    else:
                        # Wide B: project ns onto orthogonal complement of row-space
                        _, s, Vh = _np.linalg.svd(B, full_matrices=False)
                        tol = eps * float(s[0]) if len(s) > 0 else eps
                        r = int(_np.sum(s > tol))
                        if r > 0:
                            Vh_r = Vh[:r, :].astype(_np.float64, copy=False)  # (r, n)
                            if _is_sp:
                                # ns: (n, k) sparse CSC
                                # proj_coef = ns @ Vh_r.T : (n, r) dense, r is small
                                # ns_new[:, c0:c1] = ns[:, c0:c1] - proj_coef @ Vh_r[:, c0:c1]
                                # Process in column blocks to limit dense peak memory.
                                k_cols = ns.shape[1]
                                # proj_coef: (n_rows, r); Vh_r needs k_cols-sized columns
                                # but Vh_r.shape[1] == n_b == k_cols (since B is m_b x n_b
                                # and B = cons @ ns, so n_b = ns.shape[1])
                                # Important: Vh_r is over the *current* ns columns
                                proj_coef = ns.dot(Vh_r.T)  # (n_rows, r) dense
                                # Now ns_full = ns - proj_coef @ Vh_r, but we never materialize
                                # the full dense; we build sparse output column-blockwise.
                                n_rows = ns.shape[0]
                                ns_csc = ns  # alias
                                # Column-blocked projection assembled via sp.hstack.
                                # scipy auto-promotes indices to int64 when nnz > 2**31,
                                # so no manual offset arithmetic (fixes int32 overflow).
                                blocks_csc = []
                                for c0 in range(0, k_cols, _col_blk):
                                    c1 = min(c0 + _col_blk, k_cols)
                                    blk_dense = (ns_csc[:, c0:c1].toarray()
                                                 - proj_coef @ Vh_r[:, c0:c1])
                                    if _sparse_thr > 0:
                                        blk_dense[_np.abs(blk_dense) < _sparse_thr] = 0.0
                                    cn = _np.linalg.norm(blk_dense, axis=0)
                                    keep_mask = cn > eps
                                    if keep_mask.any():
                                        blocks_csc.append(_csc(blk_dense[:, keep_mask]))
                                    del blk_dense
                                if blocks_csc:
                                    ns = _sp.hstack(blocks_csc, format='csc')
                                else:
                                    ns = _csc((n_rows, 0))
                            else:
                                # Original dense path (small ns case)
                                ns = ns - (ns @ Vh_r.T) @ Vh_r
                                if _sparse_thr > 0:
                                    ns[_np.abs(ns) < _sparse_thr] = 0.0
                                col_norms = _np.linalg.norm(ns, axis=0)
                                ns = ns[:, col_norms > eps]
                    return ns

                def _eliminate_row(ns, b_row, eps):
                    """Eliminate ONE constraint row from ns via column pivoting.

                    Drops the pivot column and zeroes the constraint on the
                    remaining basis, so b_row @ ns_new == 0 (up to eps). Keeps ns
                    sparse CSC. This is the dimension-reducing replacement for the
                    wide-B projection: the wide-B path only projected (kept all
                    columns, densified), which for n_free >> m_b left a rank-deficient
                    over-complete basis and blew up nnz / wall time.
                    """
                    n_free = ns.shape[1]
                    if n_free <= 1:
                        return ns
                    b = _np.asarray(b_row.toarray()).ravel()  # (n_free,) dense
                    nz = _np.nonzero(b)[0]
                    if nz.size == 0:
                        return ns  # constraint already satisfied
                    # partial pivoting: largest |b| as pivot for stability
                    pidx = int(nz[_np.argmax(_np.abs(b[nz]))])
                    pv = b[pidx]
                    # Conservative skip heuristic: |pivot| < eps treats the row
                    # as already satisfied.  eps is the CLI --eps rank tolerance
                    # (an absolute, not scale-invariant, bound); the absolute
                    # pivot is adequate in practice -- the B3 post-loop max|C@ns|
                    # check measured ~1e-15 on C60Mg2 (n_free=42031), i.e. the
                    # skipped rows are genuinely dependent, not merely < eps.
                    if abs(pv) < eps:
                        return ns  # numerically satisfied
                    keep = _np.arange(n_free)
                    keep = keep[keep != pidx]  # (n_free-1,)
                    coefs = -b[keep] / pv      # (n_free-1,)
                    # elimination matrix E: (n_free, n_free-1)
                    #   E[keep[j], j] = 1   (identity on kept rows)
                    #   E[pidx, j]    = coefs[j] (pivot-row combination)
                    _n_keep = keep.size
                    rows = _np.concatenate([keep, _np.full(_n_keep, pidx)])
                    cols = _np.concatenate([_np.arange(_n_keep), _np.arange(_n_keep)])
                    data = _np.concatenate([_np.ones(_n_keep), coefs])
                    E = _sp.coo_matrix((data, (rows, cols)),
                                       shape=(n_free, n_free - 1)).tocsr()
                    ns_new = (ns @ E).tocsc()
                    if _sparse_thr > 0:
                        _d = ns_new.data
                        _d[_np.abs(_d) < _sparse_thr] = 0.0
                        ns_new.eliminate_zeros()
                    return ns_new

                # [B1] elimination requires the sparse path: _eliminate_row
                # assumes sparse (b_row.toarray(), (ns @ E).tocsc()).  With
                # PHEASY_ASR_SPARSE=0 ns_mat is a dense ndarray and those calls
                # raise AttributeError.  Gate it so the two switches cannot combine.
                _use_eliminate = (_os.environ.get("PHEASY_ASR_ELIMINATE", "1") == "1"
                                  and _use_sparse)
                if _use_eliminate:
                    # Sparse column elimination: dimension-reducing, sparse-preserving.
                    # Processes each constraint row independently; dependent rows
                    # (already satisfied after earlier eliminations) become zero and
                    # drop no column, so the final column count == n_free - rank(cons).
                    print(f'[ASR] elimination mode (n_free={_n_free}, '
                          f'{_total_rows} constraint rows)', flush=True)
                    import time as _time
                    _t0 = _time.time()
                    _eliminated = 0
                    _skipped = 0
                    for _ci, _cm in enumerate(cons_asr):
                        for _ri in range(_cm.shape[0]):
                            _row = _cm.getrow(_ri)
                            _b = _row.dot(ns_mat)
                            _n_before = ns_mat.shape[1]
                            ns_mat = _eliminate_row(ns_mat, _b, self._eps)
                            if ns_mat.shape[1] == _n_before - 1:
                                _eliminated += 1
                            elif ns_mat.shape[1] == _n_before:
                                _skipped += 1
                            if ns_mat.shape[1] == 0:
                                break
                        if ns_mat.shape[1] > 0:
                            ns_mat = normalize(ns_mat, axis=0)
                        if _ci % 10 == 0 or _ci == len(cons_asr) - 1:
                            _nnz = ns_mat.nnz if _sp.issparse(ns_mat) else 'dense'
                            print(f'[ASR]   cons {_ci+1}/{len(cons_asr)}: '
                                  f'n_free={ns_mat.shape[1]}, nnz={_nnz}, '
                                  f't={_time.time()-_t0:.1f}s', flush=True)
                        if ns_mat.shape[1] == 0:
                            break
                    # [B3] post-elimination verification. max|C@ns| is the
                    # acceptance criterion (should be ~1e-10); eliminated+skipped
                    # must equal total_rows unless the loop early-broke at 0 cols.
                    _res = 0.0
                    if ns_mat.shape[1] > 0:
                        for _cm in cons_asr:
                            if _cm.shape[0] > 0:
                                _res = max(_res, float(abs(_cm.dot(ns_mat)).max()))
                    print(f'[ASR] done: n_free={ns_mat.shape[1]}, '
                          f'eliminated={_eliminated}, skipped={_skipped}, '
                          f'max|C@ns|={_res:.2e}', flush=True)
                    if _eliminated + _skipped != _total_rows:
                        print(f'[ASR] WARNING: eliminated+skipped='
                              f'{_eliminated + _skipped} != total_rows={_total_rows} '
                              f'(early break?)', flush=True)
                elif _use_combined and len(cons_asr) > 1:
                    # Adaptive chunked SVD: split constraints so each chunk's
                    # lwork ~ 2 * chunk_rows * n_free stays under INT32_MAX.
                    # PHEASY_ASR_LWORK_LIMIT: override (default INT32_MAX)
                    _lwork_limit = int(float(_os.environ.get(
                        "PHEASY_ASR_LWORK_LIMIT", str(_INT32_MAX))))
                    _max_rows = max(1, _lwork_limit // (2 * max(_n_free, 1)))
                    _chunks = []
                    _cur_idx, _cur_rows = [], 0
                    for _idx, _cm in enumerate(cons_asr):
                        _r = _cm.shape[0]
                        if _cur_idx and _cur_rows + _r > _max_rows:
                            _chunks.append(_cur_idx)
                            _cur_idx, _cur_rows = [], 0
                        _cur_idx.append(_idx)
                        _cur_rows += _r
                    if _cur_idx:
                        _chunks.append(_cur_idx)
                    print(f'[ASR] chunked: {len(cons_asr)} cons -> {len(_chunks)} chunks, '
                          f'max_rows/chunk={_max_rows}, init_n_free={_n_free}, '
                          f'total_rows={_total_rows}', flush=True)

                    import time as _time
                    for _ck_i, _ck in enumerate(_chunks):
                        if ns_mat.shape[1] == 0:
                            break
                        _t0 = _time.time()
                        # cons_asr[i] sparse, ns_mat sparse or dense
                        # Build _stacked as dense (small: ~rows x n_free, ~6GB peak)
                        _parts = []
                        for _i in _ck:
                            _prod = cons_asr[_i].dot(ns_mat)
                            if _sp.issparse(_prod):
                                _prod = _prod.toarray()
                            _parts.append(_prod)
                        _stacked = _np.vstack(_parts)
                        del _parts
                        _t_vs = _time.time() - _t0
                        _ns_old = ns_mat.shape[1]
                        _t0 = _time.time()
                        ns_mat = _ns_apply(ns_mat, _stacked, self._eps)
                        _t_ap = _time.time() - _t0
                        # normalize: sklearn supports sparse (CSC/CSR) and dense
                        _has_data = (ns_mat.shape[1] > 0
                                     if _sp.issparse(ns_mat)
                                     else ns_mat.size > 0)
                        if _has_data:
                            ns_mat = normalize(ns_mat, axis=0)
                        _nnz_str = (f'nnz={ns_mat.nnz}'
                                    if _sp.issparse(ns_mat) else 'dense')
                        print(f'[ASR]   chunk {_ck_i+1}/{len(_chunks)}: '
                              f'rows={_stacked.shape[0]}, '
                              f'n_free {_ns_old}->{ns_mat.shape[1]}, '
                              f'vstack={_t_vs:.1f}s, apply={_t_ap:.1f}s, {_nnz_str}',
                              flush=True)
                        del _stacked
                else:
                    # COMBINED=0 or single constraint: per-constraint loop
                    print(f'[ASR] per-constraint ({len(cons_asr)} iters)', flush=True)
                    for _cm in cons_asr:
                        _B = _cm.dot(ns_mat)
                        if _sp.issparse(_B):
                            _B = _B.toarray()
                        ns_mat = _ns_apply(ns_mat, _B, self._eps)
                        _has = (ns_mat.shape[1] > 0
                                if _sp.issparse(ns_mat) else ns_mat.size > 0)
                        if _has:
                            ns_mat = normalize(ns_mat, axis=0)
                ns_mat = spmat.coo_matrix(ns_mat)
            if order == 2 and self._rasr is not None:
                ns_mat = ns_mat.toarray()
                for cons_mat in cons_rasr:
                    ns_mat_tmp = null_space_dense(cons_mat.dot(ns_mat), self._eps)
                    ns_mat = ns_mat.dot(ns_mat_tmp)
                    ns_mat = normalize(ns_mat, axis=0)
                ns_mat = spmat.coo_matrix(ns_mat)
            # [PATCH ns-rank] 丢弃数值上线性相关的列。
            # 实测 ns_harm(1296x608) 有 8 个近零奇异值 (4 个 ~2.5e-10 + 4 个 ~1e-15),
            # 与 0.63 之间隔着 2e9 倍谱隙 —— 608 个"自由度"只有 600 个独立。
            # 成因: ASR 投影把落在约束行空间的列压成近零后, normalize(axis=0)
            # 又把它们归一化成单位范数, 数值噪声被放大成"真列"。
            # 后果: X = SM_prime @ NS 的条件数由 ~3e2 恶化到 ~1e7, 并继承同维零空间。
            # 开关: PHEASY_NS_RANK_TOL (默认 1e-8, 设 0 关闭)
            #       PHEASY_NS_RANK_MAXCOL (默认 5000, 超过则跳过以免稠密 SVD 爆内存)
            import os as _os_nsr
            _rank_tol = float(_os_nsr.environ.get("PHEASY_NS_RANK_TOL", "1e-8"))
            _rank_maxcol = int(_os_nsr.environ.get("PHEASY_NS_RANK_MAXCOL", "5000"))
            if _rank_tol > 0 and 1 < ns_mat.shape[1] <= _rank_maxcol:
                _d = ns_mat.toarray() if spmat.issparse(ns_mat) else np.asarray(ns_mat)
                _u, _sv, _vh = np.linalg.svd(_d, full_matrices=False)
                _keep = int(np.sum(_sv > _sv[0] * _rank_tol))
                if _keep < _d.shape[1]:
                    logger.info(
                        "- [ns-rank] order {}: dropped {} numerically dependent "
                        "column(s), {} -> {}; sigma_min/sigma_max {:.3e} -> {:.3e}".format(
                            order, _d.shape[1] - _keep, _d.shape[1], _keep,
                            _sv[-1] / _sv[0], _sv[_keep - 1] / _sv[0]))
                    ns_mat = spmat.coo_matrix(_d.dot(_vh[:_keep].T))
                else:
                    logger.info(
                        "- [ns-rank] order {}: full column rank ({}), "
                        "sigma_min/sigma_max = {:.3e}".format(
                            order, _d.shape[1], _sv[-1] / _sv[0]))
            elif _rank_tol > 0:
                logger.info(
                    "- [ns-rank] order {}: skipped (n_free={} > MAXCOL={})".format(
                        order, ns_mat.shape[1], _rank_maxcol))

            ifc_free_order[order] = ns_mat.shape[1]
            null_space_list.append(ns_mat)

            end_time = datetime.datetime.now()
            total_time = end_time - start_time
            logger.info("- Time cost: {}.".format(total_time))

        if self._crys_basis:
            clus_num = cluster_space.get_number_of_clusters_each_order()
            for order in range(2, max_order + 1):
                rot_basis = kron_product_einsum(
                    normalize(self._scell.cell.T, axis=0), order
                )
                shape = [3] * order
                null_space_list[order - 2] = null_space_list[order - 2].toarray()
                ns_tensor = null_space_list[order - 2].T.reshape(
                    (-1, clus_num[order], *shape)
                )
                path1 = list(range(order * 2))
                path2 = list(range(2 * order, 2 * order + 2)) + list(
                    range(order, order * 2)
                )
                path3 = list(range(2 * order, 2 * order + 2)) + list(range(order))
                ns_tensor_rot = np.einsum(rot_basis, path1, ns_tensor, path2, path3)
                null_space_list[order - 2] = ns_tensor_rot.reshape(
                    (-1, clus_num[order] * 3 ** order)
                ).T
                null_space_list[order - 2] = spmat.coo_matrix(
                    null_space_list[order - 2]
                )
        ns_mat_full = spmat.block_diag(null_space_list)
        self._constraints = cons_mat_dict
        self._ifc_free_clus = ifc_free_clus
        self._ifc_free_order = ifc_free_order
        self._ifc_free_tot = ns_mat_full.shape[1]

        logger.info("Summary of IFCs:")
        for order in range(2, max_order + 1):
            if order == 2:
                logger.info(
                    "- HARM    | IFC number: {:<5d} | Free IFC number: {}".format(
                        *null_space_list[order - 2].shape
                    )
                )
                spmat.save_npz("ns_harm", null_space_list[order - 2])
            else:
                logger.info(
                    "- ANHARM{} | IFC number: {:<5d} | Free IFC number: {}".format(
                        order, *null_space_list[order - 2].shape
                    )
                )
                spmat.save_npz("ns_anharm{}".format(order), null_space_list[order - 2])
        logger.info("Total number of free IFCs: {}".format(self._ifc_free_tot))

        return ns_mat_full

    def build_translational_invariance(self, clusters, order):
        """Construct symmetry constraints from translational invariance.

        Parameters
        ----------
        clusters : dict
            Cluster space where translational invariance is applied.
        order : int
            The order of clusters or IFCs.

        Returns:
        -------
        list(scipy.sparse.csr_matrix)
            A list of translational invariance constraints in CSR
            sparse matrix form.

        """
        if self._crys_basis:
            rot_fmt = "rotmat"
        else:
            rot_fmt = "crotmat"
        ifc_num = np.power(3, order)
        cons_list = deque()

        if order == 2:
            asr_set = list(self._scell.get_symmetry_distinct_atoms()[:, np.newaxis])
        else:
            asr_set = deque()
            for orbit in clusters[order - 1]:
                asr_set.append(orbit[0].atom_index)

        for asr_idx in asr_set:
            cons_mat_tmp = deque()
            for orbit in clusters[order]:
                block = spmat.coo_matrix((ifc_num, ifc_num))
                for cluster in orbit[1:]:
                    permuted_index = list(asr_idx)
                    diffs = _diff_cluster(permuted_index, cluster.atom_index)
                    if diffs[0]:
                        permuted_index.append(diffs[1])
                        Rmat = get_permutation_tensor(
                            cluster.atom_index, permuted_index
                        ).reshape((ifc_num, ifc_num))
                        Gamma = kron_product(getattr(cluster, rot_fmt), order)
                        block += spmat.coo_matrix(Rmat).dot(Gamma)
                cons_mat_tmp.append(block)
            cons_mat = spmat.hstack(cons_mat_tmp)
            cons_list.append(cons_mat)

        return cons_list

    def build_rotational_invariance(self, clusters, ifc_lr=None):
        """Construct symmetry constraints of general invariance conditions.

        These are enforced on harmonic force constants only, including
        Born-Huang rotational invariance and Huang conditions for vanishing
        external stress fields. (See npj Comput Mater 8, 236 (2022)).
        Only work in Cartesian coordinates.

        Parameters
        ----------
        clusters : list
            Cluster space for harmonic force constants.
        ifc_lr : numpy.ndarray
            The long-range part of interatomic force constants. It should be in
            the same shape (natom,natoms,3,3), where natom is the number of
            atoms in unit cell and natoms is the number of atoms in supercell.

        Returns:
        -------
        list(scipy.sparse.csr_matrix)
            A list of constraints for rotational invariance and vanishing
            stress condition in CSR sparse matrix form.

        """
        nclus = len(clusters)
        ndim = np.prod(self._scell.supercell)
        natoms = self._scell.get_global_number_of_atoms()
        cons_list = deque()
        nac_sum_list = deque()

        if self._rasr in ["BH", "BHH"]:
            """Born-Huang rotational invariance"""
            for iat in range(natoms):
                cons_mat = np.zeros((9, nclus * 9))
                nac_sum = np.zeros(9)
                if iat % ndim == 0:
                    for ind1 in range(3):
                        for i, asr_idx in enumerate(
                            itertools.combinations(range(3), 2)
                        ):
                            ind2, ind3 = asr_idx
                            fc_ind1 = ind1 * 3 + ind2
                            fc_ind2 = ind1 * 3 + ind3
                            for j, orbit in enumerate(clusters):
                                block = np.zeros(9)
                                for cluster in orbit[1:]:
                                    diffs = _diff_cluster([iat], cluster.atom_index)
                                    if diffs[0]:
                                        jat = diffs[1]
                                        at_idx = cluster.s2u[
                                            cluster.atom_index.index(iat), 1
                                        ]
                                        trans = self._scell.ws_offsets[at_idx][jat]
                                        ri = self._scell.scaled_positions[iat]
                                        rj = self._scell.scaled_positions[jat]
                                        rij = rj - ri + trans
                                        rr = rij.sum(axis=0) / trans.shape[0]
                                        rr = self._scell.cell.real.T.dot(rr)
                                        Rmat = get_permutation_tensor(
                                            cluster.atom_index, [iat, jat]
                                        ).reshape((9, 9))
                                        Gamma = Rmat.dot(
                                            cluster.get_crotation_tensor().toarray()
                                        )
                                        block += (
                                            Gamma[fc_ind1] * rr[ind3]
                                            - Gamma[fc_ind2] * rr[ind2]
                                        )
                                        if self._nac != 0 and self._do_rasr == "PP":
                                            part = ifc_lr[iat // ndim, jat].flatten()
                                            nac_sum[i] += (
                                                part[fc_ind2] * rr[ind2]
                                                - part[fc_ind1] * rr[ind3]
                                            )
                                cons_mat[ind1 * 3 + i, j * 9 : (j + 1) * 9] = block
                cons_list.append(cons_mat)
                nac_sum_list.append(nac_sum)

        np.save("rasr.npy", cons_mat)

        if self._rasr in ["H", "BHH"]:
            """Huang conditions for vanishing external stress"""
            huang_set = deque()
            for _, item in enumerate(
                itertools.combinations_with_replacement(range(3), 2)
            ):
                huang_set.append(item)

            cons_mat = np.zeros((15, nclus * 9))
            nac_sum = np.zeros(15)
            for i, asr_idx in enumerate(itertools.combinations(huang_set, 2)):
                ind1, ind2 = asr_idx[0]
                ind3, ind4 = asr_idx[1]
                fc_ind1 = ind1 * 3 + ind2
                fc_ind2 = ind3 * 3 + ind4
                for j, orbit in enumerate(clusters):
                    if orbit[0].is_proper:  # skip vanishing atomic distance
                        block = np.zeros(9)
                        for cluster in orbit[1:]:
                            iat, jat = cluster.atom_index
                            if iat % ndim == 0 or jat % ndim == 0:
                                ri = self._scell.scaled_positions[iat]
                                rj = self._scell.scaled_positions[jat]
                                if iat % ndim == 0:
                                    at_idx = cluster.s2u[0, 1]
                                    trans = self._scell.ws_offsets[at_idx][jat]
                                    rij = rj - ri + trans
                                    rij = self._scell.cell.real.T.dot(rij.T).T
                                    for iw in range(trans.shape[0]):
                                        Gamma = (
                                            cluster.get_crotation_tensor().toarray()
                                            / trans.shape[0]
                                        )
                                        block += (
                                            Gamma[fc_ind1]
                                            * rij[iw, ind3]
                                            * rij[iw, ind4]
                                        )
                                        block -= (
                                            Gamma[fc_ind2]
                                            * rij[iw, ind1]
                                            * rij[iw, ind2]
                                        )
                                        if self._nac != 0 and self._do_rasr == "PP":
                                            part = (
                                                ifc_lr[iat // ndim, jat].flatten()
                                                / trans.shape[0]
                                            )
                                            nac_sum[i] += (
                                                part[fc_ind2]
                                                * rij[iw, ind1]
                                                * rij[iw, ind2]
                                            )
                                            nac_sum[i] -= (
                                                part[fc_ind1]
                                                * rij[iw, ind3]
                                                * rij[iw, ind4]
                                            )
                                if jat % ndim == 0:
                                    at_idx = cluster.s2u[1, 1]
                                    trans = self._scell.ws_offsets[at_idx][iat]
                                    rij = ri - rj + trans
                                    rij = self._scell.cell.real.T.dot(rij.T).T
                                    for iw in range(trans.shape[0]):
                                        Rmat = get_permutation_tensor(
                                            cluster.atom_index, [jat, iat]
                                        ).reshape((9, 9))
                                        Gamma = (
                                            Rmat.dot(
                                                cluster.get_crotation_tensor().toarray()
                                            )
                                            / trans.shape[0]
                                        )
                                        block += (
                                            Gamma[fc_ind1]
                                            * rij[iw, ind3]
                                            * rij[iw, ind4]
                                        )
                                        block -= (
                                            Gamma[fc_ind2]
                                            * rij[iw, ind1]
                                            * rij[iw, ind2]
                                        )
                                        if self._nac != 0 and self._do_rasr == "PP":
                                            part = (
                                                ifc_lr[jat // ndim, iat].flatten()
                                                / trans.shape[0]
                                            )
                                            nac_sum[i] += (
                                                part[fc_ind2]
                                                * rij[iw, ind1]
                                                * rij[iw, ind2]
                                            )
                                            nac_sum[i] -= (
                                                part[fc_ind1]
                                                * rij[iw, ind3]
                                                * rij[iw, ind4]
                                            )
                        cons_mat[i, j * 9 : (j + 1) * 9] = block
            cons_list.append(cons_mat)
            nac_sum_list.append(nac_sum)

        if self._nac != 0 and self._do_rasr == "PP":
            return cons_list, np.hstack(nac_sum_list)
        else:
            return cons_list

    def build_high_order_rotational_invariance(self):
        # TODO
        pass

    @staticmethod
    def construct_null_space_restart(max_order):
        """Construct full null space matrix from a restart calculation.

        Through this restart method, it will build full null space matrix
        from the saved npz files of each order IFCs.

        Parameters
        ----------
        max_order : int
            Highest order considered in cluster expansion.

        Returns:
        -------
        scipy.sparse.coo_matrix
            Full null space in COO sparse matrix form. It consists
            of null space for each order IFCs.
    
        """
        null_space_list = []
        for order in range(2, max_order + 1):
            if order == 2:
                null_space_list.append(spmat.load_npz("ns_harm.npz"))
            else:
                null_space_list.append(spmat.load_npz("ns_anharm{}.npz".format(order)))
        ns_mat_full = spmat.block_diag(null_space_list)

        logger.info("Summary of IFCs:")
        for order in range(2, max_order + 1):
            if order == 2:
                logger.info(
                    "- HARM    | IFC number: {:<5d} | Free IFC number: {}".format(
                        *null_space_list[order - 2].shape
                    )
                )
            else:
                logger.info(
                    "- ANHARM{} | IFC number: {:<5d} | Free IFC number: {}".format(
                        order, *null_space_list[order - 2].shape
                    )
                )
        logger.info("Total number of free IFCs: {}".format(ns_mat_full.shape[1]))

        return ns_mat_full

    def write(self, filename="constraints.pkl"):
        """Write SymmetryConstraints object into pickle file.

        Parameters
        ----------
        filename : str
            Filename to save symmetry constraints.
        
        """
        with open(filename, "wb") as fd:
            pickle.dump(self, fd)

    @staticmethod
    def read(filename="constraints.pkl"):
        """Read and create SymmetryConstraints object from pickle file.

        Parameters
        ----------
        filename : str
            Filename for SymmetryConstraints object to read from.

        Returns
        -------
        SymmetryConstraints object
            
        """
        with open(filename, "rb") as fd:
            return pickle.load(fd)

    def __repr__(self):
        """Return informaion of Symmetry constraints."""
        txt = "SymmetryConstraints(symmetry basis: {!r}, ifc_free: {!r}, ifc_free_tot: {!r})"
        if self._crys_basis:
            sym_bs = "crystal"
        else:
            sym_bs = "Cartesian"
        return txt.format(sym_bs, self._ifc_free, self._ifc_free_tot)


def _is_asr_unique(item, asr_set, clusters):
    """Check if the input ASR is unique or already applied.

    This function can only be appled to translational invariance
    for anharmonic force constants.

    Parameters
    ----------
        item : list or tuple
            ASR subscript.
        asr_set : set(tuple)
            A set of already applied ASR subscripts.
        clusters : list(list(Cluster))
            Cluster space of the N-1 order, where N is the 
            order of anharmonic force constants.

    Returns:
    -------
    bool
        True means the current ASR 'item' is unique, while
        False means this ASR can be equivalent to previous 
        ones in 'asr_set' through space group symmetry.
    
    """
    if tuple(item) in asr_set:
        return False
    else:
        for orbit in clusters:
            if len(set(orbit[0].atom_index)) == len(set(item)):
                orbit_set = set(map(lambda x: tuple(sorted(x.atom_index)), orbit[1:]))
                if tuple(item) in orbit_set:
                    asr_set.update(orbit_set)
                    return True


def _diff_cluster(cluster_a, cluster_b):
    """Return the different atom between cluster a and b.

    This function checks if cluster a and b only differ
    by one atom when their orders differ by one.

    Parameters
    ----------
    cluster_a : list or numpy.ndarray
        Atom index in cluster a with length of N-1.
    cluster_b : list or numpy.ndarray
        Atom index in cluster b with length of N.

    Returns:
    -------
    bool
        True for the case that cluster a and b only differ
        by one atom, False otherwise.
    int
       The atom index for the different atom. This is only
       returned when the bool value is True.

    """
    diff = Counter(cluster_b) - Counter(cluster_a)
    diff = list(diff.elements())
    if len(diff) == 1:
        return (True, diff[0])
    else:
        return (False,)
