"""Scripts for running Pheasy"""
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2023 Changpeng Lin
# All rights reserved.

import os
import datetime
from collections import deque

import numpy as np
import scipy.sparse as spmat
import pickle
import pheasy.structure.io as io
from pheasy.version import get_logo_version
from pheasy.basic_io import InputParser, logger
from pheasy.structure.atoms import NeighborList, create_supercell
from pheasy.structure.symmetry import get_spacegroup, get_symmetry
from pheasy.structure.force_constants import ForceConstants
from pheasy.core.cluster_orbit import CSGenerator, ClusterSpace
from pheasy.core.symmetry_constraints import SymmetryConstraints
from pheasy.core.utilities import get_exclude_set
from pheasy.core.optimizer import Optimizer
from pheasy.core.displacements import (
    move_atoms_simple,
    generate_displacements_from_file,
    generate_displacements_from_aimd,
    build_sensing_matrix,
)
from joblib import Parallel, delayed

# ===================================================================
# [PATCH] sparse-in-worker wrapper
# 让并行 worker 直接返回 CSR f32, 避免主进程累积大量 dense f64 (~619 GB).
# 拟合结果完全等价于原始路径; 下游代码无需修改.
# ===================================================================
def _build_sensing_matrix_sparse(CS_full, u):
    """Worker-side: build dense sensing block then immediately convert
    to CSR float32 inside the worker, so the dense f64 block (1.3 GB)
    is freed before pickling back to the main process."""
    import numpy as _np
    import scipy.sparse as _sp
    _mat = build_sensing_matrix(CS_full, u)
    if _sp.issparse(_mat):
        return _mat.tocsr().astype(_np.float32, copy=False)
    return _sp.csr_matrix(_mat.astype(_np.float32, copy=False))

from pheasy.core.forces import read_interatomic_forces, read_interatomic_forces_aimd


class WorkFlow(object):
    """Class defining a complete workflow of calculations.

    The workflow consists of mandatory and optional tasks.
    Each task has a method that runs the calculation. To 
    add a new calculation, the corresponding run method 
    should be implemented here.
    manatory :
        welcome
        run_cell
        run_neighborlist_cutoff
    optional :
        run_cluster_expansion
        run_symmetry_constraints
        run_sensing_matrix
        run_fit_force_constants

    """

    def __init__(self, parser):
        """Initialize with an input parser and set default filenames.

        Parameters
        ----------
        parser : InputParser
            An instance of class InputParser.

        """
        self.settings = parser.settings

        self.NeighborListFile = "neighbor_list.pkl"
        self.ClusterSpaceFile = "cs.pkl"
        self.ConstraintsFile = "constraints.pkl"
        self.SensingMatrixFile = "sm_prime.npz"
        self.ForceArrayFile = "fm1d.npz"
        self.ForceMatrixFile = "fm2d.npz"
        self.ForceConstantArrayFile = "phi.npz"
        self.AForceConstantArrayFile = "phi_anharm.npz"

    def welcome(self):
        """Print welcome information.
        """
        logo_list = get_logo_version()
        logo = logo_list[np.random.randint(len(logo_list))]
        logger.info("Start Pheasy." + logo)

    def run_cell(self):
        """Parse primitive cell and create supercell.

        Returns:
        -------
        pcell : pheasy.Atoms
            Primitive unit cell structure.
        scell : pheasy.Atoms
            Supercell structure.

        """
        settings = self.settings

        """Read primitive unit cell and check settings"""
        pcell = io.read_cell(settings.PCELL_FILENAME, settings=settings)
        natom = pcell.get_global_number_of_atoms()
        InputParser.check_args(settings, natom)
        ndim = np.prod(settings.DIM)
        natoms = ndim * natom

        """Set magmom in case of magnetic materials."""
        if settings.IS_MAGNETIC:
            if len(settings.MAGMOM) == 3 * natom:  # collinear case
                settings.MAGMOM = np.reshape(settings.MAGMOM, (natom, 3))
            else:
                settings.MAGMOM = np.array(settings.MAGMOM)
            self.settings = settings
            pcell.set_initial_magnetic_moments(settings.MAGMOM)

        """Print crystal system information."""
        space_group = get_spacegroup(pcell, symprec=settings.SYMPREC)
        pcell.symops = get_symmetry(
            pcell, symprec=settings.SYMPREC, is_magnetic=settings.IS_MAGNETIC
        )
        io.write_symops(pcell.symops, "pcell.symops")
        nsym = pcell.get_number_of_symmetries()
        logger.info("System: %s" % pcell.get_chemical_formula())
        logger.info(
            "Space group: %s, %i symmetry operations found." % (space_group, nsym)
        )

        """Create supercell or read supercell from file.
           In case of reading supercell from file, check if DIM is
           consistent with total numbe of atoms in the supercell."""
        if settings.READ_SCELL:  # read supercell
            logger.info(
                "Read {0[0]:d} x {0[1]:d} x {0[2]:d} supercell from file ({1:d} atoms).".format(
                    settings.DIM, natoms
                )
            )
            scell = io.read_cell(
                settings.SCELL_FILENAME, settings=settings, supercell=True
            )
            if (ndim * natom) != scell.get_global_number_of_atoms():
                logger.error(
                    "Number of atoms in supercell inconsistent with argument DIM."
                )
                raise ValueError
            scell.set_supercell(settings.DIM)
            if settings.IS_MAGNEIC:  # in case of magnetic materials
                if len(settings.MAGMOM.shape) == 1:
                    smagmom = np.repeat(settings.MAGMOM, ndim)
                else:
                    smagmom = np.repeat(settings.MAGMOM, ndim, axis=0)
                scell.set_initial_magnetic_moments(smagmom)
        else:  # create supercell
            logger.info(
                "Creating {0[0]:d} x {0[1]:d} x {0[2]:d} supercell ({1:d} atoms).".format(
                    settings.DIM, natoms
                )
            )
            scell = create_supercell(pcell, settings.DIM, settings.IS_MAGNETIC)
            io.write_cell(scell, settings=settings)
        if settings.QE:
            if os.path.isfile(settings.PW_HEADER_FILE):
                scell.read_pw_header(settings.PW_HEADER_FILE)

        """Set dielectric tensor and Born effective charges."""
        if settings.NAC != 0:
            born_info = io.read_dielectrics(settings.BORN_FILE)
            if np.shape(born_info[1])[0] != natom:
                logger.error("Shape of Born effective charge tensor is wrong.")
                raise ValueError
            pcell.set_dielectrics(born_info[0], born_info[1])
            scell.set_dielectrics(born_info[0], born_info[1])

        """Set mapping between primitive and supercell atomic indices."""
        scell.set_smap()
        scell.set_pmap()

        """Analyze supercell symmetry."""
        scell.symops = get_symmetry(
            scell, symprec=settings.SYMPREC, is_magnetic=settings.IS_MAGNETIC
        )
        io.write_symops(scell.symops, "scell.symops")

        self.pcell = pcell
        self.scell = scell

    def run_neighborlist_cutoff(self):
        """Analyze neighbor list and cutoffs.

        Generate a neighbor list or read it from pickle file.
        Config cutoff distance for interatomic force constants.

        Returns:
        -------
        nn_list : NeighborList
            An instance of class NeighborList created for 
            supercell structure.
        cutoffs : dict
            A dictionary for cutoffs at different orders.

        """
        settings = self.settings

        if os.path.isfile(self.NeighborListFile):
            nn_list = NeighborList.read(self.NeighborListFile)
            if list(nn_list.supercell) != settings.DIM:
                logger.warning(
                    "System dimension defined in neighbor list file"
                    + " is not consistent with DIM."
                )
                nn_list = NeighborList(
                    self.scell, self.pcell.symops["equivalent_atoms"]
                )
                nn_list.write(self.NeighborListFile)
        else:
            nn_list = NeighborList(self.scell, self.pcell.symops["equivalent_atoms"])
            nn_list.write(self.NeighborListFile)
        self.scell.set_wigner_seitz_offsets(nn_list.ws_offsets)

        cutoffs = {}
        for n in range(2, settings.MAX_ORDER + 1):
            cutoff = getattr(settings, "CUT" + str(n))
            if cutoff is None:
                cutoffs[n] = np.inf
            elif cutoff > 0:
                cutoffs[n] = cutoff
            else:
                nth = int(abs(cutoff))
                cutoffs[n] = nn_list.get_neighbor_cutoff_distance(nth)

        self.cutoffs = cutoffs
        self.nn_list = nn_list

    def run_cluster_expansion(self):
        """Generate cluster-orbit space CS_full."""
        settings = self.settings

        if settings.SPG_CLUS:
            start_time_sub = datetime.datetime.now()

            logger.info(
                "Starting to generate cluster space up to {}-order.".format(
                    settings.MAX_ORDER
                )
            )
            CS_generator = CSGenerator(
                self.nn_list,
                self.scell.symops,
                settings.MAX_ORDER,
                self.cutoffs,
                settings.NBODY,
            )
            CS_full = CS_generator.generate_represent_clusters_with_orbit()
            CS_full.write(self.ClusterSpaceFile)
            end_time_sub = datetime.datetime.now()
            time_cost = end_time_sub - start_time_sub
            logger.info(
                "Cluster space generation finished, time cost: {}.".format(time_cost)
            )
        else:
            """Read cluster space from file and print related information."""
            if os.path.isfile(self.ClusterSpaceFile):
                logger.info(
                    "Reading and generating cluster space from file, "
                    + f"up to {settings.MAX_ORDER}-order."
                )
                CS_full = ClusterSpace.read(self.ClusterSpaceFile)
                CS_full.print_cluster_space_info()

        self.CS_full = CS_full

    def run_symmetry_constraints(self):
        """Apply symmetry constraints and construct null space."""
        settings = self.settings

        if settings.NULL_SPACE:
            start_time_sub = datetime.datetime.now()
            logger.info("Starting to construct symmetry constraints and null space.")
            if settings.CRYS_BASIS:
                logger.info("Symmmetry constraints are imposed in crystal coordinate.")
            else:
                logger.info(
                    "Symmmetry constraints are imposed in Cartesian coordinate."
                )

            symmetry_constraints = SymmetryConstraints(
                self.scell,
                settings.CRYS_BASIS,
                rasr=settings.RASR,
                do_rasr=settings.DO_RASR,
                nac=settings.NAC,
                eps=settings.EPS,
            )
            self.NS_full = symmetry_constraints.impose_symmtery_constaints(self.CS_full)
            if settings.WRITE_SYM_CONS:
                symmetry_constraints.write(self.ConstraintsFile)

            end_time_sub = datetime.datetime.now()
            time_cost = end_time_sub - start_time_sub
            logger.info(
                "Construction of symmetry constraints finished, time cost: {}.".format(
                    time_cost
                )
            )
        else:
            if settings.FIT_IFC or settings.MODE.upper() == "PP":
                logger.info(
                    "Reconstructing null space of symmetry constraints from file."
                )
                self.NS_full = SymmetryConstraints.construct_null_space_restart(
                    settings.MAX_ORDER
                )

    def run_sensing_matrix(self):
        """Create displaced configurations and construct sensing matrix."""
        settings = self.settings

        if settings.SENSING_MAT:
            start_time_sub = datetime.datetime.now()
            logger.info("Starting to construct sensing (displacement) matrix.")

            if settings.QE:
                file_format = "qe"
                filename_pattern = "DISP.in.{{0:0{0}d}}".format(3)
            else:
                file_format = "vasp"
                filename_pattern = "DISP.POSCAR.{{0:0{0}d}}".format(3)

            sensing_mat_list = deque()

            if settings.MODE.upper() == "RANDOM":
                if settings.DISP_FILE:
                    logger.info("Reading displaced configurations from file.")
                    with open("disp_matrix.pkl", "rb") as file:
                         u_matrix = pickle.load(file)
                    _n_jobs = int(os.environ.get("PHEASY_N_JOBS", "1"))
                    _ndata = min(settings.NDATA, u_matrix.shape[0])
                    _u_list = [u_matrix[n, :, :] for n in range(_ndata)]
                    if _n_jobs != 1:
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
                            sensing_mat_list.append(sensing_mat)
                else:
                    logger.info(
                        "Displacing atoms randomly by {} A.".format(settings.U_VAL)
                    )
                    for n in range(settings.NDATA):
                        filename = filename_pattern.format(n + 1)
                        disp_scell = move_atoms_simple(self.scell, settings.U_VAL)
                        disp_scell.automatic_write(filename, file_format)
                        u_vecs = disp_scell.get_atomic_displacements()
                        logger.info(
                            "- Generating displaced configuration {} of {}.".format(
                                n + 1, settings.NDATA
                            )
                        )
                        sensing_mat = build_sensing_matrix(self.CS_full, u_vecs)
                        sensing_mat_list.append(sensing_mat)

            elif settings.MODE.upper() == "AIMD":
                logger.info("Reading displaced configurations from AIMD trajectories.")
                if settings.NSKIP is not None:
                    logger.info(
                        "- The first {} steps of AIMD are skipped.".format(
                            settings.NSKIP
                        )
                    )
                logger.info(
                    "- Number of sampled structures: {}.".format(settings.NDATA)
                )
                logger.info("- Sampling interval: {}.".format(settings.NSTEP))
                disp_scell_list = generate_displacements_from_aimd(
                    self.scell,
                    settings.NDATA,
                    settings.NSKIP,
                    settings.NSTEP,
                    file_format,
                )
                u_vecs_list = deque(
                    map(lambda x: x.get_atomic_displacements(), disp_scell_list)
                )
                _n_jobs = int(os.environ.get("PHEASY_N_JOBS", "1"))
                if _n_jobs != 1:
                    logger.info(
                        "- Parallel sensing matrix construction: {} workers.".format(_n_jobs)
                    )
                    _results = Parallel(n_jobs=_n_jobs)(
                        delayed(_build_sensing_matrix_sparse)(self.CS_full, _u)
                        for _u in u_vecs_list
                    )
                    sensing_mat_list.extend(_results)
                else:
                    for u_vecs in u_vecs_list:
                        sensing_mat = build_sensing_matrix(self.CS_full, u_vecs)
                        sensing_mat_list.append(sensing_mat)

                F_mean = deque()
                force_list = deque(map(lambda x: x.get_forces(), disp_scell_list))
                for forces in force_list:
                    F_mean.append(forces.mean(axis=0))
                F_mat = np.vstack(force_list)
                np.savez_compressed(
                    self.ForceMatrixFile, F=F_mat, mean=np.array(F_mean)
                )

            # FIX: avoid 577 GiB dense vstack. Stream-convert each dense
            # block (~1.34 GB) to CSR float32 (~0.06 GB), pop as we go to
            # release dense memory, then sparse-vstack at the end.
            import time as _t_sm
            print(f'[SM] streaming dense->sparse f32 conversion of '
                  f'{len(sensing_mat_list)} blocks...', flush=True)
            _t0 = _t_sm.time()
            _sparse_blocks = []
            # [PATCH] worker 已返回 CSR f32, 跳过转换; 仅做格式/精度校验
            while sensing_mat_list:
                _b = sensing_mat_list.popleft()
                if spmat.issparse(_b):
                    if _b.format != 'csr':
                        _b = _b.tocsr()
                    if _b.dtype != np.float32:
                        _b = _b.astype(np.float32)
                else:
                    # 串行 fallback 路径仍可能给 dense
                    _b = spmat.csr_matrix(_b, dtype=np.float32)
                _sparse_blocks.append(_b)
            print(f'[SM] blocks converted in {_t_sm.time()-_t0:.1f}s, vstacking...',
                  flush=True)
            _t0 = _t_sm.time()
            self.SM_prime = spmat.vstack(_sparse_blocks, format='csr')
            del _sparse_blocks
            _mem = (self.SM_prime.data.nbytes
                    + self.SM_prime.indices.nbytes
                    + self.SM_prime.indptr.nbytes) / 1e9
            print(f'[SM] vstack done in {_t_sm.time()-_t0:.1f}s: '
                  f'shape={self.SM_prime.shape}, nnz={self.SM_prime.nnz}, '
                  f'mem={_mem:.2f} GB (f32 sparse CSR)', flush=True)
            spmat.save_npz(self.SensingMatrixFile, self.SM_prime)

            end_time_sub = datetime.datetime.now()
            time_cost = end_time_sub - start_time_sub
            logger.info(
                "Construction of sensing (displacement) matrix finished, time cost: {}.".format(
                    time_cost
                )
            )
        else:
            if settings.FIT_IFC:
                logger.info("Reconstructing sensing (displacement) matrix from file.")
                self.SM_prime = spmat.load_npz(self.SensingMatrixFile)

    def run_fit_force_constants(self):
        """Fit interatomic force constants."""
        settings = self.settings
        natoms = self.scell.get_global_number_of_atoms()
        # AUTO: 供 PheasyRFECV 的 GroupKFold 防泄漏使用 (每配置行数=3×超胞原子数)
        import os as _os_grp
        _os_grp.environ["PHEASY_CV_GROUP_SIZE"] = str(3 * natoms)

        if settings.FIT_IFC:
            start_time_sub = datetime.datetime.now()
            logger.info("Starting to fit interatomic force constants.")

            # Create ForceConstants instance.
            FC_model = ForceConstants(self.scell, self.CS_full)

            # Pre-processing sensing matrix
            # FIX: keep SM_prime as sparse f32 (was 310 GiB dense f32 before).
            # CSR row slicing is O(nnz_of_slice). Downstream uses .dot(dense)
            # which works natively on sparse, producing dense output of size
            # (n_keep, n_free_NS) only.
            _n_keep = 3 * natoms * settings.NDATA
            _sm = self.SM_prime
            if not spmat.issparse(_sm):
                _sm = spmat.csr_matrix(_sm)
            elif not isinstance(_sm, spmat.csr_matrix):
                _sm = _sm.tocsr()
            if _n_keep < _sm.shape[0]:
                SM_prime = _sm[:_n_keep, :]
            else:
                SM_prime = _sm
            if SM_prime.dtype != np.float32:
                SM_prime = SM_prime.astype(np.float32)
            _mem = (SM_prime.data.nbytes
                    + SM_prime.indices.nbytes
                    + SM_prime.indptr.nbytes) / 1e9
            logger.info(
                f'SM_prime kept sparse f32 CSR: shape={SM_prime.shape}, '
                f'nnz={SM_prime.nnz}, mem={_mem:.2f} GB'
            )

            if settings.QE:
                file_format = "qe"
                rforce_file = "rforce.out"
                filename_pattern = "DISP.out.{{0:0{0}d}}".format(3)
            else:
                file_format = "vasp"
                rforce_file = "rforce.xml"
                filename_pattern = "vasprun.xml.{{0:0{0}d}}".format(3)

            if settings.RFORCE:
                logger.info("Residual forces of perfect structure will be removed.")
                rforces = read_interatomic_forces(rforce_file, format=file_format)
            else:
                rforces = np.zeros((natoms, 3))

            if settings.EXCLUDE is not None:
                ex_set = get_exclude_set(settings.EXCLUDE)
                logger.info(
                    "Training samples to be excluded: {}".format(
                        settings.EXCLUDE.replace(" ", "")
                    )
                )
            else:
                ex_set = set()

            if settings.MODE == "RANDOM":
                logger.info(
                    "Reading interatomic forces, {} configurations.".format(
                        settings.NDATA
                    )
                )
                force_list = deque()
                #for n in range(settings.NDATA):
                #    filename = filename_pattern.format(n + 1)
                #    if (n + 1) in ex_set:
                #        continue
                #    forces = read_interatomic_forces(filename, format=file_format)
                #   res = forces.mean(axis=0)
                #  force_list.append(forces - rforces)
                #  logger.info("- {}, average force per atom:".format(filename))
                #  logger.info("\t {} eV / A".format(res))
                with open("force_matrix.pkl", "rb") as file:
                     f_matrix = pickle.load(file)
                     f_matrix_use = []
                     for n in range(settings.NDATA):
                        if n >= f_matrix.shape[0]:
                            break
                        f_matrix_new = f_matrix[n,:,:]
                        f_matrix_use.append(f_matrix_new)
                FM = np.vstack(f_matrix_use).flatten().astype(np.float32)
                if settings.EXCLUDE is not None:
                    sensing_mat_list = deque()
                    for n in range(settings.NDATA):
                        if (n + 1) in ex_set:
                            continue
                        sensing_mat_list.append(
                            SM_prime[3 * natoms * n : 3 * natoms * (n + 1), :]
                        )
                    # FIX: SM_prime is sparse CSR after our patch
                    if spmat.issparse(SM_prime):
                        SM_prime = spmat.vstack(list(sensing_mat_list), format='csr')
                    else:
                        SM_prime = np.vstack(sensing_mat_list)

            elif settings.MODE == "AIMD":
                logger.info(
                    "Reading interatomic forces from AIMD, {} trajectories.".format(
                        settings.NDATA
                    )
                )
                logger.info("- AIMD average force per atom:")
                if os.path.isfile(self.ForceMatrixFile):
                    F_tmp = np.load(self.ForceMatrixFile)
                    F_mat = F_tmp["F"]
                    F_mean = F_tmp["mean"]
                    if F_mean.shape[0] != settings.NDATA:
                        logger.error(
                            "Wrong dimension of AIMD force database, {} trajectories".format(
                                F_mean.shape[0]
                            )
                        )
                    for n in range(F_mean.shape[0]):
                        logger.info("\t {} eV / A".format(F_mean[n]))
                        F_mat[3 * natoms * n : 3 * natoms * (n + 1), :] -= rforces
                    FM = F_mat.flatten()
                else:
                    force_list = read_interatomic_forces_aimd(
                        settings.NDATA, settings.NSKIP, settings.NSTEP, file_format
                    )
                    for n, forces in enumerate(force_list):
                        res = forces.mean(axis=0)
                        force_list[n] = forces - rforces
                        logger.info("\t {} eV / A".format(res))
                    FM = np.vstack(force_list).flatten()

            elif settings.MODE in ["SCAILD", "SPOR"]:
                # SCAILD/SPOR模式：从pickle文件读取力矩阵
                logger.info(
                    "Reading interatomic forces for {} mode, {} configurations.".format(
                        settings.MODE, settings.NDATA
                    )
                )
                if os.path.isfile("force_matrix.pkl"):
                    with open("force_matrix.pkl", "rb") as file:
                        f_matrix = pickle.load(file)
                        f_matrix_use = []
                        for n in range(settings.NDATA):
                            if n >= f_matrix.shape[0]:
                                break
                            f_matrix_new = f_matrix[n, :, :]
                            f_matrix_use.append(f_matrix_new)
                    FM = np.vstack(f_matrix_use).flatten()
                    logger.info("Loaded {} force configurations".format(len(f_matrix_use)))
                else:
                    logger.error("force_matrix.pkl not found for {} mode".format(settings.MODE))
                    raise FileNotFoundError("force_matrix.pkl is required")
            else:
                logger.error("Unsupported MODE: {}".format(settings.MODE))
                raise ValueError("MODE must be one of: RANDOM, AIMD, SCAILD, SPOR")
            
            np.savez_compressed(self.ForceArrayFile, F=FM)

            if settings.FIX_FC2:
                if settings.MAX_ORDER == 2:
                    logger.info(
                        "No force constants left for fitting "
                        + "when MAX_ORDER is 2 and FIX_FC2 is True."
                    )
                    raise RuntimeError
                else:
                    logger.info("Fix second-order IFCs during fitting.")
                    fc2_fmt = settings.FC2_FMT.upper()
                    NS_harm = spmat.load_npz("ns_harm.npz").toarray()
                    NS_anharm = self.NS_full.toarray()[
                        NS_harm.shape[0] :, NS_harm.shape[1] :
                    ]
                    if fc2_fmt == "PHONOPY":
                        if os.path.isfile("FORCE_CONSTANTS"):
                            fc2_filename = "FORCE_CONSTANTS"
                        else:
                            fc2_filename = "fc.hdf5"
                    elif fc2_fmt == "Q2R":
                        fc2_filename = "espresso.fc"
                    elif fc2_fmt == "NDARRAY":
                        fc2_filename = "Phi2.npz"
                    logger.info(
                        "Reading second-order IFCs from {}.".format(fc2_filename)
                    )
                    Phi2 = FC_model.read_force_constants(
                        self.CS_full, fc2_filename, order=2, format=fc2_fmt, full=False
                    )
                    if isinstance(Phi2, tuple):
                        Phi2 = Phi2[0].flatten()
                    else:
                        Phi2 = Phi2.flatten()
                    if settings.REMOVE_LR and settings.NAC != 0:
                        # TODO
                        pass
                ifc2_num = self.CS_full.get_number_of_ifcs_each_order()[2]
                # FIX: column slicing on CSR is slow; convert once to CSC.
                if spmat.issparse(SM_prime):
                    _SM_csc = SM_prime.tocsc()
                else:
                    _SM_csc = SM_prime
                SM2_prime = _SM_csc[:, :ifc2_num]
                SM3_prime = _SM_csc[:, ifc2_num:]
                del _SM_csc
                # FM -= SM2_prime @ Phi2 (small matrix-vector product, leave as-is)
                FM -= SM2_prime.dot(Phi2)
                del SM2_prime  # release ifc2 columns

                # FIX OLS (fix_fc2): when method is OLS, keep SM sparse.
                # SM = SM3_prime @ NS_anharm — sparse @ sparse stays sparse.
                if settings.MODEL.upper() == "OLS":
                    import numpy as _np_p, time as _ts
                    # FIX: same COO-safe memory helper as simultaneous branch
                    def _sp_mem_gb(m):
                        total = 0
                        for attr in ('data', 'indices', 'indptr', 'row', 'col'):
                            a = getattr(m, attr, None)
                            if a is not None:
                                total += a.nbytes
                        return total / 1e9

                    _ns3 = NS_anharm
                    if not spmat.issparse(_ns3):
                        _ns3 = spmat.csr_matrix(_ns3)
                    elif _ns3.format not in ('csr', 'csc'):
                        _ns3 = _ns3.tocsr()
                    if _ns3.dtype != _np_p.float32:
                        _ns3 = _ns3.astype(_np_p.float32)
                    if spmat.issparse(SM3_prime) and SM3_prime.format not in ('csr', 'csc'):
                        SM3_prime = SM3_prime.tocsr()
                    if SM3_prime.dtype != _np_p.float32:
                        SM3_prime = SM3_prime.astype(_np_p.float32)
                    print(f'[OLS-sparse fix_fc2] SM = SM3_prime @ NS_anharm  '
                          f'(sparse @ sparse, KEEP SPARSE)', flush=True)
                    print(f'[OLS-sparse fix_fc2]   SM3_prime: shape={SM3_prime.shape} '
                          f'fmt={SM3_prime.format} nnz={SM3_prime.nnz} '
                          f'mem={_sp_mem_gb(SM3_prime):.2f} GB', flush=True)
                    print(f'[OLS-sparse fix_fc2]   NS_anharm: shape={_ns3.shape} '
                          f'fmt={_ns3.format} nnz={_ns3.nnz} '
                          f'mem={_sp_mem_gb(_ns3):.2f} GB', flush=True)
                    _t0 = _ts.time()
                    SM = SM3_prime.dot(_ns3).tocsr()
                    if SM.dtype != _np_p.float32:
                        SM = SM.astype(_np_p.float32)
                    _mem = _sp_mem_gb(SM)
                    _density = SM.nnz / float(SM.shape[0] * SM.shape[1])
                    _dense_eq = SM.shape[0] * SM.shape[1] * 4 / 1e9
                    print(f'[OLS-sparse fix_fc2] result: shape={SM.shape} '
                          f'nnz={SM.nnz} density={_density:.2%} '
                          f'mem={_mem:.2f} GB  (dense would be '
                          f'{_dense_eq:.1f} GB)', flush=True)
                    print(f'[OLS-sparse fix_fc2] time={_ts.time()-_t0:.1f}s',
                          flush=True)
                    # Keep NS_anharm dense alive for APhi=NS_anharm.dot(coef)
                    # later; only SM3_prime is no longer needed.
                    del SM3_prime
                else:
                    # PATCH (fix_fc2 path): parallel sparse.dot + disk cache for
                    # SM = SM3_prime @ NS_anharm. Same logic as simultaneous path.
                    import numpy as _np_p, os as _os_p
                    _SM_F = 'sm_dense_fixfc2.npy'
                    _exp_fix = (SM3_prime.shape[0], NS_anharm.shape[1])
                    _force_fix = _os_p.environ.get('FORCE_REBUILD','false').lower()=='true'
                    _hit_fix = (not _force_fix) and _os_p.path.exists(_SM_F)
                    if _hit_fix:
                        try:
                            _chk = _np_p.load(_SM_F, mmap_mode='r')
                            if _chk.shape != _exp_fix or _chk.dtype != _np_p.float32:
                                print(f'[SM-cache fix_fc2] mismatch '
                                      f'{_chk.shape}/{_chk.dtype} vs {_exp_fix}/float32, '
                                      f'rebuild', flush=True)
                                _hit_fix = False
                                del _chk
                        except Exception as _e:
                            print(f'[SM-cache fix_fc2] read fail: {_e}, rebuild', flush=True)
                            _hit_fix = False
                    if _hit_fix:
                        import time as _ts
                        _t0 = _ts.time()
                        print(f'[SM-cache fix_fc2] HIT: loading {_SM_F} {_exp_fix}', flush=True)
                        SM = _np_p.load(_SM_F)
                        print(f'[SM-cache fix_fc2] loaded in {_ts.time()-_t0:.1f}s', flush=True)
                        del SM3_prime  # NS_anharm needed later for APhi=NS_anharm.dot(coef)
                    else:
                        # Materialize NS_anharm dense if sparse
                        if spmat.issparse(NS_anharm):
                            _NS3 = NS_anharm.toarray().astype(_np_p.float32)
                        else:
                            _NS3 = NS_anharm.astype(_np_p.float32) if NS_anharm.dtype != _np_p.float32 else NS_anharm
                        _n_thr = int(_os_p.environ.get('PHEASY_DOT_THREADS',
                                                       _os_p.environ.get('OMP_NUM_THREADS','64')))
                        print(f'[PATCH fix_fc2] parallel sparse.dot: '
                              f'{SM3_prime.shape} x {_NS3.shape}, threads={_n_thr}', flush=True)
                        SM = _np_p.empty((SM3_prime.shape[0], _NS3.shape[1]), dtype=_np_p.float32)
                        _rs = SM3_prime.shape[0]
                        _cs = max(1, (_rs + _n_thr - 1) // _n_thr)
                        def _fc_fix(i):
                            s = slice(i, min(i + _cs, _rs))
                            SM[s] = _np_p.asarray(SM3_prime[s].dot(_NS3))
                        from joblib import Parallel as _Par, delayed as _del
                        _Par(n_jobs=_n_thr, prefer='threads')(
                            _del(_fc_fix)(i) for i in range(0, _rs, _cs))
                        print(f'[PATCH fix_fc2] parallel sparse.dot done, SM={SM.shape}', flush=True)
                        del _NS3, SM3_prime  # NS_anharm needed later for APhi=NS_anharm.dot(coef)
                        import time as _ts
                        _t0 = _ts.time()
                        print(f'[SM-cache fix_fc2] saving {_SM_F} '
                              f'({SM.nbytes/1e9:.1f} GB)...', flush=True)
                        try:
                            _np_p.save(_SM_F, SM)
                            print(f'[SM-cache fix_fc2] saved in {_ts.time()-_t0:.1f}s', flush=True)
                        except Exception as _e:
                            print(f'[SM-cache fix_fc2] save fail: {_e}', flush=True)
            else:
                if settings.REMOVE_LR and settings.NAC != 0:
                    pass
                # FIX OLS: when method is OLS, keep SM as SPARSE (no densify,
                # no disk cache). Optimizer._ols_lsmr handles sparse natively
                # via LinearOperator. For (685968, 51590) this saves ~120 GB
                # peak memory and the ~25 min densify+disk-IO step.
                # [PATCH ols-twolevel-guard] 大体系(cutoff 5.2+)显式 SM_prime@NS
                # 会卡死/OOM. PHEASY_OLS_TWOLEVEL=1 时跳过此旧分支, 落到下面
                # _use_sparse 的 TwoLevelSM 两级 matvec (不生成 SM).
                import os as _os_g
                _ols_tl_guard = _os_g.environ.get('PHEASY_OLS_TWOLEVEL','1').lower() in ('1','true','yes')
                if settings.MODEL.upper() == "OLS" and not _ols_tl_guard:
                    import numpy as _np_p, time as _ts
                    # FIX: helper to compute sparse matrix memory regardless
                    # of storage format (CSR/CSC have .data/.indices/.indptr,
                    # but COO has .data/.row/.col — accessing .indices on a
                    # coo_matrix raises AttributeError. Use .nbytes via
                    # getattr fallback.)
                    def _sp_mem_gb(m):
                        total = 0
                        for attr in ('data', 'indices', 'indptr', 'row', 'col'):
                            a = getattr(m, attr, None)
                            if a is not None:
                                total += a.nbytes
                        return total / 1e9

                    # FIX: ensure SM_prime and _ns are CSR/CSC (never COO),
                    # so .dot is fast and downstream code is safe. SM_prime
                    # from earlier patches is usually CSR already; NS_full
                    # may come as COO from ASR construction.
                    _ns = self.NS_full
                    if not spmat.issparse(_ns):
                        _ns = spmat.csr_matrix(_ns)
                    elif _ns.format not in ('csr', 'csc'):
                        _ns = _ns.tocsr()
                    if _ns.dtype != _np_p.float32:
                        _ns = _ns.astype(_np_p.float32)
                    if spmat.issparse(SM_prime) and SM_prime.format not in ('csr', 'csc'):
                        SM_prime = SM_prime.tocsr()
                    if SM_prime.dtype != _np_p.float32:
                        SM_prime = SM_prime.astype(_np_p.float32)
                    _sp_mem = _sp_mem_gb(SM_prime)
                    _ns_mem = _sp_mem_gb(_ns)
                    print(f'[OLS-sparse] SM = SM_prime @ NS_full  '
                          f'(sparse @ sparse, KEEP SPARSE — no 141 GB densify)',
                          flush=True)
                    print(f'[OLS-sparse]   SM_prime: shape={SM_prime.shape} '
                          f'fmt={SM_prime.format} nnz={SM_prime.nnz} '
                          f'mem={_sp_mem:.2f} GB', flush=True)
                    print(f'[OLS-sparse]   NS_full:  shape={_ns.shape} '
                          f'fmt={_ns.format} nnz={_ns.nnz} '
                          f'mem={_ns_mem:.2f} GB', flush=True)
                    _t0 = _ts.time()
                    SM = SM_prime.dot(_ns).tocsr()
                    if SM.dtype != _np_p.float32:
                        SM = SM.astype(_np_p.float32)
                    _mem = _sp_mem_gb(SM)
                    _density = SM.nnz / float(SM.shape[0] * SM.shape[1])
                    _dense_eq = SM.shape[0] * SM.shape[1] * 4 / 1e9
                    print(f'[OLS-sparse] result: shape={SM.shape} '
                          f'nnz={SM.nnz} density={_density:.2%} '
                          f'mem={_mem:.2f} GB  (dense would be '
                          f'{_dense_eq:.1f} GB)', flush=True)
                    print(f'[OLS-sparse] time={_ts.time()-_t0:.1f}s', flush=True)
                    del _ns, SM_prime
                else:
                    # SM = SM_prime.dot(self.NS_full)
                    # PATCH: cache SM dense; skip ~3h sparse.dot on rerun
                    import numpy as _np_p, os as _os_p
                    import scipy.sparse as _sp_p

                    # ===== NEW: ALASSO/LASSO sparse 分支 (不影响 RFE) =====
                    # RFE 需要 dense SM (LinearOperator/lsmr), 故仅当
                    # PHEASY_LASSO_SPARSE=1 且非 RFE/RFE_TSQR 时走 sparse。
                    # [PATCH rfe-sparse] RFE 的 PheasyRFECV 只对 SM 做 SM@v / SM.T@u,
                    # scipy sparse CSR 原生支持, 无需 dense. 故 RFE 也走 sparse 路径,
                    # 避免 668 GB dense materialization (cutoff 5.5 规模).
                    # RFE_TSQR 仍需 dense (Q-less TSQR 走 dense R), 保持排除.
                    _is_rfe = _os_p.environ.get('PHEASY_USE_RFE','').lower() in ('1','true','yes')
                    _is_rfe_tsqr = _os_p.environ.get('PHEASY_USE_RFE_TSQR','').lower() in ('1','true','yes')
                    _lasso_sparse = _os_p.environ.get('PHEASY_LASSO_SPARSE','').lower() in ('1','true','yes')
                    # RFE 默认开 sparse (除非显式 PHEASY_RFE_SPARSE=0); LASSO/ALASSO 看 PHEASY_LASSO_SPARSE
                    _rfe_sparse = _os_p.environ.get('PHEASY_RFE_SPARSE','1').lower() in ('1','true','yes')
                    # [PATCH ols-twolevel] OLS 走两级 matvec (TwoLevelSM, 不生成 SM)
                    _is_ols = (settings.MODEL.upper() == "OLS")
                    _ols_twolevel = _os_p.environ.get('PHEASY_OLS_TWOLEVEL','1').lower() in ('1','true','yes')
                    # [PATCH rfe-twolevel] RFE 也可走两级 matvec (PHEASY_RFE_TWOLEVEL=1)
                    _rfe_twolevel = _os_p.environ.get('PHEASY_RFE_TWOLEVEL','0').lower() in ('1','true','yes')
                    _twolevel = (_is_ols and _ols_twolevel) or (_is_rfe and _rfe_twolevel)
                    _use_sparse = (
                        (not _is_rfe_tsqr)
                        and (
                            (_is_rfe and _rfe_sparse)
                            or (_lasso_sparse and not _is_rfe)
                            or (_is_ols and _ols_twolevel)
                        )
                    )
                    if _use_sparse:
                        import time as _ts_sp
                        _t0_sp = _ts_sp.time()
                        _ns_sp = self.NS_full
                        if not _sp_p.issparse(_ns_sp):
                            _ns_sp = _sp_p.csr_matrix(_ns_sp)
                        _ns_sp = _ns_sp.astype(_np_p.float32)
                        if not _sp_p.issparse(SM_prime):
                            SM_prime = _sp_p.csr_matrix(SM_prime)
                        SM_prime = SM_prime.astype(_np_p.float32)
                        print(f'[SM-sparse] ALASSO/LASSO sparse path: '
                              f'{SM_prime.shape} x {_ns_sp.shape}', flush=True)
                        if _twolevel:
                            # [PATCH ols/rfe-twolevel] 不显式相乘, 包成 TwoLevelSM.
                            # 关键: 不 del SM_prime/_ns_sp, TwoLevelSM 要持有引用.
                            _tl_who = "OLS" if _is_ols else "RFE"
                            from pheasy.core.optimizer import TwoLevelSM as _TwoLevelSM
                            # MKL matvec 要求 CSR/CSC (非 COO); NS_full 常为 COO -> 转 CSR.
                            if (not _sp_p.issparse(_ns_sp)) or _ns_sp.format not in ('csr','csc'):
                                _ns_sp = _sp_p.csr_matrix(_ns_sp)
                            if (not _sp_p.issparse(SM_prime)) or SM_prime.format not in ('csr','csc'):
                                SM_prime = SM_prime.tocsr() if _sp_p.issparse(SM_prime) else _sp_p.csr_matrix(SM_prime)
                            # 安全计算内存 (兼容 COO/CSR/CSC)
                            def _spmem(m):
                                t = 0
                                for a in ('data','indices','indptr','row','col'):
                                    x = getattr(m, a, None)
                                    if x is not None: t += x.nbytes
                                return t/1e9
                            _smp_mem = _spmem(SM_prime)
                            _nsp_mem = _spmem(_ns_sp)
                            SM = _TwoLevelSM(SM_prime, _ns_sp)
                            print(f'[SM-twolevel] {_tl_who} 两级 matvec (不生成 SM): '
                                  f'SM_prime{SM_prime.shape}({_smp_mem:.1f}GB) @ '
                                  f'NS{_ns_sp.shape}({_nsp_mem:.1f}GB), '
                                  f'SM.shape={SM.shape}, 峰值~{_smp_mem+_nsp_mem:.1f}GB',
                                  flush=True)
                        else:
                            SM = SM_prime.dot(_ns_sp).tocsr()
                            print(f'[SM-sparse] done in {_ts_sp.time()-_t0_sp:.1f}s, '
                                  f'SM={SM.shape} nnz={SM.nnz} '
                                  f'density={SM.nnz/(SM.shape[0]*SM.shape[1])*100:.2f}% '
                                  f'mem={(SM.data.nbytes+SM.indices.nbytes+SM.indptr.nbytes)/1e9:.1f}GB (sparse)',
                                  flush=True)
                            del SM_prime, _ns_sp
                    else:
                        # ===== 原 dense 路径 (RFE + 默认 LASSO, 完全不变) =====
                        _SM_F = 'sm_dense.npy'
                        _exp = (SM_prime.shape[0], self.NS_full.shape[1])
                        _force = _os_p.environ.get('FORCE_REBUILD','false').lower()=='true'
                        _hit = (not _force) and _os_p.path.exists(_SM_F)
                        if _hit:
                            try:
                                _chk = _np_p.load(_SM_F, mmap_mode='r')
                                if _chk.shape != _exp or _chk.dtype != _np_p.float32:
                                    print(f'[SM-cache] mismatch {_chk.shape}/{_chk.dtype} vs {_exp}/float32, rebuild', flush=True)
                                    _hit = False
                                    del _chk
                            except Exception as _e:
                                print(f'[SM-cache] read fail: {_e}, rebuild', flush=True)
                                _hit = False
                        if _hit:
                            import time as _ts
                            _t0 = _ts.time()
                            print(f'[SM-cache] HIT: loading {_SM_F} {_exp}', flush=True)
                            SM = _np_p.load(_SM_F)
                            print(f'[SM-cache] loaded in {_ts.time()-_t0:.1f}s', flush=True)
                            del SM_prime
                        else:
                            _NS = self.NS_full.toarray().astype(np.float32)
                            _n_thr = int(_os_p.environ.get('PHEASY_DOT_THREADS', _os_p.environ.get('OMP_NUM_THREADS','64')))
                            print(f'[PATCH] parallel sparse.dot: {SM_prime.shape} x {_NS.shape}, threads={_n_thr}', flush=True)
                            SM = _np_p.empty((SM_prime.shape[0], _NS.shape[1]), dtype=_np_p.float32)
                            _rs = SM_prime.shape[0]
                            _cs = max(1, (_rs + _n_thr - 1) // _n_thr)
                            def _fc(i):
                                s = slice(i, min(i + _cs, _rs))
                                SM[s] = _np_p.asarray(SM_prime[s].dot(_NS))
                            from joblib import Parallel as _Par, delayed as _del
                            _Par(n_jobs=_n_thr, prefer='threads')(_del(_fc)(i) for i in range(0, _rs, _cs))
                            print(f'[PATCH] parallel sparse.dot done, SM={SM.shape}', flush=True)
                            del _NS, SM_prime
                            import time as _ts
                            _t0 = _ts.time()
                            print(f'[SM-cache] saving {_SM_F} ({SM.nbytes/1e9:.1f} GB)...', flush=True)
                            try:
                                _np_p.save(_SM_F, SM)
                                print(f'[SM-cache] saved in {_ts.time()-_t0:.1f}s', flush=True)
                            except Exception as _e:
                                print(f'[SM-cache] save fail: {_e}', flush=True)
                FM = FM.astype(np.float32)

            # Train interatomic force constants
            # PATCH: optionally redirect LASSO -> ALASSO via env var
            import os as _os_alasso
            if (settings.MODEL.upper() == "LASSO" and
                _os_alasso.environ.get('PHEASY_USE_ALASSO', '').lower() in ('1','true','yes')):
                settings.MODEL = "ALASSO"
                print("[run_pheasy] PHEASY_USE_ALASSO=1 detected, "
                      "switching LASSO -> ALASSO", flush=True)
            # PATCH: [PATCH tsqr] optionally redirect LASSO -> RFE-OLS-TSQR
            if (settings.MODEL.upper() == "LASSO" and
                _os_alasso.environ.get('PHEASY_USE_RFE_TSQR', '').lower() in ('1','true','yes')):
                settings.MODEL = "RFE-OLS-TSQR"
                print("[run_pheasy] PHEASY_USE_RFE_TSQR=1 detected, "
                      "switching LASSO -> RFE-OLS-TSQR", flush=True)
            # PATCH: optionally redirect LASSO -> RFE via env var
            if (settings.MODEL.upper() == "LASSO" and
                _os_alasso.environ.get('PHEASY_USE_RFE', '').lower() in ('1','true','yes')):
                settings.MODEL = "RFE"
                print("[run_pheasy] PHEASY_USE_RFE=1 detected, "
                      "switching LASSO -> RFE", flush=True)
            optimizer = Optimizer(
                settings.MODEL,
                nalpha=settings.NALPHA,
                alpha_min=settings.ALPHA_MIN,
                alpha_max=settings.ALPHA_MAX,
                cv=settings.CV,
                tol=settings.TOL,
                max_iter=settings.MAX_ITER,
                rand_seed=settings.RAND_SEED,
                standardize=settings.STANDARDIZE,
            )
            if settings.MODEL.upper() == "LASSO":
                logger.info("Fitting force constants via the coordinate descent LASSO.")
            elif settings.MODEL.upper() == "ALASSO":
                logger.info("Fitting force constants via Adaptive LASSO (ALASSO).")
            elif settings.MODEL.upper() == "OLS":
                logger.info("Fitting force constants via the ordinary least-square.")
            elif settings.MODEL.upper() == "RFE":
                logger.info(
                    "Fitting force constants via Recursive Feature Elimination (RFE) "
                    "with Ridge base estimator (dense SM + LinearOperator, no copy).")
            elif settings.MODEL.upper() == "RFE-OLS-TSQR":
                logger.info(
                    "Fitting force constants via strict OLS + RFE "
                    "(Q-less augmented Tall-Skinny QR, R-space inheritance).")
            else:
                logger.error(
                    "Unknown linear model for fitting force constants, {}".format(
                        settings.MODEL
                    )
                )
                raise ValueError
            rank = SM.shape[1]
            optimizer.fit(SM, FM)
            fit_results = optimizer.results
            fit_metrics = optimizer.metrics
            FC_model.set_force_constant_metrics(fit_metrics)

            logger.info("Summary of force constants fitting:")
            if settings.MODEL.upper() in ("LASSO", "ALASSO"):
                logger.info(
                    "- Reaching the specified tolerance for the optimal "
                    + "alpha after {} iterations.".format(fit_results["n_iter"])
                )
                logger.info("- alpha_min: 1e{}".format(settings.ALPHA_MIN))
                logger.info("- alpha_max: 1e{}".format(settings.ALPHA_MAX))
                logger.info("- alpha_opt: {}".format(fit_results["alpha"]))
                logger.info("- RMSE_CV: {} eV/A".format(fit_metrics["rmse_path_mean"]))
            elif settings.MODEL.upper() == "RFE":
                logger.info("- RFE finished after {} rounds.".format(fit_results["n_iter"]))
                logger.info("- ridge_alpha: {}".format(fit_results["alpha"]))
                logger.info("- best CV RMSE: {} eV/A".format(fit_metrics["rmse_path_mean"]))
            logger.info("- RMSE: {} eV/A".format(fit_metrics["rmse"]))
            logger.info("- Relative error: {}".format(optimizer.metrics["re"]))
            logger.info("- Rank of coefficient matrix: {}".format(rank))
            logger.info("- Free IFC terms: {}".format(fit_results["coef"].shape[0]))
            logger.info(
                "- Non-zero IFC terms: {}".format(np.count_nonzero(fit_results["coef"]))
            )

            if settings.FIX_FC2:
                APhi = NS_anharm.dot(fit_results["coef"])
                np.savez_compressed(self.AForceConstantArrayFile, Phi=APhi)
                Phi = np.hstack([Phi2, APhi])
                np.savez_compressed(self.ForceConstantArrayFile, Phi=Phi)
                FC_model.set_force_constants(Phi)
            else:
                Phi = self.NS_full.dot(fit_results["coef"])
                np.savez_compressed(self.ForceConstantArrayFile, Phi=Phi)
                FC_model.set_force_constants(Phi)
                FC_model.write_force_constants(settings, self.CS_full, order=2)
                logger.info("Writing second-order force constants into file.")

            if settings.MAX_ORDER > 2:
                FC_model.write_force_constants(settings, self.CS_full, order=3)
                logger.info("Writing third-order force constants into file.")

            if settings.MAX_ORDER > 3:
                FC_model.write_force_constants(settings, self.CS_full, order=4)
                logger.info("Writing fourth-order force constants into file.")

            end_time_sub = datetime.datetime.now()
            time_cost = end_time_sub - start_time_sub
            logger.info(
                "Force constant fitting finished, time cost: {}.".format(time_cost)
            )

    def run_post_processing(self):
        """Post-process interatomic force constants by adding correct symmetries."""
        from sklearn.linear_model import LinearRegression

        settings = self.settings

        if settings.MODE.upper() == "PP":
            start_time_sub = datetime.datetime.now()
            logger.info("Post-process harmonic interatomic force constants.")
            if settings.MAX_ORDER != 2:
                logger.error(
                    "Force constant post-processing currently only "
                    + "implemented for the second order. Please set MAX_ORDER to 2."
                )
                raise ValueError

            # Create ForceConstants instance.
            FC_model = ForceConstants(self.scell, self.CS_full)

            # Read harmonic force constants
            fc2_fmt = settings.FC2_FMT.upper()
            if fc2_fmt == "PHONOPY":
                if os.path.isfile("FORCE_CONSTANTS"):
                    fc2_filename = "FORCE_CONSTANTS"
                else:
                    fc2_filename = "fc.hdf5"
            elif fc2_fmt == "Q2R":
                fc2_filename = "espresso.fc"
            elif fc2_fmt == "NDARRAY":
                fc2_filename = "Phi2.npz"
            logger.info("Reading second-order IFCs from {}.".format(fc2_filename))
            Phi2 = FC_model.read_force_constants(
                self.CS_full, fc2_filename, order=2, format=fc2_fmt, full=False
            )
            if isinstance(Phi2, tuple):
                Phi2 = Phi2[0].flatten()
            else:
                Phi2 = Phi2.flatten()

            # Read null space for harmonic force constants
            NS_harm = spmat.load_npz("ns_harm.npz").toarray()

            # Symmetrize harmonic force constants
            logger.info("Symmetrizing harmonic force constants.")
            linear_model = LinearRegression(fit_intercept=False).fit(NS_harm, Phi2)
            Phi2_reduced = linear_model.coef_
            Phi2_sym = np.dot(NS_harm, Phi2_reduced)
            FC_model.set_force_constants(Phi2_sym)
            FC_model.write_force_constants(settings, self.CS_full, order=2)
            logger.info("Writing harmonic force constants into file.")

            end_time_sub = datetime.datetime.now()
            time_cost = end_time_sub - start_time_sub
            logger.info(
                "Post-processing force constants finished, time cost: {}.".format(
                    time_cost
                )
            )


def main():
    # TODO: create a base CalTask class and put the realization of calculation
    #       into a class with a run method inheriting from CalTask class.
    """Pheasy Main Routine"""
    start_time = datetime.datetime.now()

    parser = InputParser()  # instantiate an input parser
    parser.read()  # config user settings via command-line and settings.nml

    logger.config(parser.settings.LOG_FILE)  # set logging outstream, console or file

    workflow = WorkFlow(parser)  # instantiate a workflow

    workflow.welcome()  # print pheasy version and logo

    workflow.run_cell()  # read primitive cell and create supercell

    workflow.run_neighborlist_cutoff()  # analyze neighbor list and cutoffs

    workflow.run_cluster_expansion()  # analyze supercell symmetry and generate cluster-orbit space

    workflow.run_symmetry_constraints()  # apply symmetry constraints and calculate null space

    workflow.run_sensing_matrix()  # create displaced configurations and construct sensing matrix

    workflow.run_fit_force_constants()  # fit interatomic force constants

    workflow.run_post_processing()  # post-process interatomic force constants

    """Finalize and estimate time cost"""
    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    logger.info("Finalize Pheasy, total time cost: {}.".format(total_time))
