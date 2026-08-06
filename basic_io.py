"""Classes and helper routines for basic io process."""
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2023 Changpeng Lin
# All rights reserved.

__all__ = [
    "CustomFormatter",
    "InputParser",
    "Logger",
    "logger",
]

import os
import sys
import logging
import argparse
from copy import deepcopy

import f90nml

from pheasy.version import __version__


class CustomFormatter(logging.Formatter):
    """Customized logging message format based on logging level."""

    def __init__(
        self,
        fmt="[%(asctime)s - %(module)s][%(levelname)s] %(message)s (%(filename)s:%(lineno)d)",
    ):
        logging.Formatter.__init__(self, fmt=fmt)

    def format(self, record):
        """Return format based on logging level."""
        fmt_default = self._style._fmt
        if record.levelno == logging.INFO:
            self._style._fmt = "[%(asctime)s] %(message)s"
        result = logging.Formatter.format(self, record)
        self._style._fmt = fmt_default
        return result


class Logger(logging.Logger):
    """Pheasy logger inheriting from logging.Logger class."""

    def __init__(self, name):
        """Constructor directly from logging.getLogger method.
        """
        super(Logger, self).__init__(name)

        # set sys.stdout as default outstream and config logging level.
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter())
        self.addHandler(console_handler)
        self.setLevel(logging.INFO)

    def config(self, filename=None, level=20):
        """Config logger including filestream, logging level and format.

        Parameters
        ----------
        filename : str
            Name of file to log.
        level: int
            Logging level, default is 20 (INFO).

        """
        fmt = CustomFormatter()
        self.setLevel(level)
        if filename is not None:
            file_handler = logging.FileHandler(filename)
            file_handler.setFormatter(fmt)
            self.addHandler(file_handler)


class InputParser(argparse.ArgumentParser):
    """Pheasy input parse inheriting from argparse.ArgumentParser class.

    Parameter settings from user-defined command line for manipulating
    Pheasy program is realized in this class. Input arguments that user
    can set must appear in upper case, while flags and commands must
    appear in lower case.

    https://docs.python.org/3/library/argparse.html

    """

    def __init__(self):
        """Initialize default settings."""
        super(InputParser, self).__init__()
        self.prog = "PHEASY " + __version__
        self.usage = "pheasy -x ... [--flags ...]"
        self.description = "Easy program for hard phonon problems."
        self.epilog = "Pheasy command-line inteface. Enjoy Pheasy !^_^!"
        self.formatter_class = argparse.ArgumentDefaultsHelpFormatter

        """Get default settings"""
        self.init_args()
        self._default = self.parse_args([])

    def init_args(self):
        """Initialize Pheasy arguments and help message."""

        """Main controller"""
        self.add_argument(
            "-s",
            dest="SPG_CLUS",
            action="store_true",
            default=False,
            help="This option asks code to analyze symmetry and dump clusters.",
        )
        self.add_argument(
            "-f",
            dest="FIT_IFC",
            action="store_true",
            default=False,
            help="This option asks code to fit force constants.",
        )
        self.add_argument(
            "-m",
            "--mode",
            dest="MODE",
            action="store",
            default="RANDOM",
            type=str,
            choices=["RANDOM", "AIMD", "SCAILD", "SPOR", "PP"],
            help="Pheasy kernel mode.",
        )
        self.add_argument(
            "--dim",
            dest="DIM",
            action="store",
            nargs=3,
            default=None,
            type=int,
            help="Supercell dimension.",
        )
        self.add_argument(
            "-u",
            "--disp",
            dest="U_VAL",
            action="store",
            default=0.01,
            type=float,
            help="Magnitude of displacement in angstrom.",
        )
        self.add_argument(
            "--pcell",
            dest="PCELL_FILENAME",
            action="store",
            default=None,
            type=str,
            help="""Primitive cell filename. By default, this filename
                    is POSCAR for VASP and is pw.in for QE.""",
        )
        self.add_argument(
            "--scell",
            dest="SCELL_FILENAME",
            action="store",
            default=None,
            type=str,
            help="""Supercell filename. By default, this filename
                    is SPOSCAR for VASP and is supercell.in for QE.""",
        )
        self.add_argument(
            "--pw_header",
            dest="PW_HEADER_FILE",
            action="store",
            default="header.pw",
            type=str,
            help="""Filename for PW header.""",
        )
        self.add_argument(
            "--read_scell",
            dest="READ_SCELL",
            action="store_true",
            default=False,
            help="""This option allows to read supercell structure
                    from file specified by SCELL_FILENAME.""",
        )
        self.add_argument(
            "--disp_file",
            dest="DISP_FILE",
            action="store_true",
            default=False,
            help="""Only valid when MODE='RANDOM'. With this option, the code
                    will read displaced configrations from files.""",
        )
        self.add_argument(
            "-w",
            "--max_order",
            dest="MAX_ORDER",
            action="store",
            default=2,
            type=int,
            help="Highest order included in cluster expansion.",
        )
        self.add_argument(
            "-b",
            "--nbody",
            dest="NBODY",
            action="store",
            nargs="+",
            default=None,
            type=int,
            help="""Excluded multibody interaction for each order force constants.
                    By default, all kinds of interacions are considered.""",
        )
        self.add_argument(
            "-n",
            "--ndata",
            dest="NDATA",
            action="store",
            default=20,
            type=int,
            help="Number of displaced configurations for training.",
        )
        self.add_argument(
            "--c2",
            "--cutoff2",
            dest="CUT2",
            action="store",
            default=None,
            type=float,
            help="""Cutoff distance for 2nd force constants.
                    A positive real value means cutoff distance in angstrom,
                    and a negative integer means cutoff in n-th nearest neighbor.
                    Please specify None to include all inteactions.""",
        )
        self.add_argument(
            "--c3",
            "--cutoff3",
            dest="CUT3",
            action="store",
            default=None,
            type=float,
            help="""Cutoff distance for 3rd force constants.
                    A positive real value means cutoff distance in angstrom,
                    and a negative integer means cutoff in n-th nearest neighbor.
                    Please specify None to include all inteactions.""",
        )
        self.add_argument(
            "--c4",
            "--cutoff4",
            dest="CUT4",
            action="store",
            default=None,
            type=float,
            help="""Cutoff distance for 4th force constants.
                    A positive real value means cutoff distance in angstrom,
                    and a negative integer means cutoff in n-th nearest neighbor.
                    Please specify None to include all inteactions.""",
        )
        self.add_argument(
            "--c5",
            "--cutoff5",
            dest="CUT5",
            action="store",
            default=None,
            type=float,
            help="""Cutoff distance for 5th force constants.
                    A positive real value means cutoff distance in angstrom,
                    and a negative integer means cutoff in n-th nearest neighbor.
                    Please specify None to include all inteactions.""",
        )
        self.add_argument(
            "--c6",
            "--cutoff6",
            dest="CUT6",
            action="store",
            default=None,
            type=float,
            help="""Cutoff distance for 6th force constants.
                    A positive real value means cutoff distance in angstrom,
                    and a negative integer means cutoff in n-th nearest neighbor.
                    Please specify None to include all inteactions.""",
        )
        self.add_argument(
            "--nskip",
            dest="NSKIP",
            action="store",
            default=None,
            type=int,
            help="""Only Valid when MODE='AIMD'. Number of steps
                    skipped in AIMD simulation, e.g. to drop the
                    process before reaching thermodynamic equilibrium.""",
        )
        self.add_argument(
            "--nstep",
            dest="NSTEP",
            action="store",
            default=10,
            type=int,
            help="""Only Valid when MODE='AIMD'.
                    Sampling interval for AIMD trajectories.""",
        )
        self.add_argument(
            "--rforce",
            dest="RFORCE",
            action="store_true",
            default=False,
            help="""Useful for low-symmetry materials. This option allows to
                    remove residual forces from perfect supercell. The filename
                    must be rforce.xml of VASP or rforce.out of QE.""",
        )

        """Linear model and fitting"""
        self.add_argument(
            "-l",
            "--model",
            dest="MODEL",
            action="store",
            default="OLS",
            type=str,
            choices=["OLS", "LASSO"],
            help="Linear model for fitting force constants.",
        )
        try:
            self.add_argument(
                "-c",
                "--null_space",
                dest="NULL_SPACE",
                action=argparse.BooleanOptionalAction,
                default=False,
                help="With this option, the code will generate null space.",
            )
        except AttributeError:
            self.add_argument(
                "-c",
                "--null_space",
                dest="NULL_SPACE",
                action="store_true",
                default=False,
                help="With this option, the code will generate null space.",
            )
        try:
            self.add_argument(
                "-d",
                "--sensing_matrix",
                dest="SENSING_MAT",
                action=argparse.BooleanOptionalAction,
                default=False,
                help="With this option, the code will generate sensing matrix.",
            )
        except AttributeError:
            self.add_argument(
                "-d",
                "--sensing_matrix",
                dest="SENSING_MAT",
                action="store_true",
                default=False,
                help="With this option, the code will generate sensing matrix.",
            )
        self.add_argument(
            "--crys_basis",
            dest="CRYS_BASIS",
            action="store_true",
            default=False,
            help="""True to deal with rotation matrix in crystal coordinate,
                    False for Cartesian coordinate.""",
        )
        self.add_argument(
            "--write_sym_cons",
            dest="WRITE_SYM_CONS",
            action="store_true",
            default=False,
            help="""True to write symmetry constraints into file.""",
        )
        self.add_argument(
            "--fix_fc2",
            dest="FIX_FC2",
            action="store_true",
            default=False,
            help="""This option allows to fix 2nd force constants during fitting
                    with values in format specified by FC2_FMT.""",
        )
        self.add_argument(
            "--fc2_fmt",
            dest="FC2_FMT",
            action="store",
            default="PHONOPY",
            type=str,
            choices=["PHONOPY", "Q2R", "NDARRAY"],
            help="""Only valid when FIX_FC2=True. The format of 2nd force constants
                    that will be read by code. In case of 'PHONOPY', the filename
                    must be FORCE_CONSTANTS or fc.hdf5; in case of 'Q2R', the filename
                    must be espresso.fc or espresso.fc.xml; in case of 'NDARRAY', the
                    filename must be Phi2.npz.""",
        )
        self.add_argument(
            "--exclude",
            dest="EXCLUDE",
            action="store",
            default=None,
            type=str,
            help="""This option allows to exclude certain range of samples in
                    training dataset and numbers should be separated either by
                    '-' or ','. For instance, 'EXCLUDE=1-8,11,15' means the samples
                    from 1 to 8, 11 and 15 will be excluded in force constant
                    fitting.""",
        )

        """"Symmetry and dielectrics"""
        self.add_argument(
            "--rasr",
            dest="RASR",
            action="store",
            default=None,
            type=str,
            choices=["BH", "H", "BHH"],
            help="""Types of rotational acoustic sum rules and equilibrium
                    conditions for vanishing stress. 'BH' for Born-Huang
                    rotaional invariances, 'H' for Huang equilibrium conditions
                    and 'BHH' for both.""",
        )
        self.add_argument(
            "--do_rasr",
            dest="DO_RASR",
            action="store",
            default="FIT",
            type=str,
            choices=["FIT", "PP"],
            help="""The way to implement rotational acoustic sum rules (RASRs).
                    'FIT' for imposing RASRs during force constant fitting, i.e.
                    RASRs included in null space construction. 'PP' for imposing
                    RASRs as an addtional post-processing.""",
        )
        self.add_argument(
            "--born",
            dest="BORN_FILE",
            action="store",
            default="BORN",
            type=str,
            help="""A file containing information of dielectric tensor
                    and Born effective charge tensor.""",
        )
        self.add_argument(
            "--born_sym",
            dest="BORN_SYM",
            action="store_true",
            default=False,
            help="""This option allows to symmetrize born effective charge
                    tensor by point group symmetry.""",
        )
        self.add_argument(
            "--nac",
            dest="NAC",
            action="store",
            default=0,
            type=int,
            help="""Non-analytic correction implemented by Gonze's method.
                    0 for disabling this option, 3 for using 3D screened
                    Coulomb potential and 2 for using 2D screened Coulomb
                    potential.""",
        )
        self.add_argument(
            "--lr",
            dest="REMOVE_LR",
            action="store_true",
            default=False,
            help="""This option allows to remove long-range part from
                    dipole-dipole interactions in the forces.""",
        )
        self.add_argument(
            "--is_magnetic",
            dest="IS_MAGNETIC",
            action="store_true",
            default=False,
            help="""(Experimental) Magnetic moment is considered in finding
                    crystal symmetry.""",
        )
        self.add_argument(
            "--magmom",
            dest="MAGMOM",
            action="store",
            nargs="+",
            default=None,
            type=float,
            help="""(Experimental) Array of magnetic moments. In case of
                    noncollinear calculation, an NATOM element array should
                    be provided; in case of collinear calculation, an NATOM*3
                    element array should be provided.""",
        )

        """Renormalization"""
        self.add_argument(
            "-t",
            "--temp",
            dest="TEMP",
            action="store",
            default=300.0,
            type=float,
            help="Renormalization temperature in Kelvin.",
        )
        self.add_argument(
            "--mesh",
            dest="MESH",
            action="store",
            nargs=3,
            default=None,
            type=int,
            help="Phonon mesh for lattice dynamics calculaions.",
        )
        self.add_argument(
            "--fc_iters",
            dest="FC_ITER",
            action="store",
            default=50,
            type=int,
            help="""Only valid when MODE='SCAILD' or MODE='SPOR'. Number of iterations
                    for self-consistency to get effective 2nd force constants.""",
        )
        self.add_argument(
            "--fc_mem",
            dest="FC_MEM",
            action="store",
            default=0.7,
            type=float,
            help="""Only valid when MODE='SCAILD'. Proportion of force-displacement 
                    dataset from previous iterations to be used in current iteraction.
                    Dataset from the step CURRENT_STEP*(1-FC_MEM) will be reused.""",
        )
        self.add_argument(
            "--fc_mix",
            dest="FC_MIX",
            action="store",
            default=0.6,
            type=float,
            help="""Only valid when MODE='SCAILD' or MODE='SPOR'. Mixing parameter
                    used in self-consistent iterations.""",
        )
        self.add_argument(
            "--fc_tol",
            dest="FC_TOL",
            action="store",
            default=1e-2,
            type=float,
            help="""Only valid when MODE='SCAILD' or MODE='SPOR'.
                    Tolerance for self-consistency.""",
        )

        """LASSO"""
        self.add_argument(
            "--cv",
            dest="CV",
            action="store",
            default=5,
            type=int,
            help="Only valid when MODEL='LASSO'. Fold of cross-validation.",
        )
        self.add_argument(
            "--nmu",
            "--nalpha",
            dest="NALPHA",
            action="store",
            default=50,
            type=int,
            help="""Only valid when MODEL='LASSO'. Number of alpha values
                    used in cross-validation of LASSO to find the optimal
                    alpha.""",
        )
        self.add_argument(
            "--mu_min",
            "--alpha_min",
            dest="ALPHA_MIN",
            action="store",
            default=-6,
            type=int,
            help="""Only valid when MODEL='LASSO'. Minimum power of alpha
                    and base is 10 and NALPHA alphas will be generated
                    ranging from 10^ALPHA_MIN to 10^ALPHA_MAX.""",
        )
        self.add_argument(
            "--mu_max",
            "--alpha_max",
            dest="ALPHA_MAX",
            action="store",
            default=-2,
            type=int,
            help="""Only valid when MODEL='LASSO'. Maximum power of alpha
                    and base is 10 and NALPHA alphas will be generated
                    ranging from 10^ALPHA_MIN to 10^ALPHA_MAX.""",
        )
        self.add_argument(
            "--max_iter",
            dest="MAX_ITER",
            action="store",
            default=20000,
            type=int,
            help="""Only valid when MODEL='LASSO'. Maximum number of
                    iterations.""",
        )
        self.add_argument(
            "--tol",
            dest="TOL",
            action="store",
            default=1e-4,
            type=float,
            help="""Only valid when MODEL='LASSO'. Tolerance for
                    LASSO optimization.""",
        )
        self.add_argument(
            "--seed",
            dest="RAND_SEED",
            action="store",
            default=666666,
            type=int,
            help="""Seed for pseudo random number generator. This is
                    useful for reproducing the same results, e.g. the
                    configurations of randomly displaced structures.""",
        )
        try:
            self.add_argument(
                "--std",
                dest="STANDARDIZE",
                action=argparse.BooleanOptionalAction,
                default=False,
                help="Training dataset will be first standardized with this option.",
            )
        except AttributeError:
            self.add_argument(
                "--std",
                dest="STANDARDIZE",
                action="store_true",
                default=False,
                help="Training dataset will be first standardized with this option.",
            )

        """Interfaces"""
        self.add_argument(
            "--qe",
            dest="QE",
            action="store_true",
            default=False,
            help="Quantum ESPRESSO mode is invoked with this option.",
        )
        self.add_argument(
            "--vasp",
            dest="VASP",
            action="store_true",
            default=True,
            help="VASP mode is invoked with this option.",
        )
        self.add_argument(
            "--hdf5",
            dest="HDF5",
            action="store_true",
            default=False,
            help="This option allows to write force constants in hdf5 format.",
        )
        self.add_argument(
            "--q2r",
            dest="Q2R",
            action="store_true",
            default=False,
            help="This option allows to write force constants in q2r.x format.",
        )
        self.add_argument(
            "--q2r_xml",
            dest="Q2R_XML",
            action="store_true",
            default=False,
            help="This option allows to write force constants in q2r.x xml format.",
        )
        self.add_argument(
            "--gpu_pbte",
            dest="GPU_PBTE",
            action="store_true",
            default=False,
            help="This option allows to write force constants in GPU_PBTE format.",
        )
        self.add_argument(
            "--full_ifc",
            dest="FULL_IFC",
            action="store_true",
            default=False,
            help="""This option allows to write full force constant tensor.
                    Only support Phonopy force constant format.""",
        )
        self.add_argument(
            "-o",
            "--log",
            dest="LOG_FILE",
            action="store",
            default=None,
            type=str,
            help="Log filename. If not specified, the code will use sys.stout." "",
        )
        self.add_argument(
            "-v",
            "--version",
            action="version",
            version="%(prog)s",
            help="Show version information.",
        )

        """Force-constant model"""
        self.add_argument(
            "--fc_model",
            dest="FC_MODEL",
            action="store_true",
            default=False,
            help="This option allows to build force-constant model.",
        )

        """System tolerance"""
        self.add_argument(
            "--symprec",
            dest="SYMPREC",
            action="store",
            default=1e-5,
            type=float,
            help="Tolerance for finding symmetry using spglib.",
        )
        self.add_argument(
            "--eps",
            dest="EPS",
            action="store",
            default=1e-4,
            type=float,
            help="Numerical tolerance of matrix norm for constructing null space.",
        )

    @property
    def settings(self):
        """argparse.Namespace : computational settings for all parameters."""
        return self._settings

    @settings.setter
    def settings(self, settings):
        """Update computational settings."""
        self._settings = settings

    def get_defaults(self):
        """Return the default settings for all parameters."""
        return self._default

    def read(self, filename="settings.nml"):
        """Parse user settings via command-line interface and F90 namelist.

        Parameters
        ----------
        filename : str
            Filename for F90 namelist which contains settings
            for calculations.

        """
        settings = deepcopy(self._default)
        if os.path.isfile(filename):
            nml = f90nml.read(filename)
            for _, arg in enumerate(nml["input"]):
                setattr(settings, arg.upper(), nml["input"][arg])
        settings = self.parse_args(namespace=settings)

        if settings.NBODY is None:
            settings.NBODY = [_ for _ in range(2, settings.MAX_ORDER + 1)]

        self._settings = settings

    @staticmethod
    def check_args(nml, natom):
        """Check the correctness of args set by user.

        Parameters
        ----------
        nml : argparse.Namespace
            Namespace generated by argparse.
        natom : int
            Number of atoms in primitive unit cell.

        """
        """DIM must be set by user."""
        if nml.DIM is None:
            logger.error("Argument DIM must be given.")
            raise NameError

        """MODE must be RANDOM, AIMD, SPOR or SCAILD."""
        if nml.MODE.upper() not in ["RANDOM", "AIMD", "SPOR", "SCAILD", "PP"]:
            logger.error(
                "Unknown running mode, please use 'pheasy -h' "
                + "to check supported calculation mode."
            )
            raise ValueError

        """The length of NBODY must be ORDER-1."""
        if nml.NBODY is not None:
            if len(nml.NBODY) != nml.MAX_ORDER - 1:
                logger.error("Wrong length of argument NBODY, must be ORDER-1.")
                raise ValueError

        """If IS_MAGNETIC=True, MAGMOM must be set. The length
           of MAGMOM must be the number of atoms in the cell
           (noncollinear) or the three times of the number of
           atoms in the cell (collinear)."""
        if nml.IS_MAGNETIC:
            if nml.MAGMOM is None:
                logger.error(
                    "The system is considered magnetic with IS_MAGNETIC=True, "
                    + "but MAGMOM is not set."
                )
                raise NameError
            else:
                mag_shape = len(nml.MAGMOM)
                if (mag_shape != natom) or (mag_shape != 3 * natom):
                    logger.error("Shape of MAGMOM is wrong.")
                    raise ValueError

        """The type of rotational acoustic sum rules are correctly set or not."""
        if nml.RASR is not None:
            if not isinstance(nml.RASR, str):
                logger.error("The value of RASR is not a string.")
                raise TypeError
            if not nml.RASR.upper() in ["BH", "H", "BHH"]:
                logger.error(
                    "The value of RASR must be BH, H or BHH, "
                    + "but you have {}.".format(nml.RASR)
                )
                raise ValueError


"""Initialize Pheasy logger"""
logging.setLoggerClass(Logger)
logger = logging.getLogger("Pheasy")
