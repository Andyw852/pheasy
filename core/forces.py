"""Various functions for dealing interatomic forces from DFT codes."""
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2023 Changpeng Lin
# All rights reserved.

__all__ = [
    "read_interatomic_forces",
    "read_interatomic_forces_aimd",
]

from collections import deque

from ase.io import read

from pheasy.basic_io import logger


def read_interatomic_forces(filename, format="vasp"):
    """Read interatomic forces from output file of DFT code.

    Parameters
    ----------
    filename : str
        Filename of the output of DFT code.
    format : str
        Format of output file, i.e. name of DFT code.

    Returns:
    -------
    numpy.ndarray
        Interatomic forces for each atom in supercell in a
        shape of (natoms,3). The unit is eV/angstrom.

    """
    if format == "qe":
        struct = read(filename, format="espresso-out")
    else:
        # VASP
        struct = read(filename, format="vasp-xml")

    return struct.get_forces()


def read_interatomic_forces_aimd(ndata, nskip=None, nstep=1, format="vasp"):
    """Read and return interatomic forces from AIMD.

    This function will read interatomic forces stored in each AIMD trajectories. 
    It can skip the number of steps during the process of thermal equilibration 
    specified by 'nskip' and sample 'ndata' number of AIMD trajectories with an 
    interval of 'nstep'. The filename of AIMD trajectories must be aimd.xml if 
    DFT code is VASP and aimd.out if DFT code is QE-pwscf.

    Parameters
    ----------
    ndata : int
        Number of displaced configurations to read.
    nskip : int
        Number of steps skipped in AIMD simulation, e.g. to drop
        the process before reaching thermodynamic equilibrium.
    nstep : int
        Sampling interval for AIMD trajectories.
    format : str
        Format of structure file, i.e. name of DFT code.

    Returns:
    -------
    list(numpy.ndarray)
        Only returned when return_force is set to True. A list of
        interatomic forces for each configuration with a shape of
        (natoms,3). The length of list should be equal to the number
        of selected AIMD trajectories.

    """
    if format == "qe":
        filename = "aimd.out"
        aimd_trj = read(filename, index=":", format="espresso-out")
    else:
        # VASP
        filename = "aimd.xml"
        aimd_trj = read(filename, index=":", format="vasp-xml")

    if nskip is None:
        nskip = 0
    ntrj_tot = len(aimd_trj)
    ntrj_exp = nskip + ndata * nstep
    if ntrj_exp > ntrj_tot:
        logger.error(
            "Insufficient AIMD trajectories for the specified sampling scheme."
            + " You need at least {} trajectories, currently {}.".format(
                ntrj_exp, ntrj_tot
            )
        )
        raise IndexError

    force_list = deque()
    for trj in aimd_trj[nskip:ntrj_exp:nstep]:
        force_list.append(trj.get_forces())

    return list(force_list)
