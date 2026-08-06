"""Various functions related to create atomic displacements and matrix."""
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2023 Changpeng Lin
# All rights reserved.

__all__ = [
    "move_atoms_simple",
    "generate_displacements_from_file",
    "generate_displacements_from_aimd",
    "build_sensing_matrix",
    "build_sensing_matrix_slow",
]

import os
import itertools
from copy import deepcopy
from collections import deque

import numpy as np
from ase.io import read

from pheasy.structure.atoms import Atoms
from pheasy.basic_io import logger


def move_atoms_simple(struct, u_val):
    """Randomly move each atom in supercell by a magnitude of displacement.

    It will displace all of atoms in the input supercell in a random
    direction by the magnitude of displacement specified by u_val.

    Parameters
    ----------
    struct : pheasy.Atoms
        Input perfect supercell structure.
    u_val : float
        Magnitude of displacement when doing a random displacement.

    Returns:
    -------
    pheasy.Atoms
        Supercell structure with displacements.

    """

    natoms = struct.get_global_number_of_atoms()
    disp_struct = deepcopy(struct)
    u_vecs = np.random.uniform(-1, 1, (natoms, 3))
    u_vecs = u_val * u_vecs / np.linalg.norm(u_vecs, axis=1).reshape(-1, 1)
    disp_struct.positions += u_vecs
    disp_struct.set_atomic_displacements(u_vecs)

    return disp_struct


def generate_displacements_from_file(struct, filename, format="vasp"):
    """Read and return displaced configuration from file.

    It will first read from standard input structure file according
    to the DFT code specified by 'format' (i.e. POSCAR of VASP and
    pw.in of QE). If these files are not found, it will read from
    the output file of DFT code (i.e. vasprun.xml of VASP and pw.out
    of QE-pwscf).

    Parameters
    ----------
    struct : pheasy.Atoms
        Input perfect supercell structure.
    filename : str
        Filename containing displaced structure.
    format : str
        Format of structure file, i.e. name of DFT code.

    Returns:
    -------
    pheasy.Atoms
        Supercell structure with displacements.

    """
    if format == "qe":
        if os.path.isfile(filename):
            # read from QE pwscf input file
            disp_struct = read(filename, format="espresso-in")
        else:
            # read from QE pwscf output file
            filename_out = filename.split(".")
            filename_out[1] = "out"
            filename_out = ".".join(filename_out)
            if os.path.isfile(filename_out):
                disp_struct = read(filename_out, format="espresso-out")
            else:
                logger.error("Unknown format of structure file (QE).")
                raise FileNotFoundError
    else:
        # VASP
        if os.path.isfile(filename):
            # read from VASP POSCAR file.
            disp_struct = read(filename, format=format)
        else:
            # read from vasprun.xml.
            filename_xml = "vasprun.xml." + filename.split(".")[2]
            if os.path.isfile(filename_xml):
                disp_struct = read(filename_xml, format="vasp-xml")
            else:
                logger.error("Unknown format of structure file (VASP).")
                raise FileNotFoundError

    assert (
        struct.get_global_number_of_atoms() == disp_struct.get_global_number_of_atoms()
    )
    assert np.allclose(struct.cell.real, disp_struct.cell.real)

    disp_struct = Atoms(disp_struct)
    u_vecs = disp_struct.scaled_positions - struct.scaled_positions
    u_vecs = np.where(u_vecs > 0.5, u_vecs - 1.0, u_vecs)
    u_vecs = np.where(u_vecs < -0.5, u_vecs + 1.0, u_vecs)
    u_vecs = np.dot(struct.cell.real.T, u_vecs.T).T
    disp_struct.set_atomic_displacements(u_vecs)

    return disp_struct


def generate_displacements_from_aimd(struct, ndata, nskip=None, nstep=1, format="vasp"):
    """Read and return displaced configurations from AIMD.

    This function will create displaced configurations according to AIMD
    trajectories. It can skip the number of steps during the process of
    thermal equilibration specified by 'nskip' and sample 'ndata' number
    of AIMD trajectories with an interval of 'nstep'. The filename of
    AIMD trajectories must be aimd.xml if DFT code is VASP and aimd.out
    if DFT code is QE-pwscf.

    Parameters
    ----------
    struct : pheasy.Atoms
        Input perfect supercell structure.
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
    list(pheasy.Atoms)
        A list of supercell structure with displacements with
        the length of ndata.

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

    disp_struct_list = deque()
    for trj in aimd_trj[nskip:ntrj_exp:nstep]:
        disp_struct = Atoms(trj)
        u_vecs = disp_struct.scaled_positions - struct.scaled_positions
        u_vecs = np.where(u_vecs > 0.5, u_vecs - 1.0, u_vecs)
        u_vecs = np.where(u_vecs < -0.5, u_vecs + 1.0, u_vecs)
        u_vecs = np.dot(struct.cell.real.T, u_vecs.T).T
        disp_struct.set_atomic_displacements(u_vecs)
        disp_struct_list.append(disp_struct)

    return disp_struct_list


def build_sensing_matrix(cluster_space, u_vecs):
    """Construct sensing matrix from the given displaced configuration.

    Parameters
    ----------
    cluster_space : ClusterSpace object
        Cluster space that contains all of symmetry-distinct representative
        clusters of each order followed by their orbits.
    u_vecs_list : numpy.ndarray
        Atomic displacements for each atom in supercell.

    Returns:
    -------
    numpy.ndarray
        Sensing matrix of input displaced structure.

    """
    max_order = cluster_space.max_order
    natoms = len(u_vecs)
    u_vecs_1d = u_vecs.flatten()
    clusters = cluster_space.get_cluster_space()

    sensing_mat_block = deque()
    for order in range(2, max_order + 1):
        shape = [3] * order
        ifc_num = np.power(3, order)
        col = np.tile(range(ifc_num), order)
        for orbit in clusters[order]:
            block = np.zeros((natoms * 3, ifc_num))
            for cluster in orbit[1:]:
                block_tmp = np.zeros((natoms * 3, ifc_num))
                comp_list = np.array(cluster.atom_index) * 3 + list(np.ndindex(*shape))
                row = comp_list.T.flatten()
                comp_list = np.hstack(
                    [np.delete(comp_list, i, 1) for i in range(order)]
                ).reshape(ifc_num, order, order - 1)
                u_prods = np.prod(np.take(u_vecs_1d, comp_list), axis=2)
                np.add.at(block_tmp, (row, col), u_prods.T.flatten())
                Gamma = cluster.get_crotation_tensor().toarray()
                block_tmp = -block_tmp.dot(Gamma) / cluster.cluster_factorial()
                block += block_tmp
            sensing_mat_block.append(block)

    return np.hstack(sensing_mat_block)


def build_sensing_matrix_slow(cluster_space, u_vecs):
    """Construct sensing matrix from the given displaced configuration.

    Parameters
    ----------
    cluster_space : ClusterSpace object
        Cluster space that contains all of symmetry-distinct representative
        clusters of each order followed by their orbits.
    u_vecs_list : numpy.ndarray
        Atomic displacements for each atom in supercell.

    Returns:
    -------
    numpy.ndarray
        Sensing matrix of input displaced structure.

    """
    max_order = cluster_space.max_order
    natoms = len(u_vecs)
    u_vecs_1d = u_vecs.flatten()
    clusters = cluster_space.get_cluster_space()

    sensing_mat_block = deque()
    for order in range(2, max_order + 1):
        ifc_num = np.power(3, order)
        for orbit in clusters[order]:
            block = np.zeros((natoms * 3, ifc_num))
            for cluster in orbit[1:]:
                block_tmp = np.zeros((natoms * 3, ifc_num))
                for i, item in enumerate(itertools.product([0, 1, 2], repeat=order)):
                    u_i = list(np.array(cluster.atom_index) * 3 + np.array(item))
                    for j in set(u_i):
                        u_j = deepcopy(u_i)
                        u_j.remove(j)
                        block_tmp[j, i] = list(u_i).count(j) * u_vecs_1d[u_j].prod()
                Gamma = cluster.get_crotation_tensor().toarray()
                block_tmp = -block_tmp.dot(Gamma) / cluster.cluster_factorial()
                block += block_tmp
            sensing_mat_block.append(block)

    return np.hstack(sensing_mat_block)
