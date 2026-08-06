"""GPU_BTE interface for force constants."""
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2023 Changpeng Lin
# All rights reserved.

__all__ = ["write_fc2", "write_fc3", "write_fc4"]

import itertools
from io import StringIO

import numpy as np

from pheasy.basic_io import logger
from pheasy.constants import RY_TO_EV, ANGSTROM_TO_BOHR
from pheasy.core.utilities import get_permutation_tensor


def write_ifc2(Phi, scell, clusters):
    """Write 2nd-order interatomic force constants into GPU_BTE format.

    Parameters
    ----------
    Phi : numpy.ndarray
        Interatomic force constants in unit of eV/angstrom^2,
        with the shape (cluster_number,3,3)
    scell : Pheasy.atoms
        Supercell configration of the system.
    clusters : list
        Cluster space of 2nd-order force constants.

    """
    Phi_amu = Phi / RY_TO_EV / ANGSTROM_TO_BOHR ** 2

    f = StringIO()
    fc2_num = 0
    for idx, orbit in enumerate(clusters):
        for cluster in orbit[1:]:
            if 0 in cluster.s2u[:, 0]:
                Gamma = cluster.get_crotation_tensor()
                Phi_tmp = Gamma.dot(Phi_amu[idx, :, :].flatten())
                for _, at_idx in enumerate(cluster.permutations):
                    if at_idx[0] in scell.pmap:  # first atom within central unit cell
                        Rmat = get_permutation_tensor(
                            cluster.atom_index, list(at_idx)
                        ).reshape((9, 9))
                        Phi_part = Rmat.dot(Phi_tmp)
                        ind_map, cell_idx = _relative_cell_indices(
                            at_idx, cluster, scell
                        )

                        for i, ind in enumerate(itertools.product([1, 2, 3], repeat=2)):
                            fc2_num += 1
                            f.write("".join(map(lambda x: f"{int(x):4d}", cell_idx[0])))
                            f.write(
                                "".join(
                                    map(lambda x: f"{cluster.s2u[x,1]+1:4d}", ind_map)
                                )
                            )
                            f.write(f"{ind[0]:4d}{ind[1]:4d}{Phi_part[i]:25.15f}\n")

    with open("HFC.dat", "w") as fd:
        fd.write(f"{fc2_num}\n")
        fd.write(f.getvalue())


def write_ifc3(Phi, scell, clusters):
    """Write 3rd-order interatomic force constants into GPU_BTE format.

    Parameters
    ----------
    Phi : numpy.ndarray
        Interatomic force constants in unit of eV/angstrom^2,
        with the shape (cluster_number,3,3,3)
    scell : Pheasy.atoms
        Supercell configration of the system.
    clusters : list
        Cluster space of 3rd-order force constants.

    """
    Phi_amu = Phi / RY_TO_EV / ANGSTROM_TO_BOHR ** 3

    f = StringIO()
    fc3_num = 0
    for idx, orbit in enumerate(clusters):
        for cluster in orbit[1:]:
            if 0 in cluster.s2u[:, 0]:
                Gamma = cluster.get_crotation_tensor()
                Phi_tmp = Gamma.dot(Phi_amu[idx, :, :, :].flatten())
                for _, at_idx in enumerate(cluster.permutations):
                    if at_idx[0] in scell.pmap:  # first atom within central unit cell
                        fc3_num += 1
                        Rmat = get_permutation_tensor(
                            cluster.atom_index, list(at_idx)
                        ).reshape((27, 27))
                        Phi_part = Rmat.dot(Phi_tmp)
                        ind_map, cell_idx = _relative_cell_indices(
                            at_idx, cluster, scell
                        )

                        for i in range(2):
                            f.write("".join(map(lambda x: f"{int(x):4d}", cell_idx[i])))
                        f.write(
                            "".join(map(lambda x: f"{cluster.s2u[x,1]+1:4d}", ind_map))
                            + "\n"
                        )
                        f.write("".join(map(lambda x: f"{x:20.15f}", Phi_part)) + "\n")

    with open("CFC.dat", "w") as fd:
        fd.write(f"{fc3_num}\n")
        fd.write(f.getvalue())


def write_ifc4(Phi, scell, clusters):
    """Write 4th-order interatomic force constants into GPU_BTE format.

    Parameters
    ----------
    Phi : numpy.ndarray
        Interatomic force constants in unit of eV/angstrom^4,
        with the shape (cluster_number,3,3,3,3)
    scell : Pheasy.atoms
        Supercell configration of the system.
    clusters : list
        Cluster space of 4th-order force constants.

    """
    Phi_amu = Phi / RY_TO_EV / ANGSTROM_TO_BOHR ** 4

    f = StringIO()
    fc4_num = 0
    for idx, orbit in enumerate(clusters):
        for cluster in orbit[1:]:
            if 0 in cluster.s2u[:, 0]:
                Gamma = cluster.get_crotation_tensor()
                Phi_tmp = Gamma.dot(Phi_amu[idx, :, :, :, :].flatten())
                for _, at_idx in enumerate(cluster.permutations):
                    if at_idx[0] in scell.pmap:  # first atom within central unit cell
                        fc4_num += 1
                        Rmat = get_permutation_tensor(
                            cluster.atom_index, list(at_idx)
                        ).reshape((81, 81))
                        Phi_part = Rmat.dot(Phi_tmp)
                        ind_map, cell_idx = _relative_cell_indices(
                            at_idx, cluster, scell
                        )

                        for i in range(3):
                            f.write("".join(map(lambda x: f"{int(x):4d}", cell_idx[i])))
                        f.write(
                            "".join(map(lambda x: f"{cluster.s2u[x,1]+1:4d}", ind_map))
                            + "\n"
                        )
                        f.write("".join(map(lambda x: f"{x:20.15f}\n", Phi_part)))

    with open("QFC.dat", "w") as fd:
        fd.write(f"{fc4_num}\n")
        fd.write(f.getvalue())


def _relative_cell_indices(indices, cluster, scell):
    """Calculate the indices of rest cells with respect to central cell.

    Parameters
    ----------
    indices : list or numpy.ndarray
        Atomic indices within a cluster after permutation.
    cluster : Cluster
        A Cluster instance for the target atoms.
    scell : pheasy.Atoms
        Supercell structure.

    Returns:
    -------
    ind_map : numpy.ndarray
        A array of indices mapping atom indices after permutation
        to the original atom indices.
    cell_pos : numpy.ndarray
        A array of relative positions of rest cells with respect 
        to central cell.

    """
    indices_org = cluster.atom_index
    ind_org = np.argsort(indices_org)
    ind = np.argsort(indices)
    ind_map = ind_org[np.argsort(ind)]

    cell_pos = np.zeros((len(indices) - 1, 3))
    trans_vecs = scell.ws_offsets[cluster.s2u[ind_map[0], 1]]
    for i, idx in enumerate(ind_map[1:]):
        at_idx = indices[i + 1]
        at_cell0 = cluster.s2u[idx, 1]
        pos_tmp = scell.scaled_positions[scell.pmap[at_cell0]]
        pos_tmp = scell.scaled_positions[at_idx] + trans_vecs[at_idx][0] - pos_tmp
        pos_tmp = pos_tmp * scell.supercell
        cell_pos[i, :] = np.rint(pos_tmp)
        if np.where(np.abs(cell_pos[i, :] - pos_tmp) > 1e-4)[0].shape != (0,):
            logger.error(f"Unit cell indices are not all integers: {pos_tmp}")
            raise RuntimeError

    return ind_map, cell_pos
