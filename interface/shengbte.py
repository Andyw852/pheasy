"""ShengBTE interface for force constants."""
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2023 Changpeng Lin
# All rights reserved.

__all__ = ["write_fc3", "write_fc4"]

import itertools
from io import StringIO

import numpy as np

from pheasy.basic_io import logger
from pheasy.core.utilities import get_permutation_tensor


def write_ifc3(Phi, scell, clusters):
    """Write 3rd-order interatomic force constants into ShengBTE format.

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
    f = StringIO()
    fc3_num = 0
    for idx, orbit in enumerate(clusters):
        for cluster in orbit[1:]:
            if 0 in cluster.s2u[:, 0]:
                Gamma = cluster.get_crotation_tensor()
                Phi_tmp = Gamma.dot(Phi[idx, :, :, :].flatten())
                for _, at_idx in enumerate(cluster.permutations):
                    if at_idx[0] in scell.pmap:  # first atom within central unit cell
                        fc3_num += 1
                        Rmat = get_permutation_tensor(
                            cluster.atom_index, list(at_idx)
                        ).reshape((27, 27))
                        Phi_part = Rmat.dot(Phi_tmp)
                        ind_map, cell_pos = _relative_cell_positions(
                            at_idx, cluster, scell
                        )
                        f.write(f"\n{fc3_num}\n")
                        for i in range(2):
                            f.write(
                                "".join(map(lambda x: f"{x:25.15f}", cell_pos[i]))
                                + "\n"
                            )
                        f.write(
                            "".join(map(lambda x: f"{cluster.s2u[x,1]+1:6d}", ind_map))
                            + "\n"
                        )
                        for j, ind in enumerate(itertools.product([1, 2, 3], repeat=3)):
                            f.write(
                                f"{ind[0]:4d}{ind[1]:4d}{ind[2]:4d}{Phi_part[j]:25.15f}\n"
                            )

    with open("FORCE_CONSTANTS_3RD", "w") as fd:
        fd.write(f"{fc3_num}\n")
        fd.write(f.getvalue())


def write_ifc4(Phi, scell, clusters):
    """Write 4th-order interatomic force constants into ShengBTE format.

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
    f = StringIO()
    fc4_num = 0
    for idx, orbit in enumerate(clusters):
        for cluster in orbit[1:]:
            if 0 in cluster.s2u[:, 0]:
                Gamma = cluster.get_crotation_tensor()
                Phi_tmp = Gamma.dot(Phi[idx, :, :, :, :].flatten())
                for _, at_idx in enumerate(cluster.permutations):
                    if at_idx[0] in scell.pmap:  # first atom within central unit cell
                        fc4_num += 1
                        Rmat = get_permutation_tensor(
                            cluster.atom_index, list(at_idx)
                        ).reshape((81, 81))
                        Phi_part = Rmat.dot(Phi_tmp)
                        ind_map, cell_pos = _relative_cell_positions(
                            at_idx, cluster, scell
                        )
                        f.write(f"\n{fc4_num}\n")
                        for i in range(3):
                            f.write(
                                "".join(map(lambda x: f"{x:25.15f}", cell_pos[i]))
                                + "\n"
                            )
                        f.write(
                            "".join(map(lambda x: f"{cluster.s2u[x,1]+1:6d}", ind_map))
                            + "\n"
                        )
                        for j, ind in enumerate(itertools.product([1, 2, 3], repeat=4)):
                            f.write(
                                f"{ind[0]:4d}{ind[1]:4d}{ind[2]:4d}{ind[3]:4d}{Phi_part[j]:25.15f}\n"
                            )

    with open("FORCE_CONSTANTS_4TH", "w") as fd:
        fd.write(f"{fc4_num}\n")
        fd.write(f.getvalue())


def _relative_cell_positions(indices, cluster, scell):
    """Calculate relative positions of rest cells with respect to central cell.

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
        cell_pos[i, :] = np.dot(scell.cell.real.T, pos_tmp)
    cell_pos = np.where(np.abs(cell_pos) < 1e-4, 0, cell_pos)

    return ind_map, cell_pos
