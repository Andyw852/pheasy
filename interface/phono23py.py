"""Phonopy and Phono3py interface for force constants."""
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2023 Changpeng Lin
# All rights reserved.

__all__ = ["read_ifc2", "write_fc2", "write_fc3", "write_fc4"]

import itertools

import numpy as np

from pheasy.basic_io import logger
from pheasy.core.utilities import get_permutation_tensor


def read_ifc2(scell, clusters, filename="FORCE_CONSTANTS", full=True):
    """Read 2nd-order interatomic force constants in Phonopy format.

    Parameters
    ----------
    scell : Pheasy.atoms
        Supercell configration of the system.
    clusters : list
        Cluster space of 2nd-order force constants.
    filename : str, optional
        Filename of force constants in Phonopy format.
    full : bool, optional
        If True, the full force constant tensor with the shape
        (natom, natoms, 3, 3) is read, where natom is the number
        of atoms in the unit cell and natoms is the number of atoms
        in the supercell; otherwise, only those corresponding to the
        symmetry-independent clusters are read.

    Returns:
    -------
    numpy.ndarray
        Interatomic force constants in unit of eV/angstrom^2.
        When 'full' is True, it contains the full force constant tensor
        with the shape (natom, natoms, 3, 3); otherwise, they corresponds
        to the ones of the symmetry-independent 2nd-order clusters with
        the shape (cluster_number, 3, 3).

    """
    natom = scell.get_number_of_atoms_unit_cell()
    natoms = scell.get_global_number_of_atoms()
    ndim = scell.supercell.prod()

    if "hdf5" in filename:
        try:
            import h5py
        except ImportError:
            logger.error("The python-h5py needs to be installed: pip install h5py.")
            raise ModuleNotFoundError

        with h5py.File(filename, "r") as fd:
            if "fc2" in fd:
                key = "fc2"
            elif "force_constants" in fd:
                key = "force_constants"
            elif "force_constants_2nd" in fd:
                key = "force_constants_2nd"
            else:
                logger.error("Unknown key for 2nd-order IFCs in {}.".format(filename))
                raise RuntimeError

            Phi_tmp = fd[key]
            if Phi_tmp.shape != (natom, natoms) or Phi_tmp.shape != (natoms, natoms):
                logger.error("Two supercell configurations are not consistent.")
                raise RuntimeError

    else:
        with open(filename, "r") as fd:
            line = fd.readline().split()
            if len(line) == 1:  # old Phonopy style
                nat_a = int(line[0])
                nat_b = nat_a
                if nat_a != natoms:
                    logger.error("Two supercell configurations are not consistent.")
                    raise RuntimeError
            else:  # new Phonopy style
                nat_a = int(line[0])
                nat_b = int(line[1])
                if nat_a not in [natom, natoms] or nat_b != natoms:
                    logger.error("Two supercell configurations are not consistent.")
                    raise RuntimeError

            Phi_tmp = np.zeros((nat_a, nat_b, 3, 3))
            for i, j in np.ndindex((nat_a, nat_b)):
                line = fd.readline()
                Phi_tmp[i, j] = np.array(
                    [[float(l) for l in fd.readline().split()] for k in range(3)]
                )

    if Phi_tmp.shape[0] == natoms and ndim != 1:
        idx = np.arange(natoms, dtype="int")
        Phi_tmp = Phi_tmp[scell.pmap, :]
    if full:
        Phi = Phi_tmp
    else:
        clus_num = len(clusters)
        Phi = np.zeros((clus_num, 3, 3))
        for idx, orbit in enumerate(clusters):
            i, j = orbit[0].atom_index
            Phi[idx] = Phi_tmp[i // ndim, j, :, :]

    return Phi


def write_ifc2(Phi, scell, clusters, hdf5=False, full=False):
    """Write 2nd-order interatomic force constants into Phonopy format.

    Parameters
    ----------
    Phi : numpy.ndarray
        Interatomic force constants in unit of eV/angstrom^2,
        with the shape (cluster_number,3,3)
    scell : Pheasy.atoms
        Supercell configration of the system.
    clusters : list
        Cluster space of 2nd-order force constants.
    hdf5 : bool, optional
        If True, write interatomic force constants using hdf5.
    full : bool, optional
        If True, the full force constant tensor with the shape
        (natoms, natoms, 3, 3) is written, where natoms is the 
        number of atoms in the supercell; otherwise, the shape
        is (natom, natoms, 3, 3) where natom is the number of 
        atoms in the unit cell.

    """
    natoms = scell.get_global_number_of_atoms()
    ifc2 = np.zeros((natoms, natoms, 3, 3))

    # nclus = len(clusters)
    # Phi2 = Phi[: 9 * nclus]
    for idx, orbit in enumerate(clusters):
        for cluster in orbit[1:]:
            ia, ib = cluster.atom_index
            Gamma = cluster.get_crotation_tensor()
            Phi_tmp = Gamma.dot(Phi[idx, :, :].flatten())
            ifc2[ia, ib] = Phi_tmp.reshape((3, 3))
            if ia != ib:
                Rmat = get_permutation_tensor([ia, ib], [ib, ia]).reshape((9, 9))
                ifc2[ib, ia] = Rmat.dot(Phi_tmp).reshape((3, 3))

    if not full:
        ifc2 = ifc2[scell.pmap, :, :, :]
#    ifc2 = np.round((ifc2), decimals=15).astype('double')
    """Force constants in plain text"""
    with open("FORCE_CONSTANTS", "w") as fd:
        nat_a = ifc2.shape[0]
        nat_b = ifc2.shape[1]
        fd.write(f"{nat_a:>5d}{nat_b:5d}\n")
        for i, j in np.ndindex((nat_a, nat_b)):
            if full:
                fd.write(f"{i+1:5d}{j+1:5d}\n")
            else:
                k = scell.pmap[i]
                fd.write(f"{k+1:5d}{j+1:5d}\n")
            fd.write("".join(map(lambda x: f"{x:25.15f}", ifc2[i, j, 0])) + "\n")
            fd.write("".join(map(lambda x: f"{x:25.15f}", ifc2[i, j, 1])) + "\n")
            fd.write("".join(map(lambda x: f"{x:25.15f}", ifc2[i, j, 2])) + "\n")

    if hdf5:
        """Force constants in hdf5"""
        try:
            import h5py
        except ImportError:
            logger.error("The python-h5py needs to be installed: pip install h5py.")
            raise ModuleNotFoundError

        with h5py.File("fc2.hdf5", "w") as fd:
            fd.create_dataset("fc2", data=ifc2, compression="gzip")


def write_ifc3(Phi, scell, clusters, full=False):
    """Write 3rd-order interatomic force constants into Phonopy format.

    Only hdf5 format is supported.

    Parameters
    ----------
    Phi : numpy.ndarray
        Interatomic force constants in unit of eV/angstrom^3,
        with the shape (cluster_number,3,3,3)
    scell : Pheasy.atoms
        Supercell configration of the system.
    clusters : list
        Cluster space of 3rd-order force constants.
    full : bool, optional
        If True, the full force constant tensor with the shape
        (natoms,natoms,3,3,3) is written, where natoms is the 
        number of atoms in the supercell; otherwise, the shape
        is (natom,natoms,3,3,3) where natom is the number of 
        atoms in the unit cell.

    """
    try:
        import h5py
    except ImportError:
        logger.error("The python-h5py needs to be installed: pip install h5py.")
        raise ModuleNotFoundError

    natoms = scell.get_global_number_of_atoms()
    ifc3 = np.zeros((natoms, natoms, natoms, 3, 3, 3))

    for idx, orbit in enumerate(clusters):
        for cluster in orbit[1:]:
            Gamma = cluster.get_crotation_tensor()
            Phi_tmp = Gamma.dot(Phi[idx, :, :, :].flatten())
            for _, at_idx in enumerate(
                set(itertools.permutations(cluster.atom_index, 3))
            ):
                ia, ib, ic = at_idx
                Rmat = get_permutation_tensor(cluster.atom_index, list(at_idx)).reshape(
                    (27, 27)
                )
                ifc3[ia, ib, ic] = Rmat.dot(Phi_tmp).reshape((3, 3, 3))
    if not full:
        ifc3 = ifc3[scell.pmap, :, :, :, :, :]

    with h5py.File("fc3.hdf5", "w") as fd:
        fd.create_dataset("fc3", data=ifc3, compression="gzip")


def write_ifc4(Phi, scell, clusters, full=False):
    """Write 4th-order interatomic force constants into Phonopy format.

    Only hdf5 format is supported.

    Parameters
    ----------
    Phi : numpy.ndarray
        Interatomic force constants in unit of eV/angstrom^4,
        with the shape (cluster_number,3,3,3,3)
    scell : Pheasy.atoms
        Supercell configration of the system.
    clusters : dict
        Cluster space of 4th-order force constants.
    full : bool, optional
        If True, the full force constant tensor with the shape
        (natoms,natoms,3,3,3,3) is written, where natoms is the 
        number of atoms in the supercell; otherwise, the shape
        is (natom,natoms,3,3,3,3) where natom is the number of 
        atoms in the unit cell.

    """
    try:
        import h5py
    except ImportError:
        logger.error("The python-h5py needs to be installed: pip install h5py.")
        raise ModuleNotFoundError

    natoms = scell.get_global_number_of_atoms()
    ifc4 = np.zeros((natoms, natoms, natoms, natoms, 3, 3, 3, 3))

    for idx, orbit in enumerate(clusters):
        for cluster in orbit[1:]:
            Gamma = cluster.get_crotation_tensor()
            Phi_tmp = Gamma.dot(Phi[idx, :, :, :, :].flatten())
            for _, at_idx in enumerate(
                set(itertools.permutations(cluster.atom_index, 4))
            ):
                ia, ib, ic, id = at_idx
                Rmat = get_permutation_tensor(cluster.atom_index, list(at_idx)).reshape(
                    (81, 81)
                )
                ifc4[ia, ib, ic, id] = Rmat.dot(Phi_tmp).reshape((3, 3, 3, 3))
    if not full:
        ifc4 = ifc4[scell.pmap, :, :, :, :, :, :, :]

    with h5py.File("fc4.hdf5", "w") as fd:
        fd.create_dataset("fc4", data=ifc4, compression="gzip")
