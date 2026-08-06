"""Classes and functions for manipulating clusters and orbits."""
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2023 Changpeng Lin
# All rights reserved.

__all__ = ["Cluster", "ClusterSpace", "CSGenerator"]

import pickle
import itertools
from collections import deque, OrderedDict
import math
import numpy as np
import scipy.spatial.distance as sp_distance
import scipy.sparse as spmat

from pheasy.basic_io import logger
from .utilities import (
    kron_product,
    get_permutation_tensor,
    get_rotation_cartesian,
)


class Cluster(object):
    """Cluster used in expansion of lattice potential.

    This applies to both representative cluster and the cluster
    in its orbit.

    """

    def __init__(
        self,
        atoms,
        atom_index,
        types,
        symbols,
        rotmat=None,
        trans=None,
        offset=None,
        max_dist=None,
        lrep=True,
        eps=1e-4,
    ):
        """Initialization function.

        Parameters
        ----------
        atoms : (n,3) list(float) or numpy.ndarray
            A list of atoms in crystal coordinate forming the cluster.
        atom_index : list(int)
            The corresponding atom index within the supercell.
        types : (max_order,) list or numpy.ndarray
            Atom type or atomic number for each atom in the cluster.
        symbols : (max_order,) list
            Chemical symbol for each atom in the cluster.
        rotmat : (3,3) numpy.ndarray
            Rotation matrix in crystal coordinate.
        trans : (3,0) numpy.ndarray
            Translational vector in crystal coordinate.
        offset : (3,) list(float)
            Translational offset of atoms in the cluster with respect to
            the original unit cell, in unit of lattice vectors.
        max_dist : float
            Maximum distance between the atoms in the cluster,
            in angstrom unit.
        lrep : bool
            True for representative cluster and False for clusters in orbit.
        eps : float
            Numerical tolerance.

        """
        assert len(atoms) == len(atom_index)

        self._eps = eps
        self._lrep = lrep
        self._order = len(atoms)
        self._atoms = atoms
        self._atom_index = atom_index
        self._types = types
        self._symbols = symbols
        if not self._lrep:
            self._rotmat = rotmat
            self._trans = trans
        if offset is None:
            self._offset = offset
        if max_dist is not None:
            self._max_dist = max_dist
        self._proper = len(self._atom_index) == len(set(self._atom_index))

    @property
    def is_rep(self):
        """Check if the cluster is representative."""
        return self._lrep

    @property
    def order(self):
        """order: the order of cluster."""
        return self._order

    @property
    def atoms(self):
        """atoms: atoms in the cluster."""
        return self._atoms

    @property
    def atom_index(self):
        """index: atom index in the supercell for each atom in the cluster."""
        return self._atom_index

    @property
    def types(self):
        """types: atom type or atomic number for each atom in the cluster."""
        return self._types

    @property
    def symbols(self):
        """symbols: chemical symbols for each atom in the cluster."""
        return self._symbols

    @property
    def offset(self):
        """
        Translational offset of atoms in the cluster with respect
        to the original unit cell, in unit of lattice vectors.
        """
        return self._offset

    @property
    def rotmat(self):
        """rotmat: Rotation matrix in crystal coordinate."""
        return self._rotmat

    @property
    def crotmat(self):
        """crotmat: Rotation matrix in Cartesian coordinate."""
        return self._crotmat

    @property
    def trans(self):
        """trans: Translational vect/public/home/wangchao/miniconda3/envs/atomate2_p_a/lib/python3.11/site-packages/pheasy/core/cluster_orbit.py
or in crystal coordinate."""
        return self._trans

    @property
    def max_dist(self):
        """
        Maximum distance between the atoms in the cluster,
        in angstrom unit.
        """
        return self._max_dist

    @property
    def is_proper(self):
        """Check the cluster is proper or not.

        A proper cluster means all atoms in the cluster are unique, while
        an improper cluster means atoms in the cluster are not unique which
        has repeated atoms with the atom index within the supercell.
        
        """
        return self._proper

    @property
    def s2u(self):
        """Return mapping of atoms in cluster from supercell to unit cell.

        It contains the unit cell index and atom index in unit cell
        for each atoms in the cluster.

        """
        return self._s2u

    @property
    def permutations(self):
        """permutations: a set of atom index after permutation."""
        if not hasattr(self, "_permutations"):
            self.set_all_permutations()
        return self._permutations

    @property
    def force_constants(self):
        """ifcs: interatomic force constants of the cluster."""
        return self._ifcs

    @crotmat.setter
    def crotmat(self, crotmat):
        """Set (3,3) rotation matrix in Cartesian coordinate.

        Parameters
        ----------
        crotmat : numpy.ndarray
            (3,3) Rotation matrix in Cartesian coordinate.

        """
        self._crotmat = crotmat

    @force_constants.setter
    def force_constants(self, ifcs):
        """Set interatomic force constants to the cluster.

        Parameters
        ----------
        ifcs : numpy.ndarray
            Interatomic force constants.

        """
        if np.prod(np.shape(ifcs)) != np.power(self._order, 3):
            logger.error(
                "The shape of force constants does not match the order of cluster."
            )
            raise ValueError
        self._ifcs = ifcs

    def set_smap(self, ndim):
        """Set mapping of atoms in cluster from supercell to unit cell.

        It generates the unit cell index and atom index in unit cell
        for each atoms in the cluster.

        Parameters
        ----------
        ndim : int
            Nunber of unit cells in the supercell.

        """
        s2u1 = np.array(self._atom_index, dtype="int") % ndim
        s2u2 = np.array(self._atom_index, dtype="int") // ndim
        self._s2u = np.vstack((s2u1, s2u2)).transpose()

    def set_force_constants(self, ifcs):
        """Set interatomic force constants to the cluster.

        Parameters
        ----------
        ifcs : numpy.ndarray
            Interatomic force constants.

        """
        if np.prod(np.shape(ifcs)) != np.power(self._order, 3):
            logger.error(
                "The shape of force constants does not match the order of cluster."
            )
            raise ValueError
        self._ifcs = ifcs

    def get_rotation_tensor(self):
        """Return rotation tensor for IFC tensor (crystal coordinate)."""
        return self._rot_tensor

    def set_rotation_tensor(self):
        """Set rotation tensor for IFC tensor (crystal coordinate)."""
        self._rot_tensor = kron_product(self._rotmat, self._order)

    def get_crotation_tensor(self):
        """Return rotation tensor for IFC tensor (Cartesian coordinate)."""
        return self._crot_tensor

    def set_crotation_tensor(self):
        """Set rotation tensor for IFC tensor (Cartesian coordinate)."""
        self._crot_tensor = kron_product(self._crotmat, self._order)

    def set_isotropy_group(self, isotropy):
        """Set the isotropy group of a representative cluster.

        Parameters
        ----------
        isotropy : list(dict)
            Isotropy group of a representative cluster. It is a list
            and each element therein is a dictionary corresponding to
            a cluster that is isotropic to the representative one. The
            dictionary should have keys of 'atom_index', 'atoms', 'rotmat',
            'crotmat' and 'trans' which are the same quantities defined
            in the cluster.

        """
        self._isotropy = isotropy

    def get_isotropy_group(self):
        """Return the isotropy group of a representative cluster."""
        return self._isotropy

    def build_isotropy_symmetry_constraints(self, crys_basis=False):
        """Construct symmetry constraints from isotropy group.

        If there is no symmetry constraint imposed on the cluster,
        an identity matrix will be returned.

        TODO: check if the construction of sparse matrix in LIL form
              will be faster.

        Parameters
        ----------
        crys_basis : bool
            True to deal with rotation matrix in crystal coordinate,
            False for Cartesian coordinate.

        Returns:
        -------
        list(scipy.sparse.coo_matrix)
            A list of isotropy symmerty constraints in COO sparse matrix form.

        """
        if not self._lrep:
            logger.error(
                "Isotropy symmetry cannot be built for non-representative cluster."
            )
            raise NotImplementedError

        if crys_basis:
            rot_fmt = "rotmat"
        else:
            rot_fmt = "crotmat"
        constraints = []
        ifc_num = np.power(3, self._order)
        for cluster in self._isotropy:
            Rmat = get_permutation_tensor(
                self._atom_index, cluster["atom_index"]
            ).reshape((ifc_num, ifc_num))
            Gamma = kron_product(cluster[rot_fmt], self._order)
            constraints.append(Gamma - spmat.coo_matrix(Rmat))

        return constraints

    def build_improper_permutation_symmetry_constraints(self):
        """Construct permutation symmetry constraints for an improper cluster.

        An improper cluster means atoms in the cluster are not unique, e.g.
        (a, a) for 2nd cluster and (a, a, a) or (a, a, b) for 3rd cluster.
        This means if one exchange two same atoms the resulting IFCs will
        remain unchanged.

        TODO: check if the construction of sparse matrix in LIL form
              will be faster.

        Returns:
        -------
        list(scipy.sparse.coo_matrix)
            A list of permutation symmerty constraints in COO sparse matrix form.

        """
        if not self._lrep:
            logger.error(
                "Isotropy symmetry cannot be built for non-representative cluster."
            )
            raise NotImplementedError
        if self._proper:
            logger.error("Permutation symmetry cannot be built for proper cluster.")
            raise NotImplementedError

        permuted_list = []
        for i in range(self._order - 1):
            for j in range(i + 1, self._order):
                if self._atom_index[j] == self._atom_index[i]:
                    idx = list(range(self._order))
                    idx[i], idx[j] = idx[j], idx[i]
                    permuted_list.append(idx)

        constraints = []
        ifc_num = np.power(3, self._order)
        idx = list(range(self._order))
        for item in permuted_list:
            Rmat = get_permutation_tensor(idx, item).reshape((ifc_num, ifc_num))
            cons_mat = spmat.identity(ifc_num) - spmat.coo_matrix(Rmat)
            constraints.append(cons_mat)

        return constraints

    def set_all_permutations(self):
        """Set permutated list of atom index."""
        self._permutations = set(itertools.permutations(self._atom_index, self._order))

    def get_all_permutation_matrices(self):
        """Get all possible matrices for permutating atoms in a cluster.

        This method generates matrix representation R of permutation
        symmetry for IFCs, i.e. R \Phi(index) = \Phi(index_permuted).

        Returns:
        -------
        dict
            It contains the key of the atom index list after permutation
            with the corresponding value as permutation matrx with the shape
            (ifc_num, ifc_num) numpy.ndarray.

        """
        ifc_num = np.power(3, self._order)
        permute_mat = {}
        for i, idx in enumerate(
            set(itertools.permutations(self._atom_index, self._order))
        ):
            if self._atom_index == idx:
                permute_mat[idx] = np.identity(ifc_num)
            else:
                Rmat = get_permutation_tensor(self._atom_index, idx).reshape(
                    (ifc_num, ifc_num)
                )
                permute_mat[idx] = spmat.coo_matrix(Rmat)

        return permute_mat

    def cluster_factorial(self):
        """Calculate factorial of proper/improper cluster.

        Returns:
        -------
        int
            The factorial of the cluster.

        """
        index = self._atom_index
        factorial = 1
        for n in set(index):
            factorial = factorial * math.factorial(index.count(n))
        return factorial

    def __repr__(self):
        """Return cluster information."""
        txt = "Cluster(order: {!r}, index: {!r}, atoms: {!r}, is_rep: {!r})"
        return txt.format(self._order, self._atom_index, self._atoms, self._lrep)

    def __len__(self):
        """Return the order of the cluster."""
        return self._order

    def __eq__(self, clus):
        """Check if two clusters are equal based on atomic index."""
        return sorted(self._atom_index) == sorted(clus.atom_index)

    def __ne__(self, clus):
        """Check if two clusters are not equal based on atomic index."""
        return sorted(self._atom_index) != sorted(clus.atom_index)


class ClusterSpace(object):
    """Class defines a container-like object for building lattice potential.

    The cluster space is a container that includes all of representative
    (symmetry-independent) clusters and their orbits (obtained by symmetry
    operations of a space group), which are further represented by class
    Cluster.

    """

    def __init__(self, cell, cluster_space, symops, max_order, cutoffs, nbody):
        """Initialize essential information used to generate cluster space.

        Parameters
        ----------
        cell : numpy.ndarray
            Lattice vectors of supercell
        cluster_space : dict, shape of max_order-1
            It has keys for each order (int) and values corresponding to
            the cluster space of each order (list). Each element in the list
            starts by a representative cluster followed by its orbit.
        symops : dict
            Symmetry operations of space group.
            It has the keys "translations" and "rotations",
            and the values has the length of total number of symmetry.
        max_order : int
            highest order in generating cluster space.
        cutoffs : numpy.ndarray
            cutoff distances for each order clusters.
        nbody : list or numpy.ndarray
            excluded multibody interaction for each order force constants.

        """
        self._cell = cell
        self._CS = cluster_space
        self._nsym = symops["translations"].shape[0]
        self._symops = symops
        self._max_order = max_order
        self._cutoffs = cutoffs
        self._nbody = nbody

        clus_num = OrderedDict()
        ifc_num = OrderedDict()
        for order in range(2, max_order + 1):
            clus_num[order] = len(cluster_space[order])
            ifc_num[order] = len(cluster_space[order]) * np.power(3, order)
        self._num_of_clus = clus_num
        self._num_of_ifcs = ifc_num

    @property
    def max_order(self):
        """Return the maximum order of clusters."""
        return self._max_order

    @property
    def cutoffs(self):
        """Return cutoffs at each order."""
        return list(self._cutoffs.values)

    @property
    def nbody(self):
        """Return the maximum allowed multibody interaction of each order."""
        return self._nbody

    @property
    def cell(self):
        """cell: lattice vectors of supercell."""
        return self._cell

    @property
    def number_of_symmetries(self):
        """Return number of symmetry operations."""
        return self._nsym

    @property
    def total_number_of_clusters(self):
        """Return total number of representative clusters."""
        return np.sum(list(self._num_of_clus.values()))

    @property
    def total_number_of_ifcs(self):
        """Return total number of force constants."""
        return np.sum(list(self._num_of_ifcs.values()))

    @property
    def number_of_clusters_each_order(self):
        """Return the number of clusters at each order (list)."""
        return list(self._num_of_clus.values())

    @property
    def number_of_ifcs_each_order(self):
        """Return the number of force constants at each order (list)."""
        return list(self._num_of_ifcs.values())

    def get_number_of_clusters_each_order(self):
        """Return the number of clusters at each order (ordered dict)."""
        return self._num_of_clus

    def get_number_of_ifcs_each_order(self):
        """Return the number of force constants at each order (ordered dict)."""
        return self._num_of_ifcs

    def get_cluster_space(self, order=None):
        """Return the full cluster space or the one at the input order."""
        if order is not None:
            return self._CS[order]
        else:
            return self._CS

    def get_symmetry(self):
        """Return symmetry operations."""
        return self._symops

    def flatten(self):
        """Return clusters and their orbits of all orders as a list.
        
        Each element in the list is also a list which starts by a 
        representative cluster followed by its orbit.
        """
        clus_orbit = []
        for order in range(2, self._max_order):
            clus_orbit += self._CS_full[order]
        return clus_orbit

    def write(self, filename="cs.pkl"):
        """Write ClusterSpace object into pickle file.

        # TODO: only dump necessary attributes.

        Parameters
        ----------
        filename : str
            Filename to save cluster space.
        
        """
        with open(filename, "wb") as fd:
            pickle.dump(self, fd)

    @staticmethod
    def read(filename="cs.pkl"):
        """Read and create ClusterSpace object from pickle file.

        Parameters
        ----------
        filename : str
            Filename for ClusterSpace object to read from.

        Returns
        -------
        ClusterSpace object
            
        """
        with open(filename, "rb") as fd:
            return pickle.load(fd)

    def print_cluster_space_info(self):
        """Print cutoff, number of clusters, number of IFCs of each order."""
        for order in range(2, self._max_order + 1):
            logger.info(
                "Cutoff distance (A) for {0}-order IFCs: {1:3.2f}".format(
                    order, self._cutoffs[order]
                )
            )
        logger.info("Summary of cluster space:")
        for order in range(2, self._max_order + 1):
            if order == 2:
                logger.info(
                    "- HARM    | cluster number: {:<3d} | IFC number: {}".format(
                        self._num_of_clus[order], self._num_of_ifcs[order]
                    )
                )
            else:
                logger.info(
                    "- ANHARM{} | cluster number: {:<3d} | IFC number: {}".format(
                        order, self._num_of_clus[order], self._num_of_ifcs[order]
                    )
                )

    def __repr__(self):
        """Return cluster space information."""
        txt = "ClusterSpace(cell: {!r}, cutoffs: {!r}, num_of_clus: {!r}, num_of_ifcs: {!r})"
        return txt.format(
            self._cell, self._cutoffs, self._num_of_clus, self._num_of_ifcs
        )

    def __len__(self):
        """Return total number of representative clusters."""
        return self.total_number_of_clusters

    def __getitem__(self, index):
        """Overload the slice operation of a container."""
        return self.flatten()[index]

    def __iter__(self):
        """Overload the iter method to return a cluster space iterator."""
        self._current_index = 0
        return self

    def __next__(self):
        """Overload the next method for an iterator."""
        if self._current_index < len(self.total_number_of_clusters):
            clus_orbit = self.__getitem__(self._current_index)
            self._current_index += 1
            return clus_orbit
        raise StopIteration


class CSGenerator(object):
    """Class contains methods to generate cluster space."""

    def __init__(self, nn_list, symops, max_order, cutoffs, nbody, eps=1e-4):
        """Initialize with essential information used to generate cluster space.

        Parameters
        ----------
        nn_list : pheasy.Atoms
            Neighbor list for each atom in central unit cell.
        symops : dict
            Symmetry operations of space group.
            It has the keys "translations" and "rotations",
            and the values has the length of total number of symmetry.
        max_order : int
            highest order in generating cluster space.
        cutoffs : (max_order-1,) numpy.ndarray
            cutoff distances for each order clusters.
        nbody : (max_order-1,) list or numpy.ndarray
            excluded multibody interaction for each order force constants.

        """
        self._eps = eps
        self._cell = nn_list.cell
        self._supercell = nn_list.supercell
        self._natoms = nn_list.positions.shape[0]
        self._atoms = nn_list.positions
        self._central_atoms = nn_list.central_atoms
        self._nn_list = nn_list.get_neighbor_list()
        self._nsym = symops["translations"].shape[0]
        self._symops = symops
        self._max_order = max_order
        self._cutoffs = cutoffs
        self._nbody = nbody
        if hasattr(nn_list, "distinct_atoms"):
            self._distinct_atoms = nn_list.distinct_atoms

    def generate_represent_clusters_with_orbit(self):
        """Generate representative clusters and their orbit space.

        Returns
        -------
        An object of ClusterSpace.
            It contains all representative clusters with their orbits
            up to the maximum order used in cluster expansion.

        """
        ndim = np.prod(self._supercell)
        if hasattr(self, "_distinct_atoms"):
            first_atoms = self._distinct_atoms
        else:
            first_atoms = self._central_atoms

        CS_dict = {}
        clus_num = {}
        ifc_num = {}
        for i, order in enumerate(range(2, self._max_order + 1)):
            logger.info(
                "Cutoff distance (A) for {0}-order IFCs: {1:3.2f}".format(
                    order, self._cutoffs[order]
                )
            )
            orbit_space = set()
            CS_dict[order] = []
            for j, atom0 in enumerate(first_atoms):
                atoms_left, index_left = _select_atoms_within_cutoff(
                    self._cutoffs[order], self._nn_list[j]
                )
                for k, item in enumerate(
                    itertools.combinations_with_replacement(index_left, order - 1)
                ):
                    idx = [atom0[0]] + list(item)
                    if len(set(idx)) > self._nbody[i]:
                        continue
                    rep_atoms = np.vstack(
                        (atom0[3], atoms_left[[index_left.index(n) for n in item]])
                    )
                    in_cutoff, dist = self.check_distance(
                        self._cutoffs[order], rep_atoms
                    )
                    if in_cutoff and tuple(sorted(idx)) not in orbit_space:
                        clus_orbit = deque()
                        isotropy = deque()
                        repeated = set()
                        types = [self._nn_list[j][n][0] for n in idx]
                        symbols = [self._nn_list[j][n][1] for n in idx]
                        offset = np.array([self._nn_list[j][n][5][0] for n in idx])
                        rep_cluster = Cluster(
                            rep_atoms,
                            idx,
                            types,
                            symbols,
                            offset=offset,
                            max_dist=dist,
                            lrep=True,
                        )
                        rep_cluster.set_smap(ndim)
                        for s in range(self._nsym):
                            rotmat = self._symops["rotations"][s]
                            trans = self._symops["translations"][s]
                            hid_atoms = np.dot(rotmat, rep_atoms.T).T + trans
                            hid_idx = [
                                np.where(
                                    np.linalg.norm(
                                        self._atoms - pos - np.rint(self._atoms - pos),
                                        axis=1,
                                    )
                                    < self._eps
                                )[0][0]
                                for pos in hid_atoms
                            ]
                            if sorted(idx) == sorted(hid_idx):
                                crotmat = get_rotation_cartesian(rotmat, self._cell)
                                iso_clus = {
                                    "atom_index": hid_idx,
                                    "atoms": hid_atoms,
                                    "rotmat": rotmat,
                                    "crotmat": crotmat,
                                    "trans": trans,
                                }
                                isotropy.append(iso_clus)
                            if tuple(sorted(hid_idx)) not in repeated:
                                repeated.add(tuple(sorted(hid_idx)))
                                orbit_space.add(tuple(sorted(hid_idx)))
                                types = [self._nn_list[j][n][0] for n in hid_idx]
                                symbols = [self._nn_list[j][n][1] for n in hid_idx]
                                hid_cluster = Cluster(
                                    hid_atoms,
                                    hid_idx,
                                    types,
                                    symbols,
                                    rotmat=rotmat,
                                    trans=trans,
                                    max_dist=dist,
                                    lrep=False,
                                )
                                hid_cluster.set_smap(ndim)
                                hid_cluster.crotmat = get_rotation_cartesian(
                                    hid_cluster.rotmat, self._cell
                                )
                                hid_cluster.set_crotation_tensor()
                                clus_orbit.append(hid_cluster)
                        rep_cluster.set_isotropy_group(list(isotropy))
                        clus_orbit.appendleft(rep_cluster)
                        CS_dict[order].append(list(clus_orbit))
            clus_num[order] = len(CS_dict[order])
            ifc_num[order] = len(CS_dict[order]) * np.power(3, order)

        logger.info("Summary of cluster space:")
        for order in range(2, self._max_order + 1):
            if order == 2:
                logger.info(
                    "- HARM    | cluster number: {:<3d} | IFC number: {}".format(
                        clus_num[order], ifc_num[order]
                    )
                )
            else:
                logger.info(
                    "- ANHARM{} | cluster number: {:<3d} | IFC number: {}".format(
                        order, clus_num[order], ifc_num[order]
                    )
                )

        cluster_space = ClusterSpace(
            self._cell,
            CS_dict,
            self._symops,
            self._max_order,
            self._cutoffs,
            self._nbody,
        )

        return cluster_space

    def check_distance(self, cutoff, atoms):
        """Check whether the cluster satisfies the cutoff distance.

        Parameters
        ----------
        cutoff : float
            cutoff distance.
        atoms : numpy.ndarray
            scaled positions for atoms in the cluster.

        Returns:
        -------
        bool : True if the cluster satisfies the cutoff distance,
               False otherwise.
        max_dist : float
            maximum distance between atoms in the cluster.

        """
        positions = np.dot(self._cell.T, atoms.T).T
        dists = sp_distance.cdist(positions, positions, "euclidean")
        max_dist = np.amax(dists)
        if max_dist > cutoff:
            return (False, max_dist)
        else:
            return (True, max_dist)

    def __repr__(self):
        """Return information of cluster space generator."""
        configs = {
            "max_order": self._max_order,
            "cutoffs": self._cutoffs,
            "nbody": self._nbody,
        }
        txt = "CSGenerator(cell: {!r}, supercell: {!r}, natoms: {!r}, nsym: {!r}, configs: {!r})"
        return txt.format(self._cell, self._cutoffs, self._natoms, self._nsym, configs)


def _select_atoms_within_cutoff(cutoff, neighbor_list):
    """Select atoms in supercell within cutoff from the chosen central atom.

    Parameters
    ----------
    cutoff : float
        cutoff distance.
    neighbor_list : list(tuple)
        The neighbor list from which atoms are selected.

    Returns:
    -------
    atoms : numpy.ndarray
        Atoms in supercell that is within the cutoff distance from the
        chosen central atom.
    atom_index : list
        The corresponding atom index within supercell.

    """
    dists = np.array(list(map(lambda x: x[3], neighbor_list)))
    atom_index = np.where(dists < cutoff)
    atoms = np.array(list(map(lambda x: x[4][0], neighbor_list)))[atom_index]
    return atoms, list(atom_index[0])
