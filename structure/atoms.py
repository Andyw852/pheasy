"""Atoms class and related functions for building matter structures."""
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Changpeng Lin
# All rights reserved.

__all__ = ["Atoms", "NeighborList", "create_supercell"]

import pickle
import itertools
from io import StringIO
from collections import OrderedDict

import numpy as np
from ase import Atoms as aseAtoms

from pheasy.basic_io import logger


class Atoms(aseAtoms):
    """Pheasy-adapted version of Atoms class inheriting from aseAtoms class.

    Dielectric tensor, Born effective charges, displacement for each atom
    in the cell, supercell dimension and atomic mapping between primitive
    unit cell and supercell are added as attributes.

    https://gitlab.com/ase/ase/-/blob/master/ase/atoms.py

    """

    def __init__(
        self,
        aseatoms=None,
        positions=None,
        scaled_positions=None,
        numbers=None,
        symbols=None,
        magmoms=None,
        cell=None,
        dim=None,
        epsilon=None,
        zeff=None,
    ):
        """
        To initialize an object, if aseatoms is given, scaled_positions,
        numbers, magmoms and cell arguments will be ignored.

        Parameters
        ----------
        aseatoms : ase.Atoms
            ase.Atoms class used in init function itself.
        positions : list or ndarray
            Cartesian positions of atoms.
        scale_positions : list or ndarray
            Scaled positions of atoms.
        numbers : list or ndarray
            List of atomic numbers
        symbols : list or ndarray
            List of chemical symbols.
        magmoms : list or ndarray
            Magnetic moments.  Can be either a single value for each atom
            for collinear calculations or three numbers for each atom for
            non-collinear calculations.
        cell : (3,3) list(float) or numpy.ndarray
            Lattice vectors of cell.
        dim : list(int) or numpy.ndarray
            Supercell dimension in shape (3,).
        epsilon : list or ndarray
            Macroscopic dielectric tensor.
        zeff : list or ndarray
            Born effective charges for atoms in primitive cell only.
            In case of supercell, only those in cell position 0 is
            provided.
        pw_header : str
            File containing the head of PW input, i.e. PW settings
            excluding CELL_PARAMETERS and ATOMIC_POSITIONS blocks.

        """
        if aseatoms is not None:
            super(Atoms, self).__init__(aseatoms)
        else:
            if positions is not None:
                super(Atoms, self).__init__(
                    positions=positions,
                    numbers=numbers,
                    symbols=symbols,
                    magmoms=magmoms,
                    cell=cell,
                )
            else:
                super(Atoms, self).__init__(
                    scaled_positions=scaled_positions,
                    numbers=numbers,
                    symbols=symbols,
                    magmoms=magmoms,
                    cell=cell,
                )

        self._ntyp = len(np.unique(self.numbers))
        if dim is not None:
            self._supercell = np.array(dim, dtype="int")
        # self._epsilon = np.identity(3)
        # self._zeff = np.zeros((self.get_number_of_atoms_unit_cell(), 3))
        if epsilon is not None:
            self._epsilon = epsilon
        if zeff is not None:
            self._zeff = zeff

    @property
    def ntyp(self):
        """int : number of atomic types."""
        return self._ntyp

    @property
    def epsilon(self):
        """numpy.ndarray : dielectric tensor."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        """Set delectic tensor.

        Parameters
        ----------
        epsilon : numpy.ndarray
            Macroscopic dielectric tensor.

        """
        if np.shape(epsilon) != (3, 3):
            logger.error("The shape of dielectric tensor must be (3,3).")
            raise ValueError
        self._epsilon = epsilon

    @property
    def zeff(self):
        """numpy.ndarray : Born effective charge tensor."""
        return self._zeff

    @zeff.setter
    def zeff(self, zeff):
        """Set Born effective charges.

        Parameters
        ----------
        zeff : numpy.ndarray
            Born effective charges for atoms in primitive cell only.
            In case of supercell, only those in cell position 0 is
            provided.

        """
        natom = self.get_number_of_atoms_unit_cell()
        if np.shape(zeff) != (natom, 3, 3):
            logger.error("The shape of Born effective charge tensor is wrong.")
            raise ValueError
        self._zeff = zeff

    @property
    def symops(self):
        """numpy.ndarray : symmetry operations from space group."""
        return self._symops

    @symops.setter
    def symops(self, symops):
        """Set space group symmetry operaions."""
        self._symops = symops
        self._nsym = symops["translations"].shape[0]

    @property
    def supercell(self):
        """numpy.ndarray : supercell dimension."""
        return self._supercell

    @supercell.setter
    def supercell(self, dim):
        """Set supercell dimension.

        Parameters
        ----------
        dim : list(int) or numpy.ndarray
            Supercell dimension in shape (3, ).

        """
        if np.shape(dim) != (3,):
            logger.error("The shape of supercell dim must be (3,).")
            raise ValueError
        self._supercell = np.array(dim, dtype="int")

    @property
    def smap(self):
        """ndarray : mapping of atoms in supercell to primitive unit cell."""
        return self._smap

    @property
    def pmap(self):
        """ndarray : mapping of atoms in primitive unit cell to supercell."""
        return self._pmap

    @property
    def scaled_positions(self):
        """ndarray : scaled positions of atoms."""
        return self.get_scaled_positions().copy()

    @property
    def ws_offsets(self):
        """numpy.ndarray : Wigner-Seitz translational vectors."""
        return self._ws_offsets

    def get_number_of_symmetries(self):
        """Return number of symmetry operations."""
        return self._nsym

    def get_number_of_atoms_unit_cell(self):
        """Return number of atoms in primitive unit cell.

        This method is deprecated in ASE and thus defined
        to return number of atoms in primitive unit cell.

        """
        if not hasattr(self, "_supercell"):
            ndim = 1
        else:
            ndim = np.prod(self._supercell)
        natom = int(self.get_global_number_of_atoms() / ndim)
        return natom

    def get_atomic_types(self):
        """Return atomic type for each atom in the cell.

        Returns:
        -------
        numpy.array
            The array contains the atomic type for each
            atom in the cell. The atomic type starts from 0.

        """
        _, idx, counts = np.unique(self.numbers, return_index=True, return_counts=True)
        numbers = self.numbers[np.sort(idx)]
        counts = counts[np.argsort(idx)]
        typs = []
        for i in range(len(numbers)):
            typs += [i] * counts[i]

        return np.array(typs, dtype=int)

    def get_atomic_displacements(self):
        """Return displacement vector for each atom in the cell.
        """
        return self._u_vecs

    def set_supercell(self, dim):
        """Set supercell dimension.

        Parameters
        ----------
        dim : list(int) or numpy.ndarray
            Supercell dimension in shape (3, ).

        """
        self._supercell = np.array(dim, dtype="int")

    def set_dielectrics(self, epsilon, zeff):
        """Set delectic tensor and Born effective charges.

        Parameters
        ----------
        epsilon : numpy.ndarray
            Macroscopic dielectric tensor.
        zeff : numpy.ndarray
            Born effective charges for atoms in primitive cell only.
            In case of supercell, only those in cell position 0 is
            provided.

        """
        self._epsilon = epsilon
        self._zeff = zeff

    def set_atomic_displacements(self, u_vecs):
        """Set displacement vector for each atom in the cell.

        Parameters
        ----------
        u_vecs : numpy.ndarray
            Atomic displacement vector for each atom in
            the cell. The Cartesian coordinate is used
            with values in angstrom.

        """
        self._u_vecs = u_vecs

    def set_smap(self):
        """Set mapping of atoms in supercell to unit cell.

        It generates the unit cell index and atom index in unit cell
        for each atoms in the supercell.

        """
        ndim = np.prod(self._supercell)
        natoms = self.get_global_number_of_atoms()
        smap1 = np.array(np.arange(natoms) % ndim, dtype="int")
        smap2 = np.array(np.arange(natoms) // ndim, dtype="int")
        self._smap = np.vstack((smap1, smap2)).transpose()

    def set_pmap(self):
        """Set mapping of atoms in unit cell to supercell."""
        ndim = np.prod(self._supercell)
        natom = self.get_number_of_atoms_unit_cell()
        self._pmap = np.array(np.arange(natom) * ndim, dtype="int")

    def set_wigner_seitz_offsets(self, ws_offsets):
        """Set the wigner-Seitz scaled positions."""
        self._ws_offsets = ws_offsets

    def get_symmetry_distinct_atoms(self):
        """Return an array of symmetry distinct atoms in the cell."""
        mapping = np.unique(self._symops["equivalent_atoms"], return_index=True)
        return mapping[1]

    def read_dielectrics(self, filename="BORN"):
        """Return dielectric properties read from file.

        Parameters
        ----------
        filename : str
            Name of file containing dielectric tensor
            and Born effective charges. The format of
            file should be:
            3x3 block of dielectric tensor
            3x3 block of Born effective charge of first atom
            3x3 block of Born effective charge of second atom
            ...

        """
        born_info = np.loadtxt(filename)
        self._epsilon = born_info[:3, :]
        self._zeff = np.reshape(born_info[3:, :], (-1, 3, 3))

    def to_spglib_tuple(self):
        """Convert and return structure information as a tuple.

        Returns:
        -------
        tuple
            The tuple consists of
            (lattice vectors,
             scaled positions of atoms in the cell,
             the corresponding atom types).

        """
        cell = self.cell.real
        scaled_positions = self.scaled_positions
        types = self.get_atomic_numbers()
        spglib_tuple = (cell, scaled_positions, types)

        return spglib_tuple

    def read_pw_header(self, filename="header.pw"):
        """Read PW input file.

        Parameters
        ----------
        filename: str
            File containing the head of PW input, i.e. PW settings
            excluding CELL_PARAMETERS and ATOMIC_POSITIONS blocks.

        """
        with open(filename, "r") as fd:
            header = fd.readlines()
        self._pw_header = StringIO()
        for line in header:
            self._pw_header.write(line)

    def write_pw_in(self, filename=None, direct=True):
        """Write crystal structure into PW input file.

        Parameters
        ----------
        filename: str
            File to be written as the PW input.
        direct : bool
            True to write atom positions in crystal coordinate,
            False to write in Cartesian coordinate.

        """
        fd = StringIO()
        if hasattr(self, "_pw_header"):
            fd.write(self._pw_header.getvalue())
        fd.write("CELL_PARAMETERS (angstrom)\n")
        for i in range(3):
            fd.write(
                "{0[0]:>20.15f} {0[1]:>20.15f} {0[2]:>20.15f}\n".format(
                    self.cell[i].tolist()
                )
            )
        symbols = self.get_chemical_symbols()
        if direct:
            fd.write("ATOMIC_POSITIONS (crystal)\n")
            for i, txt in enumerate(symbols):
                fd.write("%-10s" % txt)
                fd.write(
                    "{0[0]:>20.15f} {0[1]:>20.15f} {0[2]:>20.15f}\n".format(
                        self.scaled_positions[i].tolist()
                    )
                )
        else:
            fd.write("ATOMIC_POSITIONS (angstrom)\n")
            for i, txt in enumerate(symbols):
                fd.write("%-10s" % txt)
                fd.write(
                    "{0[0]:>20.15f} {0[1]:>20.15f} {0[2]:>20.15f}\n".format(
                        self.positions[i].tolist()
                    )
                )
        with open(filename, "w") as finalf:
            finalf.write(fd.getvalue())
        fd.close()

    def automatic_write(self, filename, format="vasp", direct=True):
        """Write crystal structure into file according to DFT code.

        Parameters
        ----------
        filename: str
            File for structure to be written.
        format : str
            Format of structure file, i.e. name of DFT code.
        direct : bool
            True to write atom positions in crystal coordinate,
            False to write in Cartesian coordinate.

        """
        if format == "qe":
            self.write_pw_in(filename, direct)
        else:
            # vasp
            self.write(filename, format=format, direct=direct)


def create_supercell(primitive, dim, is_magnetic=False):
    """Create supercell for the input structure.

    Parameters
    ----------
    primitive : pheasy.Atoms
        Primitive unit cell structure.
    dim : numpy.ndarray
        Supercell dimension.
    is_magnetic : bool
        True to consider a magnetic material

    Returns
    -------
    pheasy.Atoms
        The corresponding supercell structure.

    """
    ndim = np.prod(dim)
    natoms = primitive.get_global_number_of_atoms() * ndim
    numbers = np.repeat(primitive.numbers, ndim)

    trans = [x[::-1] for x in np.ndindex(*dim[::-1])]
    scaled_positions = np.zeros((3, natoms))
    for i, pos in enumerate(primitive.scaled_positions):
        scaled_positions[:, i * ndim : (i + 1) * ndim] = np.transpose(
            (trans + pos) / dim
        )
    scaled_positions = np.transpose(scaled_positions)

    cell = primitive.cell.real
    scell = np.zeros((3, 3))
    for i in range(3):
        scell[i, :] = cell[i, :] * dim[i]

    if is_magnetic:
        magmoms = primitive.get_initial_magnetic_moments()
        if len(magmoms.shape) == 1:
            smagmoms = np.repeat(magmoms, ndim)
        else:
            smagmoms = np.repeat(magmoms, ndim, axis=0)
        supercell = Atoms(
            scaled_positions=scaled_positions,
            numbers=numbers,
            magmoms=smagmoms,
            cell=scell,
            dim=dim,
        )
    else:
        supercell = Atoms(
            scaled_positions=scaled_positions, numbers=numbers, cell=scell, dim=dim
        )

    return supercell


class NeighborList(object):
    """Neighbor list object created for a supercell structure.

    The class builds a neighbor list for each atom in central
    primtive unit cell (cell position 0). The neighbor list
    corresponds to those atoms in optimal Wigner-Seitz cell.
    The nearest periodic image is considered.

    """

    def __init__(self, scell, equivalent_atoms=None, eps=1e-4):
        """Initialize a neighbor list based on input supercell.

        Parameters
        ----------
        scell : pheasy.Atoms
            Supercell structure.
        equivalent_atoms : (num_atoms,) numpy.ndarray
            A mapping of symmetry-distinct atom index in
            primtive unit cell.
        eps : float
            Numerical tolerance.

        """
        self._eps = eps
        self._cell = scell.cell.real
        self._supercell = scell.supercell
        self._positions = scell.scaled_positions
        self._central_atoms = [
            (
                idx,
                scell.get_atomic_numbers()[idx],
                scell.get_chemical_symbols()[idx],
                scell.scaled_positions[idx],
            )
            for idx in scell.pmap
        ]
        if equivalent_atoms is not None:
            mapping = np.unique(equivalent_atoms, return_index=True)
            distinct_atom_index = [scell.pmap[idx] for idx in mapping[1]]
            self._distinct_atoms = [
                (
                    idx,
                    scell.get_atomic_numbers()[idx],
                    scell.get_chemical_symbols()[idx],
                    scell.scaled_positions[idx],
                )
                for idx in distinct_atom_index
            ]

        self.run(scell)

    @property
    def cell(self):
        """cell: lattice vectors of supercell."""
        return self._cell

    @property
    def supercell(self):
        """ndarray : supercell dimension."""
        return self._supercell

    @property
    def positions(self):
        """ndarray : original scaled positions in supercell."""
        return self._positions

    @property
    def ws_offsets(self):
        """ndarray : Wigner-Seitz translational vectors."""
        return self._ws_offsets

    @property
    def nn_dists(self):
        """Return a list of neighbor distances for atoms in central unit cell.

        It depends on how the neighbor list is built. The first dimension is
        the number of atoms for which the neighbor list is constructed, and it
        differs when neighbor list is built for all atoms in central unit cell
        or only for those symmetry distict ones. The second dimension is the
        the distance for each nearest neighbor.

        Returns
        -------
        list(list)
            A list of neighbor distance for all atoms in central unit cell
            or for those symmetry-distict ones.

        """
        return self._nn_dists

    @property
    def eps(self):
        """eps: tolerance for generating neighbor list."""
        return self._eps

    @property
    def central_atoms(self):
        """Return atoms in central unit cell and their properties.

        Returns
        -------
        list(tuple)
            (atom index within supercell,
             the corresponding atomic number,
             the corresponding chemical symbol,
             the corresponding scaled position)

        """
        return self._central_atoms

    @property
    def distinct_atoms(self):
        """Return distinct atoms in central unit cell and their properties.

        Returns
        -------
        list(tuple)
            (atom index within supercell,
             the corresponding atomic number,
             the corresponding chemical symbol,
             the corresponding scaled position)

        """
        return self._distinct_atoms

    def get_neighbor_list(self):
        """Get the neighbor_list for atoms in central unit cell.

        It depends on how the neighbor list is built. The first dimension is
        the number of atoms for which the neighbor list is constructed, and it
        differs when neighbor list is built for all atoms in central unit cell
        or only for those symmetry distict ones.

        Returns
        -------
        list(list(tuple))
            This list contains the neighbor list for all atoms in central
            unit cell or for those symmetry-distict ones. Each element in the
            neighbor list contains a tuple of atomic properties:
            (atomic number,
             chemical symbol,
             weight in case of degenerate surface atoms,
             minimum distance for periodic image,
             the corresponding scaled positions,
             translational offset relative to central unit cell)

        """
        return self._neighbor_list

    def get_shell_neighbor_list(self):
        """Get neighbor list classified by shell for atoms in central unit cell.

        It depends on how the neighbor list is built. The first dimension is
        the number of atoms for which the neighbor list is constructed, and it
        differs when neighbor list is built for all atoms in central unit cell
        or only for those symmetry distict ones.

        Returns
        -------
        list(OrderedDict)
            This list contains an ordered dict which has the keys of shell
            distance with the corresponding values of all neighbor atoms on the
            shell, for all atoms in central unit cell or for those symmetry distict
            ones. Each neighbor atom on the shell has a tuple of atomic properties:
            (atomic number,
             chemical symbol,
             weight in case of degenerate surface atoms,
             minimum distance for periodic image,
             the corresponding scaled positions,
             translational offset relative to central unit cell)

        """
        shell_neighbor_list = []
        for i in range(len(self._neighbor_list)):
            neighbors_in_shell = OrderedDict()
            for dist in self._nn_dists[i]:
                key = np.around(dist, decimals=4)
                neighbors_in_shell[key] = []
            shell_neighbor_list.append(neighbors_in_shell)

        for i, neighbors in enumerate(self._neighbor_list):
            for item in neighbors:
                key = np.around(item[3], decimals=4)
                shell_neighbor_list[i][key].append(item)

        return shell_neighbor_list

    def wigner_seitz(self, atom1, atom2):
        """Return the weight and minimum periodic images with its distance.

        It calculates the number of degeneracies as weight when
        on the surface of the supercell Wigner-Seitz cell.

        Parameters
        ----------
        atom1 : (3,0) numpy.ndarray
            scaled position for the atom in central primitive unit cell.
        atom2 : (3,0) numpy.ndarray
            scaleds position for the atom in supercell.

        """
        trans_vecs = np.array(list(itertools.product(np.arange(-1.0, 2.0), repeat=3)))
        images = atom2 + trans_vecs
        rr = images - atom1
        dists = np.linalg.norm(np.dot(self._cell.T, rr.T), axis=0)
        min_dist = np.amin(dists)
        min_images = images[np.where(abs(dists - min_dist) < self._eps)]
        offsets = trans_vecs[np.where(abs(dists - min_dist) < self._eps)]
        weight = len(min_images)

        return (weight, min_dist, min_images, offsets)

    def run(self, scell):
        """Build neighbor list.

        Generate neighbor list for each symmetry-distinct atom in central unit cell.
        If the object does not have distinct_atoms attribute, then the neighbor list
        for all atoms in central unit cell will be generated.

        Parameters
        ----------
        scell : pheasy.Atoms
            Supercell structure.

        """
        nn_dists = []
        offset_list = []
        neighbor_list = []
        for _, atom1 in enumerate(self._central_atoms):
            dists = []
            trans = []
            neighbors = []
            for j, atom2 in enumerate(scell.scaled_positions):
                weight, min_dist, min_images, offsets = self.wigner_seitz(
                    atom1[3], atom2
                )
                atom_tuple = (
                    scell.get_atomic_numbers()[j],
                    scell.get_chemical_symbols()[j],
                    weight,
                    min_dist,
                    min_images,
                    offsets,
                )
                if True not in np.isclose(min_dist, dists, atol=self._eps):
                    dists.append(min_dist)
                trans.append(offsets)
                neighbors.append(atom_tuple)
            nn_dists.append(sorted(dists)[1:])
            offset_list.append(trans)
            neighbor_list.append(neighbors)

        if hasattr(self, "_distinct_atoms"):
            distinct_indices = list(map(lambda x: x[0], self._distinct_atoms))
            central_indices = list(map(lambda x: x[0], self._central_atoms))
            self._nn_dists = [
                nn_dists[central_indices.index(_)] for _ in distinct_indices
            ]
            self._neighbor_list = [
                neighbor_list[central_indices.index(_)] for _ in distinct_indices
            ]
        else:
            self._nn_dists = nn_dists
            self._neighbor_list = neighbor_list
        self._ws_offsets = np.array(offset_list, dtype=object)

    def get_neighbor_cutoff_distance(self, nth):
        """Find nearest neighbor up to the maximum allowed one.

        This function finds the cutoff distance up to the maximum allowed one.
        The cutoff distance is taken as the farthest distance of n-th nearest
        neighbor checked for the atom in the unit cell.

        Parameters
        ----------
        nth : int
            The n-th nearest neighbor.

        Returns
        -------
        float
            Cutoff distance in angstrom for the n-th nearest neighbor.

        """
        natom = len(self._nn_dists)
        nns = np.amin([len(self._nn_dists[n]) for n in range(natom)])
        if nth > nns - 1:
            logger.error(
                "Maximum allowed nearest neighbor is {}".format(nns - 1)
                + ", you have {}".format(nth)
            )
            raise ValueError
        dist1 = self._nn_dists[0][nth - 1]
        dist2 = self._nn_dists[0][nth]
        for i in range(1, natom):
            if dist1 > self._nn_dists[i][nth - 1]:
                dist1 = self._nn_dists[i][nth - 1]
            if dist2 < self._nn_dists[i][nth]:
                dist2 = self._nn_dists[i][nth]
        dist = (dist1 + dist2) / 2.0

        return dist

    def write(self, filename="neighbor_list.pkl"):
        """Write NeighborList object into pickle file.
    
        # TODO: only dump necessary attributes.

        Parameters
        ----------
        filename : str
            Filename to save NeighborList object.
        
        """
        with open(filename, "wb") as fd:
            pickle.dump(self, fd)

    @staticmethod
    def read(filename="neighbor_list.pkl"):
        """Read and create NeighborList object from pickle file.

        Parameters
        ----------
        filename : str
            Filename for NeighborList object to read from.

        Returns
        -------
        NeighborList object
            
        """
        with open(filename, "rb") as fd:
            return pickle.load(fd)

    def __repr__(self):
        """Return central atoms and neighbor distance."""
        txt = "NeighborList(central atoms: {!r}, nn_dists: {!r})"
        return txt.format(self._central_atoms, self._nn_dists)
