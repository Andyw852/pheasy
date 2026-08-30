"""Classes and functions for force constant manipulation and its symmetry."""
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Changpeng Lin
# All rights reserved.

__all__ = ["ForceConstants"]

from collections import OrderedDict

import numpy as np

from pheasy.basic_io import logger
from pheasy.interface import phono23py, qe_d3q, shengbte, gpu_pbte


class ForceConstants(object):
    """Class for handling force constants.
    
    It contains basic information about building the interatomic
    force constants of a system (e.g. supercell configuration) and
    related methods for operations on force constants (e.g. symmetry
    operations and IO process.)
    
    """

    def __init__(self, scell, cluster_space=None):
        """Initialization function.

        Parameters
        ----------
        scell : Pheasy.Atoms
            Supercell configuration.
        cluster_space : ClusterSpace, optional
            A cluster space containing all of symmetry-distinct 
            representative clusters of each order followed by 
            their orbits.

        """
        self._scell = scell

        if cluster_space is not None:
            atom_indices = OrderedDict()
            for order in range(2, cluster_space.max_order + 1):
                clusters = cluster_space.get_cluster_space(order)
                atom_indices[2] = list(map(lambda x: x[0].atom_index, clusters))
            self._atom_indices = atom_indices
            self._max_order = cluster_space.max_order
            self._num_of_ifcs = cluster_space.get_number_of_ifcs_each_order()
            self._num_of_clus = cluster_space.get_number_of_clusters_each_order()
        else:
            self._max_order = None
            self._num_of_ifcs = OrderedDict()
            self._num_of_clus = OrderedDict()
            self._atom_indices = OrderedDict()

        self._ifcs = OrderedDict()
        self._full_ifcs = OrderedDict()

    @property
    def scell(self):
        """Pheasy.atoms : supercell configuration."""
        return self._scell

    @property
    def force_constants(self):
        """Return a dictionary of force constants of each order.

        Returns:
        -------
        dict(numpy.ndarray)
            The key is the order of force constants and only those of
            symmetry-independent clusters are returned. The shape of
            force constants at each order is (cluster_number,3,3,...).
        
        """
        return self._ifcs

    @property
    def full_force_constants(self):
        """Return a dictionary of full force constants of each order.

        Returns:
        -------
        dict(numpy.ndarray)
            The key is the order of force constants and the shape of
            force constants is (natom,natoms,...,3,3,...).
        
        """
        return self._full_ifcs

    @property
    def max_order(self):
        """Return the maximum order of clusters."""
        return self._max_order

    @max_order.setter
    def max_order(self, max_order):
        """Set the maximum order of force constants."""
        self._max_order = max_order

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

    def get_force_constants(self, order=None, full=False):
        """Return a dictionary of force constants of each order.

        order : int, optional
            Force constants of that order will be returned.
            If None, a dictionary of numpy.ndarray for force
            constants at all orders will be returned.
        full : bool, optional
            If True, the full force constant tensor with the shape
            (natom,natoms,...,3,3,...) is returned, where natom is 
            the number of atoms in unit cell and natoms is the 
            number of atoms in supercell; otherwise, only those 
            corresponding to the symmetry-independent clusters 
            (cluster_number,3,3,...) are returned.
        
        """
        if order is not None:
            if order > self._max_order:
                logger.error(
                    f"The maximum allowed order is {self._max_order},"
                    + "but you specified the order {order}."
                )
                raise ValueError
            if full:
                if self._full_ifcs == OrderedDict():
                    logger.error("Full force constant tensor not initialized yet.")
                    raise AttributeError
                else:
                    return self._full_ifcs[order]
            else:
                return self._ifcs[order]
        else:
            if full:
                if self._full_ifcs == OrderedDict():
                    logger.error("Full force constant tensor not initialized yet.")
                    raise AttributeError
                else:
                    return self._full_ifcs
            else:
                return self._ifcs

    def set_force_constants(self, Phi, order=None):
        """Set force constants of all orders or at the specified order.

        Phi : numpy.ndarray
            A 1D array for interatomic force constants of 
            all orders. It only consists of those for 
            symmetry-distinct representative clusters.
        order : int
            The order of force constants to be set. If not set, the
            force constants up to the maximum order will be all set.
        
        """
        if order is None:
            start = 0
            for order in range(2, self._max_order + 1):
                nclus = self._num_of_clus[order]
                num = self._num_of_ifcs[order]
                shape = [nclus] + [3] * order
                self._ifcs[order] = Phi[start : start + num].reshape(shape)
                start += num
        else:
            self._ifcs[order] = Phi


    def set_force_constant_metrics(self, metrics):
        """Set the quality of the obtained force constants.
        
        Phi :dict
            A dictionary of diverise statistic quantities to 
            evaluate the quality of fitted force constants,
            such as R^2 score, mean squared error and so on.

        """
        self._metrics = metrics

    def check_acoustic_sum_rules():
        pass

    def read_force_constants(self, cluster_space, filename, order, format, full=True):
        """Read interatomic force constants from files.

        Parameters
        ----------
        cluser_space : ClusterSpace
            The full cluster space of the system.
        filename : str
            The name of force constants file.
        order : int
            The order of force constants to be read.
        format : str
            The format of force constant file, can be 'PHONOPY',
            'Q2R', 'SHENGBTE' and 'NDARRAY'.
        full : bool, optional
            If True, the full force constant tensor with the shape
            (natom,natoms,...,3,3,...) is read, where natom is the number
            of atoms in the unit cell and natoms is the number of atoms
            in the supercell; otherwise, only those corresponding to the
            symmetry-independent clusters are read.

        Returns:
        -------
        Phi : numpy.ndarray
            Interatomic force constants in unit of eV/angstrom^order.
            When 'full' is True, it contains the full force constant tensor
            with the shape (natom,natoms,...,3,3,...); otherwise, they
            correspond to the ones of the symmetry-independent clusters
            with the shape (cluster_number,3,3,...).

        """
        clusters = cluster_space.get_cluster_space(order)

        if order == 2:
            if format == "PHONOPY":
                Phi = phono23py.read_ifc2(self._scell, clusters, filename, full)
            elif format == "Q2R":
                Phi = qe_d3q.read_ifc2(self._scell, clusters, filename, full)
            elif format == "NDARRAY":
                Phi = np.load(filename)["Phi"]
            else:
                logger.error("Unknown format of force contants: {}".format(format))
                raise ValueError
        else:
            logger.error("Unimplemented value of the order of IFCs: {}".format(order))
            raise ValueError

        if full:
            self._full_ifcs[order] = Phi
        else:
            self._ifcs[order] = Phi
        self._atom_indices[order] = list(map(lambda x: x[0].atom_index, clusters))
        self._num_of_ifcs[order] = cluster_space.get_number_of_ifcs_each_order()[order]
        self._num_of_clus[order] = cluster_space.get_number_of_clusters_each_order()[
            order
        ]

        return Phi

    def write_force_constants(self, settings, cluster_space, order=2):
        """Write interatomic force constants of one specified order into file.

        Parameters
        ----------
        settings : argparse.Namespace
            A namespace containing computational settings.
        cluser_space : ClusterSpace
            The full cluster space of the system.
        order : int, optional
            The order of force constants to be write.

        """
        clusters = cluster_space.get_cluster_space(order)

        if order == 2:
            phono23py.write_ifc2(
                self._ifcs[2], self._scell, clusters, settings.HDF5, settings.FULL_IFC
            )
            if settings.Q2R or settings.Q2R_XML:
                qe_d3q.write_ifc2(
                    self._ifcs[2], self._scell, clusters, settings.NAC, settings.Q2R_XML
                )
            if settings.GPU_PBTE:
                gpu_pbte.write_ifc2(self._ifcs[2], self._scell, clusters)
        elif order == 3:
            shengbte.write_ifc3(self._ifcs[3], self._scell, clusters)
            if settings.HDF5:
                phono23py.write_ifc3(
                    self._ifcs[3], self._scell, clusters, settings.FULL_IFC
                )
            if settings.GPU_PBTE:
                gpu_pbte.write_ifc3(self._ifcs[3], self._scell, clusters)
        elif order == 4:
            shengbte.write_ifc4(self._ifcs[4], self._scell, clusters)
            # [PATCH fc4-hdf5] phono23py.write_ifc4 开 natoms^4*81 稠密数组,
            # natoms=189 需 770 GB 必爆. FourPhonon 只读 FORCE_CONSTANTS_4TH(已写),
            # 故仅小体系(<=80 原子, ~26GB)才写多余的 fc4.hdf5, 大体系跳过.
            _na4 = self._scell.get_global_number_of_atoms()
            if settings.HDF5 and _na4 <= 80:
                phono23py.write_ifc4(
                    self._ifcs[4], self._scell, clusters, settings.FULL_IFC
                )
            elif settings.HDF5:
                logger.info(
                    "Skip dense fc4.hdf5 for natoms=%d (would need ~%.0f GB); "
                    "FORCE_CONSTANTS_4TH (ShengBTE/FourPhonon) already written."
                    % (_na4, _na4**4 * 81 * 8 / 1e9)
                )
            if settings.GPU_PBTE:
                gpu_pbte.write_ifc4(self._ifcs[4], self._scell, clusters)
        else:
            logger.error("Unimplemented value of the order of IFCs: {}".format(order))
            raise ValueError

    def read(self):
        pass

    def write(self):
        pass

    def __repr__(self):
        pass
