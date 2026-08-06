"""Quantum ESPRESSO and D3Q interface for force constants."""
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2023 Changpeng Lin
# All rights reserved.

__all__ = ["read_ifc2", "write_ifc2"]

import os

import numpy as np

import pheasy.constants as const
from pheasy.basic_io import logger
from pheasy.core.utilities import get_permutation_tensor
from pheasy.structure.symmetry import get_ir_reciprocal_grid


def read_ifc2(scell, clusters, filename="espresso.fc", full=True):
    """Read 2nd-order interatomic force constants in q2r format.

    This function only reads the interatomic force constants of
    symmetry-independent clusters.

    Parameters
    ----------
    scell : Pheasy.atoms
        Supercell configration of the system.
    clusters : list
        Cluster space of 2nd-order force constants. 
    filename : str, optional
        Filename of force constants in Q2R format.
    full : bool, optional
        If True, the full force constant tensor with the shape
        (natom, natoms, 3, 3) is read, where natom is the number
        of atoms in the unit cell and natoms is the number of atoms
        in the supercell; otherwise, only those corresponding to the
        symmetry-independent clusters are read.

    Returns:
    -------
    Phi : numpy.ndarray
        Interatomic force constants in unit of eV/angstrom^2.
        When 'full' is True, it contains the full force constant tensor
        with the shape (natom, natoms, 3, 3); otherwise, they corresponds
        to the ones of the symmetry-independent 2nd-order clusters with
        the shape (cluster_number, 3, 3).
    epsilon (optional) : (3,3) numpy.ndarray
        Dielectric tensor, only returned when the system is polar.
    zeff (optional) : (natom,3,3) numpy.ndarray
        Born effective charge tensor, only returned when the system is polar.

    """
    natom = scell.get_number_of_atoms_unit_cell()
    natoms = scell.get_global_number_of_atoms()

    filename_xml = filename + ".xml"
    if os.path.isfile(filename_xml):
        import xml.etree.ElementTree as ET

        ifc_root = ET.parse(filename_xml).getroot()
        struct_info = ifc_root.find("GEOMETRY_INFO")
        ifc_info = ifc_root.find("INTERATOMIC_FORCE_CONSTANTS")
        ntyp = int(struct_info.find("NUMBER_OF_TYPES").text)
        nat = int(struct_info.find("NUMBER_OF_ATOMS").text)
        dim = np.int32(ifc_info.find("MESH_NQ1_NQ2_NQ3").text.replace("\n", "").split())
        ndim = dim.prod()
        if nat * ndim != natoms or nat != natom:
            logger.error("Two supercell configurations are not consistent.")
            raise RuntimeError

        dielectric_info = ifc_root.find("DIELECTRIC_PROPERTIES")
        if (
            dielectric_info.attrib["epsil"] == "true"
            and dielectric_info.attrib["zstar"] == "true"
        ):
            has_born = True
            epsilon = dielectric_info.find("EPSILON").text.replace("\n", "").split()
            epsilon = np.float64(epsilon).reshape([3, 3])
            zeff = np.zeros([natom, 3, 3])
            for i, zstar in enumerate(dielectric_info.find("ZSTAR").getchildren()):
                zeff[i] = np.float64(zstar.text.replace("\n", "").split()).reshape(
                    [3, 3]
                )
        else:
            has_born = False

        Phi_tmp = np.zeros((natom, natoms, 3, 3), dtype="double", order="C")
        for i, j, k, l, m in np.ndindex((natom, natom, *dim)):
            icell = m + l * dim[1] + k * dim[1] * dim[2]
            Phi_blk = np.zeros([3, 3])
            for ifc_child in ifc_info.find(
                f"s_s1_m1_m2_m3.{i+1}.{j+1}.{k+1}.{l+1}.{m+1}"
            ).getchildren():
                Phi_blk += np.float64(ifc_child.text.replace("\n", "").split()).reshape(
                    [3, 3]
                )
            Phi_tmp[j, i * ndim + icell] = Phi_blk

    else:
        with open(filename, "r") as fd:
            line = fd.readline()  # first line
            ntyp = int(line.split()[0])
            nat = int(line.split()[1])
            ibrav = int(line.split()[2])
            if ibrav == 0:
                for i in range(3):
                    fd.readline()  # skip lattice vectors
            for i in range(ntyp):
                fd.readline()  # skip masses
            for i in range(natom):
                fd.readline()  # skip atomic positions
            has_born = fd.readline().split()[0]
            if has_born == "T":
                epsilon = np.zeros([3, 3])
                zeff = np.zeros([natom, 3, 3])
                for i in range(3):  # read dielectric tensor
                    line = fd.readline()
                    epsilon[i] = np.float64(line.split())
                for i in range(natom):
                    fd.readline()  # skip atom index
                    for j in range(3):
                        line = fd.readline()
                        zeff[i, j, :] = np.float64(line.split())
            dim = np.int32(fd.readline().split())  # supercell dimension
            ndim = dim.prod()
            if nat * ndim != natoms or nat != natom:
                logger.error("Two supercell configurations are not consistent.")
                raise RuntimeError

            Phi_tmp = np.zeros((natom, natoms, 3, 3), dtype="double", order="C")
            for l, k, j, i in np.ndindex((3, 3, natom, natom)):
                line = fd.readline()
                for i_dim in range(ndim):
                    line = fd.readline()
                    Phi_tmp[i, j * ndim + i_dim, k, l] = np.float64(
                        line.split()[3:]
                    ).sum()

    if full:
        Phi = Phi_tmp
    else:
        clus_num = len(clusters)
        Phi = np.zeros((clus_num, 3, 3))
        for idx, orbit in enumerate(clusters):
            i, j = orbit[0].atom_index
            Phi[idx] = (
                Phi_tmp[i // ndim, j, :, :]
                * const.RY_TO_EV
                * const.ANGSTROM_TO_BOHR ** 2
            )

    if has_born == "T":
        return (Phi, epsilon, zeff)
    else:
        return Phi


def write_ifc2(Phi, scell, clusters, nac=False, xml=False):
    """Write 2nd-order interatomic force constants in q2r format.

    Parameters
    ----------
    Phi : numpy.ndarray
        Interatomic force constants in unit of eV/angstrom^2,
        with the shape (cluster_number,3,3)
    scell : Pheasy.atoms
        Supercell configration of the system.
    clusters : list
        Cluster space of 2nd-order force constants.
    nac : bool, optional
        If True, write also dieletric and Born effective charge tensors.
    xml : bool, optional
        If True, write interatomic force constants in q2r xml format.

    """
    filename = "espresso.fc"
    natoms = scell.get_global_number_of_atoms()
    Phi_amu = Phi / const.RY_TO_EV / const.ANGSTROM_TO_BOHR ** 2
    ifc2 = np.zeros((natoms, natoms, 3, 3))

    for idx, orbit in enumerate(clusters):
        for cluster in orbit[1:]:
            ia, ib = cluster.atom_index
            Gamma = cluster.get_crotation_tensor()
            Phi_tmp = Gamma.dot(Phi_amu[idx, :, :].flatten())
            ifc2[ia, ib] = Phi_tmp.reshape((3, 3))
            if ia != ib:
                Rmat = get_permutation_tensor([ia, ib], [ib, ia]).reshape((9, 9))
                ifc2[ib, ia] = Rmat.dot(Phi_tmp).reshape((3, 3))
    ifc2 = ifc2[scell.pmap, :, :, :]

    ibrav = 0
    zero = 0.0
    ntyp = scell.ntyp
    dim = scell.supercell
    ndim = np.prod(dim)
    cell = scell.cell.real / dim
    lat = np.linalg.norm(cell[0])
    acell = cell / lat
    alat = lat * const.ANGSTROM_TO_BOHR
    gcell = scell.cell.reciprocal().real * dim * lat
    natom = scell.get_number_of_atoms_unit_cell()
    avol = scell.get_volume() / ndim * const.ANGSTROM_TO_BOHR ** 3
    _, idx = np.unique(scell.get_masses(), return_index=True)
    masses = scell.get_masses()[np.sort(idx)]
    masses_amu = scell.get_masses()[np.sort(idx)] / const.MASS_RY_TO_QE
    _, idx = np.unique(scell.get_chemical_symbols(), return_index=True)
    symbols = np.array(scell.get_chemical_symbols())[np.sort(idx)]
    symbols_full = np.array(scell.get_chemical_symbols())[scell.pmap]
    typs = scell.get_atomic_types()[scell.pmap]
    _, weights = get_ir_reciprocal_grid(scell, dim)
    nqpts = len(weights)

    if xml:
        """Force constants in xml"""
        import xml.dom.minidom as DOM

        fc_xml = DOM.Document()
        root = fc_xml.createElement("Root")
        fc_xml.appendChild(root)

        # GEOMETRY_INFO
        geometry = fc_xml.createElement("GEOMETRY_INFO")
        root.appendChild(geometry)
        node = fc_xml.createElement("NUMBER_OF_TYPES")
        geometry.appendChild(node)
        text = fc_xml.createTextNode(f"{ntyp}")
        node.appendChild(text)
        node = fc_xml.createElement("NUMBER_OF_ATOMS")
        geometry.appendChild(node)
        text = fc_xml.createTextNode(f"{natom}")
        node.appendChild(text)
        node = fc_xml.createElement("BRAVAIS_LATTICE_INDEX")
        geometry.appendChild(node)
        text = fc_xml.createTextNode(f"{ibrav}")
        node.appendChild(text)
        node = fc_xml.createElement("SPIN_COMPONENTS")
        geometry.appendChild(node)
        text = fc_xml.createTextNode("1")
        node.appendChild(text)
        node = fc_xml.createElement("CELL_DIMENSIONS")
        geometry.appendChild(node)
        text = fc_xml.createTextNode(f"{alat:24.15e}{zero:24.15e}{zero:24.15e}")
        node.appendChild(text)
        text = fc_xml.createTextNode(f"{zero:24.15e}{zero:24.15e}{zero:24.15e}")
        node.appendChild(text)
        node = fc_xml.createElement("AT")
        geometry.appendChild(node)
        text = fc_xml.createTextNode("".join(map(lambda x: f"{x:24.15e}", acell[0])))
        node.appendChild(text)
        text = fc_xml.createTextNode("".join(map(lambda x: f"{x:24.15e}", acell[1])))
        node.appendChild(text)
        text = fc_xml.createTextNode("".join(map(lambda x: f"{x:24.15e}", acell[2])))
        node.appendChild(text)
        node = fc_xml.createElement("BG")
        geometry.appendChild(node)
        text = fc_xml.createTextNode("".join(map(lambda x: f"{x:24.15e}", gcell[0])))
        node.appendChild(text)
        text = fc_xml.createTextNode("".join(map(lambda x: f"{x:24.15e}", gcell[1])))
        node.appendChild(text)
        text = fc_xml.createTextNode("".join(map(lambda x: f"{x:24.15e}", gcell[2])))
        node.appendChild(text)
        node = fc_xml.createElement("UNIT_CELL_VOLUME_AU")
        geometry.appendChild(node)
        text = fc_xml.createTextNode(f"{avol:.12f}")
        node.appendChild(text)
        for i in range(ntyp):
            node = fc_xml.createElement(f"TYPE_NAME.{i+1}")
            geometry.appendChild(node)
            text = fc_xml.createTextNode(f"{symbols[i]}")
            node.appendChild(text)
            node = fc_xml.createElement(f"MASS.{i+1}")
            geometry.appendChild(node)
            text = fc_xml.createTextNode(f"{masses[i]:.12f}")
            node.appendChild(text)
        for i in range(natom):
            apos = scell.positions[scell.pmap[i], :] / lat
            node = fc_xml.createElement(f"ATOM.{i+1}")
            node.setAttribute("SPECIES", f"{symbols_full[i]}")
            node.setAttribute("INDEX", f"{typs[i]+1}")
            node.setAttribute("TAU", f"{apos[0]:.15e}{apos[1]:24.15e}{apos[2]:24.15e}")
            geometry.appendChild(node)
        node = fc_xml.createElement("NUMBER_OF_Q")
        geometry.appendChild(node)
        text = fc_xml.createTextNode(f"{nqpts}")
        node.appendChild(text)

        # DIELECTRIC_PROPERTIES
        dielectric_info = fc_xml.createElement("DIELECTRIC_PROPERTIES")
        root.appendChild(dielectric_info)
        if nac:
            dielectric_info.setAttribute("epsil", "true")
            dielectric_info.setAttribute("zstar", "true")
            dielectric_info.setAttribute("raman", "true")
            node = fc_xml.createElement("EPSILON")
            dielectric_info.appendChild(node)
            text = fc_xml.createTextNode(
                "".join(map(lambda x: f"{x:24.15e}", scell.epsilon[0]))
            )
            node.appendChild(text)
            text = fc_xml.createTextNode(
                "".join(map(lambda x: f"{x:24.15e}", scell.epsilon[1]))
            )
            node.appendChild(text)
            text = fc_xml.createTextNode(
                "".join(map(lambda x: f"{x:24.15e}", scell.epsilon[2]))
            )
            node.appendChild(text)
            node = fc_xml.createElement("ZSTAR")
            dielectric_info.appendChild(node)
            for i in range(natom):
                data_node = fc_xml.createElement(f"Z_AT_.{i+1}")
                node.appendChild(data_node)
                text = fc_xml.createTextNode(
                    "".join(map(lambda x: f"{x:24.15e}", scell.zeff[i, 0]))
                )
                data_node.appendChild(text)
                text = fc_xml.createTextNode(
                    "".join(map(lambda x: f"{x:24.15e}", scell.zeff[i, 1]))
                )
                data_node.appendChild(text)
                text = fc_xml.createTextNode(
                    "".join(map(lambda x: f"{x:24.15e}", scell.zeff[i, 2]))
                )
                data_node.appendChild(text)
        else:
            dielectric_info.setAttribute("epsil", "false")
            dielectric_info.setAttribute("zstar", "false")
            dielectric_info.setAttribute("raman", "false")

        # INTERATOMIC_FORCE_CONSTANTS
        fc_info = fc_xml.createElement("INTERATOMIC_FORCE_CONSTANTS")
        root.appendChild(fc_info)
        node = fc_xml.createElement("MESH_NQ1_NQ2_NQ3")
        fc_info.appendChild(node)
        text = fc_xml.createTextNode(f"\n\t{dim[0]:5d}{dim[1]:5d}{dim[2]:5d}\n      ")
        node.appendChild(text)
        for j, i in np.ndindex((natom, natom)):
            for k, l, m in np.ndindex(*dim[::-1]):
                icell = m + l * dim[1] + k * dim[0] * dim[1]
                ifc2_part = ifc2[i, j * ndim + icell]
                node = fc_xml.createElement(
                    f"s_s1_m1_m2_m3.{j+1}.{i+1}.{m+1}.{l+1}.{k+1}"
                )
                fc_info.appendChild(node)
                data_node = fc_xml.createElement("IFC")
                node.appendChild(data_node)
                text = fc_xml.createTextNode(
                    "".join(map(lambda x: f"{x:24.15e}", ifc2_part[0]))
                )
                data_node.appendChild(text)
                text = fc_xml.createTextNode(
                    "".join(map(lambda x: f"{x:24.15e}", ifc2_part[1]))
                )
                data_node.appendChild(text)
                text = fc_xml.createTextNode(
                    "".join(map(lambda x: f"{x:24.15e}", ifc2_part[2]))
                )
                data_node.appendChild(text)

        with open(filename + ".xml", "w") as fd:
            fc_xml.writexml(
                fd, indent="  ", addindent="  ", newl="\n", encoding="utf-8"
            )

    else:
        """Force constants in plain text"""

        with open(filename, "w") as fd:
            fd.write(
                f"{ntyp:>3d}{natom:3d}{ibrav:3d}{alat:12.7f}{zero:12.7f}"
                + f"{zero:12.7f}{zero:12.7f}{zero:12.7f}{zero:12.7f}\n"
            )
            fd.write("".join(map(lambda x: f"{x:16.9f}", acell[0])) + "\n")
            fd.write("".join(map(lambda x: f"{x:16.9f}", acell[1])) + "\n")
            fd.write("".join(map(lambda x: f"{x:16.9f}", acell[2])) + "\n")
            for i in range(ntyp):
                fd.write(f"\t{i+1:>5d}\t'{symbols[i]:3s}'\t{masses_amu[i]:16.10f}\n")
            for i in range(natom):
                apos = scell.positions[scell.pmap[i], :] / lat
                fd.write(
                    f"{i+1:>5d}{typs[i]+1:5d}{apos[0]:20.10f}{apos[1]:20.10f}{apos[2]:20.10f}\n"
                )
            if nac:
                fd.write(f"{'T':3s}\n")
                fd.write(
                    "\t"
                    + "".join(map(lambda x: f"{x:16.12f}", scell.epsilon[0]))
                    + "\n"
                )
                fd.write(
                    "\t"
                    + "".join(map(lambda x: f"{x:16.12f}", scell.epsilon[1]))
                    + "\n"
                )
                fd.write(
                    "\t"
                    + "".join(map(lambda x: f"{x:16.12f}", scell.epsilon[2]))
                    + "\n"
                )
                for i in range(natom):
                    fd.write(f"{i+1:>5d}\n")
                    fd.write(
                        "".join(map(lambda x: f"{x:12.7f}", scell.zeff[i, 0])) + "\n"
                    )
                    fd.write(
                        "".join(map(lambda x: f"{x:12.7f}", scell.zeff[i, 1])) + "\n"
                    )
                    fd.write(
                        "".join(map(lambda x: f"{x:12.7f}", scell.zeff[i, 2])) + "\n"
                    )
            else:
                fd.write(f"{'F':>3s}\n")
            fd.write(f"{dim[0]:>5d}{dim[1]:5d}{dim[2]:5d}\n")
            for k, l, i, j in np.ndindex((3, 3, natom, natom)):
                fd.write(f"{k+1:5d}{l+1:5d}{i+1:5d}{j+1:5d}\n")
                for k_dim, j_dim, i_dim in np.ndindex(*dim[::-1]):
                    icell = k_dim * dim[1] * dim[0] + j_dim * dim[0] + i_dim
                    ifc2_part = ifc2[j, i * ndim + icell, l, k]
                    fd.write(
                        f"{i_dim+1:>5d}{j_dim+1:5d}{k_dim+1:>5d}{ifc2_part:21.11e}\n"
                    )
