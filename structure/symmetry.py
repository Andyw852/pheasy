"""Classes and functions related to symmetry operations."""
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2022 Changpeng Lin
# All rights reserved.

__all__ = [
    "Symmetrizer",
    "get_primitive_cell",
    "get_spacegroup",
    "get_symmetry",
    "get_ir_reciprocal_grid",
]


import numpy as np

try:
    import spglib as spg
except ImportError:
    try:
        import phonopy.structure.spglib as spg
    except ImportError:
        from pyspglib import spglib as spg


class Symmetrizer(object):
    """Symmetrizer for seveal physical quantities.

    Space group is used to symmetrize Born-effective charges
    with acoustic sum rules included.
    
    """

    pass


def get_primitive_cell(struct, symprec=1e-5, angle_tolerance=-1.0):
    """Find lattice vectors of primitive unit cell for a given structure.

    This is a wrapper for spglib.find_primitive to find the
    primitive unit cell.

    Parameters
    ----------
    struct : pheasy.Atoms
        Input structure.
    symprec : float
        Tolerance for finding primitive unit cell.
    angle_tolerance : float
        Tolerance for determing angle of cell.

    Returns
    -------
    A tuple of (lattice, positions, numbers).

    """
    struct_tuple = struct.to_spglib_tuple()
    unit_cell_tuple = spg.find_primitive(
        struct_tuple, symprec=symprec, angle_tolerance=angle_tolerance
    )
    if unit_cell_tuple is None:
        return None
    else:
        return unit_cell_tuple


def get_spacegroup(struct, symprec=1e-5, angle_tolerance=-1.0, symbol_type=0):
    """Return space group symbol and number as a string for a given structure.

    This is a wrapper for spglib.get_spacegroup.

    Parameters
    ----------
    struct : pheasy.Atoms
        Input structure.
    symprec : float
        Tolerance for determining symmetry.
    angle_tolerance : float
        Tolerance for determing angle of cell.
    symbol_type : int
        Space group symbol type.
        0 for international symbol.
        1 for schoenflies symbol.

    Returns
    -------
    A tuple of (symbol, number).

    """
    struct_tuple = struct.to_spglib_tuple()
    return spg.get_spacegroup(
        struct_tuple,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        symbol_type=symbol_type,
    )


def get_symmetry(
    struct, symprec=1e-5, angle_tolerance=-1.0, is_magnetic=False, mag_symprec=-1.0
):
    """Find and return symmetry operations.

    This is a wrapper for spglib.get_symmetry. The docstring here has been
    adapted from the spglib one, see:
    https://github.com/spglib/spglib/blob/develop/python/spglib/spglib.py

    Parameters
    ----------
    struct : pheasy.Atoms
        Input structure
    symprec : float
        Tolerance for determining symmetry.
    angle_tolerance : float
        Tolerance for determing angle of cell.
    is_magnetic : bool
        True to consider magnetic symmetry with the provided magmom data.
        In case of collinear calculation, an array with the shape (natoms,)
        should be provided. In case of non-collinear calculation, an array
        with the shape (natoms,3) should be provided.
    mag_symprec : float
        Tolerance for magnetic symmetry search in the unit of magmoms.
        If not specified, use the same value as symprec.
    

    Returns
    -------
    dict
        Rotation parts and translation parts of symmetry operations
        represented with respect to basis vectors and atom index
        mapping by symmetry operations.
        'rotations' : numpy.ndarray
            Rotation (matrix) parts of symmetry operations
            shape=(num_operations, 3, 3), order='C', dtype='intc'
        'translations' : numpy.ndarray
            Translation (vector) parts of symmetry operations
            shape=(num_operations, 3), dtype='double'
        'equivalent_atoms' : numpy.ndarray
            shape=(num_atoms, ), dtype='intc'
    
    """
    cell, scaled_positions, types = struct.to_spglib_tuple()
    if is_magnetic:
        # TODO
        magmoms = struct.get_initial_magnetic_moments()
        struct_tuple = (cell, scaled_positions, types, magmoms)
        return spg.get_magnetic_symmetry(
            struct_tuple,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            mag_symprec=mag_symprec,
            with_time_reversal=True,
        )
    else:
        struct_tuple = (cell, scaled_positions, types)
        return spg.get_symmetry(
            struct_tuple, symprec=symprec, angle_tolerance=angle_tolerance,
        )


def get_ir_reciprocal_grid(struct, grid_size, is_shift=[0, 0, 0]):
    """Generate irreducible k/q-points of the given uniform grid.

    This is a wrapper for spglib.get_ir_reciproca_mesh. The docstring 
    here has been adapted from the spglib one, see:
    https://github.com/spglib/spglib/blob/develop/python/spglib/spglib.py

    Parameters
    ----------
    struct : pheasy.Atoms
        Input structure
    grid_size : list or numpy.array
        Input uniform grid where irreducible k/q-points is searched.
        All of elements should be integer.
    is_shift : list or numpy.array, optional
        When is_shift is set for each reciprocal primitive axis, 
        the mesh is shifted along the axis in half of adjacent mesh 
        points irrespective of the mesh numbers. Defaults to [0,0,0].


    Returns:
        ir_grid : numpy.array
            An array of irreducible k/q-points in scaled positions,
            ranging from [-0.5, 0.5) and the shape is (nipts, 3).
        weights : numpy.array
            The weight of each irreducible points.

    """
    struct_tuple = struct.to_spglib_tuple()

    mapping, ir_grid = spg.get_ir_reciprocal_mesh(
        grid_size, struct_tuple, is_shift=is_shift
    )
    ir_inds = np.unique(mapping)

    weights = np.zeros_like(mapping)
    for pt in mapping:
        weights[pt] += 1
    ir_grid = np.array(ir_grid) / grid_size
    ir_grid = np.where(ir_grid >= 0.5, ir_grid - 1.0, ir_grid)

    return ir_grid[ir_inds], weights[ir_inds]
