#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert a displacement/force dataset into the pickles pheasy expects.

pheasy's ``--disp_file`` path reads ``disp_matrix.pkl`` (Cartesian
displacements, shape ``(ndata, natoms, 3)``, Angstrom) and ``force_matrix.pkl``
(forces, same shape, eV/A). This script builds both from a pair of .npy files.

The displacement array may hold either

  * Cartesian displacements already, or
  * **fractional coordinates** of each configuration -- in which case pass
    ``--frac``; the reference frame (``--ref``, default the last one) is
    subtracted, minimum-image wrapped and converted with the supercell lattice.

The corresponding residual forces of the reference frame are subtracted from
every configuration unless ``--no-subtract-residual`` is given.

Usage
-----
    python3 tools/prepare_dataset.py SPOSCAR dataset_disps.npy dataset_forces.npy --frac
"""
import argparse
import pickle

import numpy as np


def read_lattice(sposcar):
    lines = open(sposcar).read().splitlines()
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    return cell * scale


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sposcar")
    ap.add_argument("disps_npy")
    ap.add_argument("forces_npy")
    ap.add_argument("--frac", action="store_true",
                    help="input holds fractional coordinates, not displacements")
    ap.add_argument("--ref", type=int, default=-1,
                    help="index of the undisplaced reference frame (default: last)")
    ap.add_argument("--no-subtract-residual", action="store_true")
    ap.add_argument("--disp-out", default="disp_matrix.pkl")
    ap.add_argument("--force-out", default="force_matrix.pkl")
    args = ap.parse_args()

    d = np.load(args.disps_npy)
    f = np.load(args.forces_npy)
    if d.shape != f.shape or d.ndim != 3 or d.shape[2] != 3:
        raise SystemExit("expected matching (ndata, natoms, 3) arrays, got "
                         "%s and %s" % (d.shape, f.shape))

    if args.frac:
        cell = read_lattice(args.sposcar)
        ref = d[args.ref]
        u = d - ref[None]
        u = u - np.round(u)                      # minimum image
        keep = [i for i in range(d.shape[0]) if i != args.ref % d.shape[0]]
        U = u[keep] @ cell
        F = f[keep]
        if not args.no_subtract_residual:
            F = F - f[args.ref][None]
    else:
        U, F = d, f
        if not args.no_subtract_residual:
            U = np.delete(U, args.ref % d.shape[0], axis=0)
            F = np.delete(F, args.ref % d.shape[0], axis=0) - f[args.ref][None]

    print("displacements %s  rms=%.4f A  max=%.4f A" %
          (U.shape, np.sqrt((U ** 2).mean()), np.abs(U).max()))
    print("forces        %s  max=%.4f eV/A" % (F.shape, np.abs(F).max()))
    if np.abs(U).max() > 0.5:
        print("WARNING: displacements above 0.5 A -- did you mean --frac?")

    with open(args.disp_out, "wb") as fh:
        pickle.dump(np.ascontiguousarray(U, dtype=np.float64), fh)
    with open(args.force_out, "wb") as fh:
        pickle.dump(np.ascontiguousarray(F, dtype=np.float64), fh)
    with open("ndata_total.txt", "w") as fh:
        fh.write(str(U.shape[0]))
    print("wrote %s, %s, ndata_total.txt (--ndata %d)"
          % (args.disp_out, args.force_out, U.shape[0]))


if __name__ == "__main__":
    main()
