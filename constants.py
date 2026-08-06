"""Contants, adapted from scipy.constants"""
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2023 Changpeng Lin
# All rights reserved.

import scipy.constants as const


"""Physical quantities"""
N_A = const.N_A  # Avogadro constant

"""Atomic Rydberg units"""
MASS_RY = const.electron_mass * 2.0

"""Unit conversion"""
ANGSTROM_TO_BOHR = const.angstrom / const.physical_constants["Bohr radius"][0]
KG_TO_G = 1000.0
MASS_RY_TO_QE = MASS_RY * N_A * KG_TO_G
RY_TO_EV = const.physical_constants["Rydberg constant times hc in eV"][0]
