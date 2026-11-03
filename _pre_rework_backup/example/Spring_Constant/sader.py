# -*- coding: utf-8 -*-
"""
Created on Mon May 17 17:00:19 2021

@author: Edward
"""

import numpy as np
from scipy.special import kv as besselk


def sader(L, W, *args):
    RHO = 1.18
    ETA = 1.86e-5

    def Ren(f, b):
        return RHO * f * b**2 / (4 * ETA)

    def Gr(z):
        G = (0.91324 - 0.48274 * z + 0.46842 * z**2
             - 0.12886 * z**3 + 0.044055 * z**4 - 0.0035117 * z**5
             + 0.00069085 * z**6) / (1 - 0.56964 * z + 0.48690*z**2 - 0.13444 * z**3
             + 0.045155 * z**4 - 0.0035862 * z**5 + 0.00069085 * z**6)
        return G

    def Gi(z):
        G = (-0.024134 - 0.029256 * z + 0.016294 * z**2
             - 0.00010961 * z**3 + 0.000064577 * z**4
             - 0.00004451 * z**5) / (1 - 0.59702 * z
             + 0.55182 * z**2 - 0.18357 * z**3
             + 0.079156 * z**4 - 0.014369 * z**5 + 0.0028361 * z**6)
        return G

    def Cfun(z):
        return Gr(z) + 1j * Gi(z)

    def Gcirc(f,b):
        G = (1 + (4 * 1j * besselk(1, -1j * np.sqrt(1j * Ren(f, b)))) /
             (np.sqrt(1j * Ren(f, b)) * besselk(0, -1j * np.sqrt(1j * Ren(f, b)))))
        return G

    def Grect(f, b):
        G = Cfun(np.log10(Ren(f, b))) * Gcirc(f, b)
        return G
   
    result = 0.1906 * (2 * np.pi * args[0])**2 * \
             RHO * W**2 * L * args[1] * \
             np.imag(Grect(2 * np.pi * args[0], W))
    return result
