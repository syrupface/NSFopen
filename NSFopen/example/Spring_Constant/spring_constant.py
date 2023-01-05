# -*- coding: utf-8 -*-
"""
Created on Mon May 17 09:19:11 2021

@author: Edward
"""

import numpy as np
# from NSFopen.read import read
from read import read
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from sader import sader


def SHO(x, *args):
    f0, Q, A0, WN = args
    qterm = 1 / Q**2
    wnterm = WN**2
    # return np.sqrt((DC*f0/x)**2/((f0/x - x/f0)**2 + qterm) + wnterm)
    # return WN + DC/np.sqrt((1-(x/f0)**2)**2+(x/(f0*Q))**2)
    return WN + A0*f0**4/((x**2-f0**2)**2 + (x*f0/Q)**2)


file = 'thermal.nid'

# cantilever dimensions
L = 225e-6
W = 40e-6

afm = read(file)
amp = afm.data.Sweep.FFT['Amplitude Spectral Density'][0]

f_min = afm.param.X['min'][0]
f_range = afm.param.X.range[0]

freq = np.linspace(f_min, f_min + f_range, len(amp))

amp = amp[freq > 1000]
freq = freq[freq > 1000]


f0 = freq[np.where(amp == np.max(amp))[0][0]]
df = 15e3

# fit range
x = freq[(freq>f0-df) & (freq<f0+df)]
y = amp[(freq>f0-df) & (freq<f0+df)]

p0 = [f0, 500, 1e-13, 1e-13]
# bounds = ((100,0,-np.inf,-np.inf),(np.inf))
coeff, _ = curve_fit(SHO, x, y, p0=p0)

plt.plot(x/1e3, y*1e12, linewidth=3)
plt.plot(x/1e3, SHO(x, *coeff)*1e12, linewidth = 1, color = 'r', linestyle = '--')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Amp. Spec. Density (pm/$\sqrt{Hz}$)')
plt.legend(['Data', 'Fit'])


print(coeff)
print(sader(L, W, *coeff))
