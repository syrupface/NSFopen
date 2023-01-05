from NSFopen.read import read as afmreader

import os
from os import listdir
from os.path import isfile, join

import numpy as np
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt

dirpath = os.getcwd()

# list all *.nid files
onlyfiles = [f for f in listdir(dirpath) if isfile(join(dirpath, f)) and f.split(".")[-1] == "nid"]

# list only files where there is "spectroscopy_" in the filename
force_files = [f for f in onlyfiles if "spectroscopy" in f.split("_")]

print(dirpath, "with", len(force_files), " *.nid force files:")
pprint(force_files)

# Read Data
filename = 'spectroscopy_0001.nid'
alldata = afmreader(filename, verbose=False).data

adhesion_data = alldata['Spec']['Backward']['Deflection']
zaxis = alldata['Spec']['Backward']['Z-Axis Sensor']

# Plot Data
font = {'size': 16}
matplotlib.rc('font', **font)

adhesion_force = []

for i in range(len(adhesion_data)):
    plt.plot(zaxis[i]*1e9+1250, adhesion_data[i]*1e9)
    adhesion_force.append(np.min(adhesion_data[i]*1e9))

plt.xlim([-100, 100])
plt.ylim([-15, 5])
plt.xticks(np.arange(-50, 150, step=50))

plt.xlabel('Z (nm)')
plt.ylabel('Force (nN)')

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(11, 10)
fig.savefig('adhesion.png', dpi=300)

plt.show()

# Plot Histogram
med = np.median(adhesion_force)
stdev = np.std(adhesion_force)
n, bins, patches = plt.hist(adhesion_force, 100, density=True, facecolor='g', alpha=0.75)
plt.title('\u00B5 = {:.1f} nN, \u03C3 = {:.2f} nN'.format(med, stdev))
plt.xlabel('Force (nN)')
plt.ylabel('Occurence')
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(11, 10)
fig.savefig('adhesion_dist.png', dpi=300)
plt.show()
