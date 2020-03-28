#Note that for the script to function, the filenames should only have one "." right before the extention, i.e. "MFM_test_whatnot.nid".
#Place this Notebook with your data in the same folder.
#For MFM and EFM we are interested in looking at phase of a second scan.

#Import packages and find all the *.nid files.
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import math
#import scipy.io as sio # Only if you need smoothing/interpolation

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

font = {'size': 8}
matplotlib.rc('font', **font)

import collections
data_dict = collections.OrderedDict()
xaxis_dict = collections.OrderedDict()
yaxis_dict = collections.OrderedDict()

from NSFopen import read as afmreader

dirpath = os.getcwd()
onlyfiles = [f for f in listdir(dirpath) if isfile(join(dirpath, f)) and f.split(".")[1] == "nid"]
#print(dirpath, "with", len(onlyfiles), "*.nid files")

# Define what to plot
signal_to_plot = 'Phase' # 'Phase', 'Amplitude', 'ZAxis', 'ZAxisSensor' there could be more, i.e. 2nd lock-in signal, etc.
scan_direction = 'Forward' # 'Forward' or 'Backward'
scan_to_plot = 'Image2nd' # 'Image' or 'Image2nd'  - first or second scan

# Build X, Y axes and read data

for filename in onlyfiles:
    
    data = afmreader(filename).data[scan_to_plot][scan_direction][signal_to_plot]
    param = afmreader(filename).param
    xrange = param['X']['range'][0]
    yrange = param['Y']['range'][0]
    
    dict_key = filename.split(".nid")[0]
    data_dict[dict_key] = data
    
    xaxis = np.linspace(0, xrange*1E6, len(data[0]))
    xaxis_dict[dict_key] = xaxis
    
    yaxis = np.linspace(0, yrange*1E6, len(data))
    yaxis_dict[dict_key] = yaxis

# Prepare the figure and plot
force_limits = True # in case of glitches and spikes in the plots, make it False in the beginning

im = collections.OrderedDict()
axnum = 0
plot_columns = 4
plot_rows = math.ceil(len(data_dict)/plot_columns)
xlbl = 'X (um)'
ylbl = 'Y (um)'
fig, axarr = plt.subplots(plot_rows, plot_columns, figsize=(2*plot_columns,6*plot_rows))
fig.tight_layout(pad=5.0)

vmin = 345
vmax = 358
cmap = plt.get_cmap('jet')

for ax in axarr.flat:
    try: # In case the number of your files is not a multiple of 4
        dict_key = onlyfiles[axnum].split(".nid")[0]
        im[str(axnum)] = ax.pcolormesh(xaxis_dict[dict_key], yaxis_dict[dict_key], data_dict[dict_key], cmap = cmap)
        if force_limits:
            im[str(axnum)].set_clim(vmin = vmin, vmax = vmax)

        ax.set_title(dict_key[0:15]) # 15 characters from the filename
        ax.set(xlabel=xlbl, ylabel=ylbl)
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="5%", pad="2%")
        colorbar(im[str(axnum)], cax=cax)
        axnum += 1
    except:
        pass
fig.subplots_adjust(wspace = 0.6, hspace = 0.8)
plt.show()