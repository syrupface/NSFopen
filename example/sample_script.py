# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 09:31:09 2019

@author: nelson
"""
from NSFopen import read

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# quick flattening of image
def flatten(data,order=1):
    data_out=np.copy(data)
    for idx,line in enumerate(data_out):
        x=np.arange(len(line))
        p=np.polyfit(x,line,order)
        y=np.polyval(p,x)
        data_out[idx]=line-y
    return data_out

file='image.nid'
file='M01232-Lattice-MFM-m8000000028.nid'

afm=read(file) # loads data in afm object
data=afm.data
param=afm.param
    
# IMAGE DATA
# get height data for example in nanometers
height=data.Image.Forward.ZAxis*1e9

# 2nd order flattening
height_flattened=flatten(height,order=2)

# get X (&Y) start and finish
X0=param.X['min'][0]*1e6
X1=param.X['range'][0]*1e6

extents=[X0,X1,X0,X1]


# plot image
im=plt.imshow(height_flattened,extent=extents,cmap=cm.afmhot)
plt.xlabel('[um]')
plt.ylabel('[um]')

# add a colorbar
cb=plt.colorbar(im)
cb.set_label('[nm]')
plt.show()


