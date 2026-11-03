#!/usr/bin/env python
# coding: utf-8

# # Lateral Force Processing Demo

# This notebook demonstrates how to open a Nanosurf image file (*.nid) containing lateral force data and perform basic post-processing operations

# In[1]:


# import required modules
from NSFopen.read import read

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


# In[2]:


# import the data
data_file = "SBS-PMMA.nid"
afm = read(data_file)
data = afm.data


# In[3]:


# load the data from the lateral channel
frict0 = data['Image']['Forward']['Friction force']
frict1 = data['Image']['Backward']['Friction force']


# We need to calculate both the sum and difference between the forward and backward channels.  Frictional forces will appear as contrast in the difference between the two channels.  Because of the length of the AFM tip, there will be an offset between the forward and backward channels, which we will have to subtract.

# In[4]:


offset = 5 # pixels

sum_ = (frict1[:, :-offset] + frict0[:, offset:])/2
diff = (frict1[:, :-offset] - frict0[:, offset:])/2


# In[5]:


# X and Y data is stored in the parameters
param = afm.param
extents = [param[i][j][0] * 1e6 for i in ['X','Y'] for j in ['min','range']]

Xmax = extents[1]

# calculate the new X-range with the offset removed
pixel_size = Xmax/np.shape(frict0)[0]
Xmax -= pixel_size * offset


# In[6]:


fig, ax = plt.subplots(1,2, figsize=(12,8), dpi=300)
fig.tight_layout(pad=6.0)

im = np.copy(ax)
im[0] = ax[0].imshow(sum_, extent=extents)
ax[0].set_title('Sum')
im[1] = ax[1].imshow(diff, extent=extents)
ax[1].set_title('Difference')

for i in range(2):
    ax[i].set(xlabel='X [$\mu$m]', ylabel='Y [$\mu$m]')
    ax_divider = make_axes_locatable(ax[i])
    cax = ax_divider.append_axes("right", size="5%", pad="2%")
    cb = plt.colorbar(im[i], cax=cax)
    cb.set_label('[V]')
plt.show()


# In[ ]:




