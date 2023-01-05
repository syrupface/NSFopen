#!/usr/bin/env python
# coding: utf-8

# # Lateral Force Processing Demo

# This notebook demonstrates how to open a Nanosurf image file (*.nid) containing lateral force data and perform basic post-processing operations

# In[1]:


# import required modules
from NSFopen.read import read

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


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

sum = (frict1[:, :-offset] + frict0[:, offset:])/2
diff = (frict1[:, :-offset] - frict0[:, offset:])/2


# In[5]:


# X and Y data is stored in the parameters
param = afm.param
Xmin = param.X['min'][0] * 1e6  # the minimum of X of channel 0 scaled to um
Xmax = param.X['range'][0] * 1e6  # the range of X of channel 0 scaled to um

Ymin = param.Y['min'][0] * 1e6  # the minimum of Y for channel 0 scaled to um
Ymax = param.Y['range'][0] * 1e6  # the range of Y for channel 0 scaled to um

# calculate the new X-range with the offset removed
pixel_size = Xmax/np.shape(frict0)[0]
Xmax -= pixel_size * offset

extents = [Xmin, Xmax, Ymin, Ymax]


# In[6]:


fig, ax = plt.subplots(1,2, figsize=(20,10))
ax[0].imshow(sum, extent=extents)
ax[0].set_title('Sum')
ax[1].imshow(diff, extent=extents)
ax[1].set_title('Difference')

[ax[i].set_xlabel('[$\mu$m]') for i in range(2)];
[ax[i].set_ylabel('[$\mu$m]') for i in range(2)];
plt.show()


# In[ ]:




