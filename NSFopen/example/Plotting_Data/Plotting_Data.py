#!/usr/bin/env python
# coding: utf-8

# # Processing data and making figures
# 
# This notebook opens a Nanosurf image file, levels it, masks the features so they do not disturb the leveling, and makes
# figures for a report. Run it from this folder, next to `MoS2.nid`.

# In[1]:


from NSFopen.process import process

import matplotlib.pyplot as plt


# ## Open the file
# 
# `afm.channels` lists what the file holds. Every processing and plot function works on one channel, by default the
# forward height. `afm.array()` gives a channel as a plain array in SI units, for your own calculations.

# In[2]:


afm = process("MoS2.nid")
print(afm.channels)

z = afm.array()  # the forward height, in m
width, height = afm.size()  # in m
print(f"{z.shape[1]} x {z.shape[0]} pixels, {width * 1e6:.2f} x {height * 1e6:.2f} um, unit {afm.unit()}")


# ## Level the lines
# 
# `flatten` subtracts a straight line from every scan line. The unit, labels and color bar come from the file.
# `color_range='p1'` leaves the top and bottom 1 % of the pixels out of the color scale, so a few spikes cannot wash out
# the picture.

# In[3]:


afm.flatten(order=1)

fig = afm.image(color_range="p1")
plt.show()


# The bright islands pull the fit of every line they cross, which leaves dark bands beside them. To fix that, mask
# everything higher than 2 nm and fit the lines only to the rest. Heights are in m, so 2 nm is `2e-9`.

# In[4]:


afm.set_mask(afm.array() > 2e-9)
afm.flatten(order=1, fit_on="exclude")

fig = afm.image(color_range="p1", show_mask=True)
plt.show()


# ## Figures for a report
# 
# The same call with a scale bar and without a title, and a dark version. To keep a figure, use
# `fig.savefig("MoS2.png")`.

# In[5]:


fig = afm.image(color_range="p1", scalebar=True, title=None, figsize=(5, 4))
plt.show()

fig = afm.image(color_range="p1", scalebar=True, title=None, figsize=(5, 4), theme="dark", cmap="Gold")
plt.show()


# Pass `ax=` to put several plots into one figure. `layout='constrained'` keeps their labels apart.

# In[6]:


fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained", dpi=150)
afm.image(ax=axes[0], color_range="p1")
afm.hist(ax=axes[1], color_range="p1", bins=500)
axes[1].set_xlim(-2, 4)  # in nm; the few high spikes would otherwise stretch the axis to 25 nm
plt.show()


# ## 3D
# 
# `z_scale` sets how tall the height range looks. `plot3d` colors by height, or draws a `material` under a light;
# `rotation`, `tilt` and `light` change the view, see `help(afm.plot3d)`.

# In[7]:


fig = afm.plot3d(color_range="p1", z_scale=0.15)
plt.show()

fig = afm.plot3d(material="Gold", z_scale=0.15, rotation=-40, tilt=60)
plt.show()

