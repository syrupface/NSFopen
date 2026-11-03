#!/usr/bin/env python
# coding: utf-8

# # Batch leveling with a plane
# 
# This notebook levels every `.nid` file in this folder by subtracting a plane and shows each before and after. Run it
# from this folder. `process` finds the height channel whether the file calls it `Z-Axis` or `Topography`.

# In[1]:


from NSFopen.process import process

from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt

files = sorted(glob("*.nid"))
print(files)


# The right column limits the color scale to 3 standard deviations around the mean height, so that outliers do not set the
# contrast. This changes only the display, not the data.

# In[2]:


fig, axes = plt.subplots(len(files), 2, figsize=(9, 3.8 * len(files)), layout="constrained", dpi=150, squeeze=False)
fig.get_layout_engine().set(w_pad=0.15)  # room for the labels of the color bars

for (before, after), filename in zip(axes, files):
    name = Path(filename).stem
    afm = process(filename)
    afm.image(ax=before, title=name)

    afm.plane_level()

    z = afm.array()
    limits = (z.mean() - 1.5 * z.std(), z.mean() + 1.5 * z.std())
    afm.image(ax=after, title=f"{name} leveled", color_range=limits)

plt.show()

