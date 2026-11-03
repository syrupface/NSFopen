#!/usr/bin/env python
# coding: utf-8

# # Grain analysis
# 
# This notebook finds the circles in an image, measures each of them, and plots their size distribution. It uses
# `circles.nid` from the `Plane_Removal` folder; run it from this folder.

# In[1]:


from NSFopen.process import process

import matplotlib.pyplot as plt


# ## Level around the features
# 
# Features disturb leveling, so: level roughly, find the features with `mask_otsu`, which divides the heights into two
# levels, and then level on what is not masked. Finally mask once more on the clean image.

# In[2]:


afm = process("../Plane_Removal/circles.nid")
afm.level_flatten()
afm.mask_otsu()
afm.flatten(fit_on="exclude")
afm.mask_otsu()

fig = afm.image(show_mask=True, color_range="p1")
plt.show()


# ## Measure the grains
# 
# `grains` labels each connected part of the mask and returns a table with one row per grain, in SI units.
# `exclude_edge=True` leaves out grains cut by the edge of the image, whose size would be wrong.

# In[3]:


grains = afm.grains(exclude_edge=True)
table = grains.table
print(f"{len(table)} grains")
print(table[["Area", "Equivalent diameter", "Circularity", "Height above base"]].head())
print(grains.units["Equivalent diameter"], grains.units["Area"])


# ## Size distribution
# 
# The table is a pandas DataFrame, so it can be filtered, sorted or saved with `table.to_csv("grains.csv")`. "Height above
# base" uses the highest pixel of a grain, so a speck of dirt on top raises it, as on the two high points here. The median
# is not affected by them.

# In[4]:


diameter = table["Equivalent diameter"] * 1e6
height = table["Height above base"] * 1e9
print(f"Diameter: {diameter.mean():.2f} +- {diameter.std():.2f} um")
print(f"Height: median {height.median():.0f} nm")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained", dpi=150)
axes[0].hist(diameter, bins=8)
axes[0].set(xlabel="Equivalent diameter [µm]", ylabel="Grains")
axes[1].scatter(diameter, height)
axes[1].set(xlabel="Equivalent diameter [µm]", ylabel="Height above base [nm]")
plt.show()

