#!/usr/bin/env python
# coding: utf-8

# # Lateral force processing
# 
# This notebook opens an image with lateral force data and separates friction from topography effects. Friction turns the
# tip the opposite way in the forward and backward scans, so it shows in the difference of the two. What both have in
# common, such as slopes of the surface, shows in their mean. Run it from this folder, next to `SBS-PMMA.nid`.

# In[1]:


from NSFopen.process import process

import matplotlib.pyplot as plt


# In[2]:


afm = process("SBS-PMMA.nid")
print(afm.channels["Forward"])
channel = "Friction force"


# ## The offset between the scans
# 
# The forward and backward images lie a few pixels apart. It is best measured on the height, which should be the same in
# both scans. `find_shift` returns the shift in pixels and how well the images match, 1 being a perfect match.

# In[3]:


shift_x, shift_y, match = afm.find_shift(("Forward", "Z-Axis"), ("Backward", "Z-Axis"))
print(f"The backward scan lies {shift_x:.2f} pixels to the side and {shift_y:.2f} lines up, match {match:.2f}")


# ## Friction and mean
# 
# `combine` shifts the backward scan by that amount and stores each result as a new channel, with the unit of the
# original. Only the shift along the scan line is used. `half_diff` is (forward - backward) / 2. The outermost columns
# have no partner in the other scan after the shift, so ignore a few pixels at the left and right edges.

# In[4]:


forward, backward = ("Forward", channel), ("Backward", channel)
afm.combine(forward, backward, "half_diff", shift_x=shift_x, name="Friction")
afm.combine(forward, backward, "mean", shift_x=shift_x, name="Mean")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained", dpi=150)
afm.flatten()
afm.image(ax=axes[0], color_range="p1", title="Height")
afm.image(ax=axes[1], channel="Mean", cmap="viridis", color_range="p1", title="Mean of the two scans")
afm.image(ax=axes[2], channel="Friction", cmap="viridis", color_range="p1", title="Friction: half the difference")
plt.show()


# The new channels work like any other, for example for the distribution of friction, which separates the two polymers.

# In[5]:


fig = afm.plot_histogram_fit(channel="Friction", model="gauss2")
plt.show()

fit = afm.histogram_fit(channel="Friction", model="gauss2")
print(fit.summary(1e3, "mV"))

