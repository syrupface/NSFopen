#!/usr/bin/env python
# coding: utf-8

# # Calculating surface roughness
# 
# This notebook levels an image of sapphire and calculates its roughness: the areal parameters of ISO 25178, the height
# distribution, the power spectrum, and the roughness along a single line. Run it from this folder, next to `sapphire.nid`.

# In[1]:


from NSFopen.process import process

import matplotlib.pyplot as plt


# In[2]:


afm = process("sapphire.nid")
afm.flatten(order=1)  # subtract a line from every scan line

fig = afm.image(color_range="p1")
plt.show()


# ## Surface parameters
# 
# `statistics()` returns a table with values and units. The values are in SI units. Sku follows ISO 25178, so it is 3 for
# a Gaussian surface.

# In[3]:


table = afm.statistics()
print(table)


# ## Height distribution
# 
# `hist` returns a normal matplotlib figure, so anything can be added to it, here a box with the main parameters.

# In[4]:


Sa, Sq, Ssk, Sku = table.loc[["Sa", "Sq", "Ssk", "Sku"], "value"]

text = "\n".join((f"RMS (Sq): {Sq * 1e12:.1f} pm", f"Mean (Sa): {Sa * 1e12:.1f} pm",
                  f"Skew (Ssk): {Ssk:.3f}", f"Kurtosis (Sku): {Sku:.3f}"))
print(text)

fig = afm.hist(bins=256, figsize=(6, 4))
fig.axes[0].text(0.025, 0.95, text, va="top", transform=fig.axes[0].transAxes, fontsize=8,
                 bbox=dict(boxstyle="round", facecolor="w", alpha=0.5))
plt.show()


# ## Bearing curve and power spectrum
# 
# The bearing curve shows how much of the surface lies above each height. The power spectral density shows at which
# lengths the roughness sits.

# In[5]:


fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", dpi=150)
afm.plot_bearing(ax=axes[0], title="Bearing curve")
afm.plot_psd(ax=axes[1], title="Power spectral density")
plt.show()


# ## Roughness along a line
# 
# A line cut, averaged over 5 pixels across the line, with its roughness after ISO 21920. End points are in m, as on the
# axes of the image.

# In[6]:


width, height = afm.size()
profile = afm.line_profile((0.1 * width, 0.5 * height), (0.9 * width, 0.5 * height), width=5)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained", dpi=150)
afm.image(ax=axes[0], profile=profile, color_range="p1")
afm.plot_profile(profile, ax=axes[1], title="Line cut")
plt.show()

roughness = afm.profile_roughness(profile).params
for name in ("Ra", "Rq", "Rz"):
    print(f"{name}: {roughness[name][0] * 1e12:.1f} pm")

