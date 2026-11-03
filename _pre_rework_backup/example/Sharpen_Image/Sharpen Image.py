#!/usr/bin/env python
# coding: utf-8

# #### Import packages

# In[1]:


from NSFopen.read import read

import numpy as np
import collections

import cv2 as cv  # aka opencv-python
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

font = {'size': 12}
matplotlib.rc('font', **font)


# #### Define functions

# In[2]:


def fittingFunc(matrix, xaxis, order):
    newmatrix = np.zeros(np.shape(matrix))
    for i in range(len(matrix)):
        p = np.polyfit(xaxis, matrix[i,:], order)
        f = np.poly1d(p)
        fit_array = f(xaxis)
        newmatrix[i, :] = matrix[i, :] - fit_array 
    return newmatrix

def unsharp_mask(image, kernel_size=5, sigma=1.0, amount=1.0, threshold=0):
    if kernel_size%2 != 1:
        kernel_size += 1  
    blurred = cv.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def laplacian_sharp(image, ddepth = cv.CV_16S, kernel_size=3, amount = 0.5):
    edges = cv.Laplacian(image, ddepth, ksize=kernel_size)
    sharpened = (1-amount)*image - amount*edges
    armax = np.max(sharpened)
    armin = np.min(sharpened)
    sharpened = (sharpened - armin)/(armax-armin)*255
    return sharpened


# #### Set filename and define what to plot

# In[3]:


filename = 'grid.nid'

signals = ['Topography','Z-Axis','Z-Axis Sensor'] # 'Amplitude', there could be more, i.e. 2nd lock-in signal, etc.
scan_direction = 'Forward' # 'Forward', 'Backward', '2nd Forward', '2nd Backward'
scan_to_plot = 'Image' # 'Image', 'Spec' or 'Sweep'
fitting_order = 1 # order of the fitting polynom
fitting = True # Use line fit or not


# #### Build X, Y axes and read data

# In[4]:


afm = read(filename, verbose = False)
data = afm.data

available_channels = list(data[scan_to_plot][scan_direction].index)
signal_to_plot = [i for i in available_channels if i in signals][0]

rawdata = data[scan_to_plot][scan_direction][signal_to_plot]*1e9 # nm

param = afm.param
xrange = param['X']['range'][0]
yrange = param['Y']['range'][0]

dict_key = filename.split(".nid")[0]
xaxis = np.linspace(0, xrange*1e6, len(rawdata[0]))
yaxis = np.linspace(0, yrange*1e6, len(rawdata))

if fitting:
    data = fittingFunc(rawdata, xaxis, fitting_order)
else:
    data = rawdata


# #### Prepare the figure and plot

# In[5]:


force_limits = True # in case of glitches and spikes in the plots, make it False in the beginning

xlbl = 'X [$\mu$m]'
ylbl = 'Y [$\mu$um]'
fig, axarr = plt.subplots(1, 1, figsize=(7.5,7), dpi=300)

vmin = -40
vmax = 40
cmap = 'afmhot'

im = axarr.pcolormesh(xaxis, yaxis, data, cmap = cmap, shading='auto')
if force_limits:
    im.set_clim(vmin = vmin, vmax = vmax)

axarr.set_title(filename.split(".nid")[0])
axarr.set(xlabel=xlbl, ylabel=ylbl)
ax_divider = make_axes_locatable(axarr)
cax = ax_divider.append_axes("right", size="5%", pad="2%")
p = plt.colorbar(im, cax=cax)
p.ax.set_ylabel("Z (nm)")
plt.show()


# #### Change data to 0 - 255

# In[6]:


for i in range(len(data)):
    for j in range(len(data[0])):
        if data[i,j] < vmin:
            data[i,j] = vmin
        if data[i,j] > vmax:
            data[i,j] = vmax
            
spread = 1/(vmax - vmin)
imdata = (data-vmin)*spread*255
imdata_new = np.uint8(imdata)


# #### Sharpen

# In[7]:


img = Image.fromarray(imdata_new, 'L')
img.save('img.jpg')

img = cv.imread('img.jpg', cv.IMREAD_GRAYSCALE) # Save and read img file, because cv2.imdecode is producing an error

img_sharp_g = unsharp_mask(img, kernel_size=7, sigma=1.0, amount=25.0, threshold=0.5)
img_sharp_l = laplacian_sharp(img, ddepth = cv.CV_16S, kernel_size=3, amount = 0.6)

img_sharp_g = img_sharp_g/spread/255 + vmin # Bring back to the physical range
img_sharp_l = img_sharp_l/spread/255 + vmin # Bring back to the physical range


# #### Plot original and sharpened together

# In[8]:


im_im = collections.OrderedDict()
axnum = 0
fig, axarr = plt.subplots(1, 3, figsize=(14,5), dpi=300)
fig.tight_layout(pad=5.0)

for ax in axarr:
    index = str(axnum)
    if axnum == 0:
        im_im[str(axnum)] = ax.pcolormesh(xaxis, yaxis, data, cmap = cmap, shading = 'auto')
        ax.set_title(filename.split(".nid")[0])
        ax.set(xlabel=xlbl, ylabel=ylbl)
        im_im[str(axnum)].set_clim(vmin = vmin, vmax = vmax)
    elif axnum == 1:
        im_im[str(axnum)] = ax.pcolormesh(xaxis, yaxis, img_sharp_g, cmap = cmap, shading = 'auto')
        ax.set_title(filename.split(".nid")[0] + " unsharp")
        ax.set(xlabel=xlbl, ylabel='')
        im_im[str(axnum)].set_clim(vmin = vmin, vmax = vmax)
    else:
        im_im[str(axnum)] = ax.pcolormesh(xaxis, yaxis, img_sharp_l, cmap = cmap, shading = 'auto')
        ax.set_title(filename.split(".nid")[0] + " laplacian")
        ax.set(xlabel=xlbl, ylabel='')
        im_im[str(axnum)].set_clim(vmin = vmin/3, vmax = vmax/3)
    
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="5%", pad="2%")
    p = plt.colorbar(im_im[str(axnum)], cax=cax)
    if axnum == 2:
        p.ax.set_ylabel("Z (nm)")
    axnum += 1
plt.show()

