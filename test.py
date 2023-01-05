from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from NSFopen.read import nhf_read as read


def flatten(data_in, order=1):
    data_out = np.array(data_in, dtype=float)
    x = np.arange(np.shape(data_in)[1])

    for idx, (out, line) in enumerate(zip(data_out, data_in)):
        ix = np.isfinite(line)
        p = np.polyfit(x[ix], line[ix], order)
        y = np.polyval(p, x)
        data_out[idx] = out - y
    
    return data_out
  

files = glob('*.nhf')

for file in files:

    afm = read(file)
    
    x = afm.prop['image_size_x']*1e6
    y = afm.prop['image_size_y']*1e6
    
    h=afm.data['Forward']['Position Z']*1e6    
    h=flatten(h)

    fig = plt.figure(dpi=300)
    plt.imshow(h, extent=[0,x,0,y],cmap='afmhot',origin='lower')
    plt.colorbar()
    plt.xlabel('X [$\mu$m]')
    plt.ylabel('Y [$\mu$m]')
    plt.show()