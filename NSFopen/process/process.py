# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:13:47 2020

@author: Edward
"""

from NSFopen.read import nid_read, nhf_read
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import pandas as pd
import scipy.stats as ss
import os, sys


class process():
    def __init__(self, filename):
        
        _,ext = os.path.splitext(filename)
        if ext == ".nid":
            afm = nid_read(filename, verbose=False)
        elif ext == ".nhf":
            afm = nhf_read(filename)
        else:
            print("Not a valid filetype")
            sys.exit()
        self.data = afm.data
        self.param = afm.param

    def flatten(self, order=1, direction='Forward', channel='Z-Axis', mask=[]):
        data_out = self.data['Image'][direction][channel]
        data_in = data_out.copy()
        data_in = np.array(data_in, dtype=float)
        x = np.arange(np.shape(data_in)[1])

        if np.any(mask):
            data_in[mask] = np.nan
        for idx, (out, line) in enumerate(zip(data_out, data_in)):
            ix = np.isfinite(line)
            p = np.polyfit(x[ix], line[ix], order)
            y = np.polyval(p, x)
            data_out[idx] = out - y
        self.data['Image'][direction][channel] = data_out

    def image(self,
              direction='Forward',
              channel='Z-Axis',
              scale=1e9,
              cmap=cm.afmhot,
              cbar=[],
              fontsize=10,
              scalebar=False,
              theme='light'):

        d = np.array(self.data['Image'][direction][channel]*scale,dtype=float)
        if hasattr(afm.param, 'X'):
            x = self.param.X.range[0] * 1e6
            y = self.param.Y.range[0] * 1e6
        else:
            x = self.param.image_size_x * 1e6
            y = self.param.image_size_y * 1e6
            
        fig = plt.figure()

        if theme=='light':
            color = 'k'
        else:
            color= 'w'
            fig.patch.set_facecolor('k')

        im = plt.imshow(d, extent=(0, x, 0, y), cmap=cmap, origin='lower')

        for spine in im.axes.spines.values():
            spine.set_edgecolor(color)   

        if scalebar:
            if isinstance(scalebar,dict):
                dx = scalebar['length']
                c = scalebar['color']
            else:
                dx = np.round(x / 5)
                c = 'k'

            pad = y * 0.05
            plt.plot([x - dx - pad, x - pad], [pad, pad], linewidth=5, color=c)
            plt.text(x - pad - dx / 2, pad + x * 0.03, f'%i $\mu$m' % dx,
                     horizontalalignment='center',
                     fontsize=fontsize,
                     color=c)
            plt.xticks([])
            plt.yticks([])
        else:
            plt.xlabel('$[\mu{}$m]', fontsize=fontsize, color=color)
            plt.ylabel('$[\mu{}$m]', fontsize=fontsize, color=color)
            plt.xticks(fontsize=fontsize, color=color)
            plt.yticks(fontsize=fontsize, color=color)
            im.axes.tick_params(color=color, labelcolor=color)
        plt.title(f'{channel} - {direction}',fontsize=fontsize, color=color)
        if cbar:
            cb = plt.colorbar(pad=0.02)
            cb.set_label(label=cbar,size=fontsize, color=color)
            cb.ax.tick_params(labelsize=fontsize, color=color)
            cb.ax.yaxis.set_tick_params(color=color)
            cb.outline.set_edgecolor(color)
            plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=color)
        
        return fig

    # def surf(self,
    #          direction='Forward',
    #          channel='Z-Axis',
    #          scale=1e9,
    #          cmap = cm.afmhot,
    #          cbar = [],
    #          fontsize = 10):

    #     Z = self.data['Image'][direction][channel]*scale
    #     xx = self.param.X.range[0] * 1e6
    #     yy = self.param.Y.range[0] * 1e6
    #     X = np.linspace(0,xx,np.shape(Z)[0])
    #     Y = np.linspace(0,yy,np.shape(Z)[1])
    #     X, Y = np.meshgrid(X,Y)

    #     fig = plt.figure()
    #     ax = fig.gca(projection='3d')
    #     ax.set_proj_type('ortho')
    #     surf = ax.plot_surface(X, Y, Z,
    #                            cmap=cmap,
    #                            linewidth=0,
    #                            antialiased=False)

    #     ax.set_xlabel('$[\mu{}$m]', fontsize=fontsize)
    #     ax.set_ylabel('$[\mu{}$m]', fontsize=fontsize)
    #     ax.view_init(elev=45., azim=32)
    #     ax.set_frame_on(False)
    #     ax.grid(False)
    #     ax.xaxis.pane.fill=False
    #     ax.yaxis.pane.fill=False
    #     ax.zaxis.pane.fill=False
    #     if cbar:
    #         cb = plt.colorbar(surf,pad=0.02, shrink=0.5, aspect=10)
    #         cb.set_label(label=cbar,size=fontsize)
    #         cb.ax.tick_params(labelsize=fontsize)
    #         ax.set_zticks([])

    #     else:
    #         ax.set_zlabel('',fontsize=fontsize)

    def stats(self):
        def mad(x):
            return np.mean(np.abs(x-x.mean()))

        # stat=[]
        names = []
        s0, s1, s2, s3 = [], [], [], []
        for d, df in self.data['Image'].groupby(level=0):

            names.append(d)
            s0.append(df[d].apply(lambda x: mad(x)))
            s1.append(df[d].apply(np.std))
            s2.append(df[d].apply(lambda x: ss.skew(x, axis=None)))
            s3.append(df[d].apply(lambda x: ss.kurtosis(x, axis=None)))

        snames = ['Sa','Sq','Skew','Kurtosis']
        s = [s0, s1, s2, s3]
        return pd.DataFrame(data=s,columns=names,index=snames)


if __name__ == "__main__":
    file='..\\example\\image.nhf'
    
    # fig = plt.figure(dpi=300)
    afm = process(file)

    chan = 'Position Z'
    afm.flatten(channel=chan)
    # plt.clim(-100,50)
    afm.image(cbar='Z [nm]', channel=chan, scalebar=False, scale=1e9, theme='light')
    plt.show()
    
    # fig = plt.figure(dpi=300)
    rms = afm.stats().Forward.Sa[chan]
    z = afm.data.Image.Forward[chan]
    mask = z > 2*rms
    afm.flatten(channel=chan, mask=mask)
    fig = afm.image(cbar='Z [nm]', channel=chan, scalebar=False, scale=1e9, theme='dark')
    fig.set_dpi(300)
    # plt.clim(-100,50)
    plt.show()
    

    print(rms)
