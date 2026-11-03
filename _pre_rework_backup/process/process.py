# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:13:47 2020

@author: Edward
"""

from NSFopen.read import read
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from scipy.ndimage import gaussian_filter, median_filter
from sklearn.impute import KNNImputer
from scipy import stats

import pandas as pd
import scipy.stats as ss
import os, sys


class process:
    def __init__(self, filename):

        afm = read(filename)
        self.data = afm.data
        self.param = afm.param

    def flatten(
        self,
        order=1,
        direction="Forward",
        channel="Z-Axis",
        mask=[],
        inplace=True,
    ):
        """Flattens an image using a nth-order polynomial

        Parameters
        ----------

            order : int, default 1
                Order of the polynomial
            direction : {'Forward','Backward'}, defulat 'Forward'
                Scan direction to use
            channel : string, default 'Z-Axis'
                Name of channel to use
            mask : boolean array
                Mask where True values are ignored in the input array
            inplace : boolean, default True

        Return
        ------
            array or None
                Flattened image or None if inplace=True

        """
        data_out = self.data["Image"][direction][channel]
        data_in = data_out.copy()
        data_in = np.array(data_in, dtype=float)
        x = np.arange(data_in.shape[1])

        if np.any(mask):
            data_in[mask] = np.nan
        for idx, (out, line) in enumerate(zip(data_out, data_in)):
            ix = np.isfinite(line)
            p = np.polyfit(x[ix], line[ix], order)
            y = np.polyval(p, x)
            data_out[idx] = out - y
        if inplace:
            self.data["Image"].loc[direction, channel] = data_out
            return None
        else:
            return data_out

    def plane_level(self, direction="Forward", channel="Z-Axis", inplace=True):
        data_out = self.data["Image"][direction][channel]
        data_in = data_out.copy()
        data_in = np.array(data_in, dtype=float)

        # x_size = self.param.X.range[0] * 1e6
        # y_size = self.param.Y.range[0] * 1e6

        # xs = np.linspace(0,x_size,data_in.shape[0])
        # ys = np.linspace(0,y_size,data_in.shape[1])

        xs = np.arange(0, data_in.shape[0])
        ys = np.arange(0, data_in.shape[1])
        xs, ys = np.meshgrid(xs, ys)
        ones = np.ones(xs.shape)

        reshape = lambda x: np.reshape(x, (len(x) ** 2, 1))

        A = np.matrix(np.hstack([reshape(xs), reshape(ys), reshape(ones)]))
        b = np.matrix(reshape(data_in))

        fit = (A.T * A).I * A.T * b
        plane = np.reshape(A * fit, data_in.shape)
        data_out -= plane
        if inplace:
            self.data["Image"].loc[direction, channel] = data_out
            return None
        else:
            return data_out

    def remove_outliers(
        self,
        direction="Forward",
        channel="Z-Axis",
        q=0.01,
        neighbors=5,
        inplace=True,
    ):
        data_out = self.data["Image"][direction][channel]
        data_in = data_out.copy()
        data_in = np.array(data_in, dtype=float)

        if isinstance(q, list):
            q0, q1 = q
        else:
            q0 = q1 = q

        value = np.quantile(data_in, q=[q0, 1 - q1])
        mask = (data_in < value[0]) | (data_in > value[1])
        data_in[mask] = np.nan

        data_out = KNNImputer(n_neighbors=neighbors).fit_transform(data_in)
        if inplace:
            self.data["Image"].loc[direction, channel] = data_out
            return None
        else:
            return data_out

    def denoise(
        self,
        direction="Forward",
        channel="Z-Axis",
        function="gaussian",
        size=5,
        inplace=True,
        **kwargs,
    ):
        """Applies a 2D filter to an image.

        Parameters
        ----------

            direction: {'Forward','Backward'}, default 'Forward'
                Scan direction to use
            channel: string, default 'Z-Axis'
                Name of channel to use
            function: {'gaussian','median'}, default 'gaussian'
                Name of filter to apply
            size: scalar
                Size of the filter
            inplace: boolean, default True
                Whether to modify the input array or create new one
            **kwargs:
                Passed to filters

        Return
        ------
            array or None
                Filtered image or None if inplace=True

        """
        data_in = self.data["Image"][direction][channel]
        if function == "median":
            data_out = median_filter(data_in, size=size, **kwargs)
        else:
            data_out = gaussian_filter(data_in, sigma=size, **kwargs)
        if inplace:
            self.data["Image"].loc[direction, channel] = data_out
            return None
        else:
            return data_out

    def image(
        self,
        direction="Forward",
        channel="Z-Axis",
        scale=1e9,
        cmap=cm.afmhot,
        cbar=None,
        fontsize=10,
        scalebar=None,
        theme="light",
        title="XXX",
        **kwargs,
    ):
        """Plots an AFM image

        Parameters
        ----------

            direction: {'Forward','Backward'}, default 'Forward'
                Scan direction to use
            channel: string, default 'Z-Axis'
                Name of channel to use
            scale: scalar, default 1e9
                Scale the data before plotting
            cmap: colormap from matplotlib, default afmhot
                Colormap to use in false color plot
            cbar: string, default None
                Add a colorbar to image with label
            fontsize: scalar, default 10
                Font size to use throughout
            scalebar: dict, {'length':scalar,'color':string}, default None
                Applies scalebar to bottom right of image instead of XY axis scale.
                * 'length' : length of scalebar in um
                * 'color' : color of scalebar and label
            theme: {'light','dark'}, default 'light'
                Use either 'light' or 'dark' theme
            **kwargs:
                Passed to matplotlib imshow

        Return
        ------
            figure
                matplotlib figure object

        """
        d = np.array(
            self.data["Image"][direction][channel] * scale, dtype=float
        )
        if hasattr(self.param, "X"):
            x = self.param.X.range[0] * 1e6
            y = self.param.Y.range[0] * 1e6
        elif hasattr(self.param, "image_size_x"):
            x = self.param.image_size_x * 1e6
            y = self.param.image_size_y * 1e6
        else:
            x, y = self.param.rect_axis_range
            x *= 1e6
            y *= 1e6

        fig = plt.figure(dpi=300)

        if theme == "light":
            color = "k"
        else:
            color = "w"
            fig.patch.set_facecolor("k")

        im = plt.imshow(
            d, extent=(0, x, 0, y), cmap=cmap, origin="lower", **kwargs
        )

        for spine in im.axes.spines.values():
            spine.set_edgecolor(color)

        if scalebar:
            if isinstance(scalebar, dict):
                dx = scalebar["length"]
                c = scalebar["color"]
            else:
                dx = np.round(x / 5)
                c = "k"

            pad = y * 0.05
            plt.plot([x - dx - pad, x - pad], [pad, pad], linewidth=5, color=c)
            txt = plt.text(
                x - pad - dx / 2,
                pad + x * 0.03,
                f"%i $\mu$m" % dx,
                horizontalalignment="center",
                fontsize=fontsize,
                color=c,
            )
            if dx < 1:
                txt.set_text(f"%i nm" % (dx * 1e3))
            plt.xticks([])
            plt.yticks([])
        else:
            plt.xlabel("X $[\mu{}$m]", fontsize=fontsize, color=color)
            plt.ylabel("Y $[\mu{}$m]", fontsize=fontsize, color=color)
            plt.xticks(fontsize=fontsize, color=color)
            plt.yticks(fontsize=fontsize, color=color)
            im.axes.tick_params(color=color, labelcolor=color)
        if title == "XXX":
            title = f"{channel} - {direction}"
        plt.title(title, fontsize=fontsize, color=color)
        if cbar:
            cb = plt.colorbar(pad=0.02)
            cb.set_label(label=cbar, size=fontsize, color=color)
            cb.ax.tick_params(labelsize=fontsize, color=color)
            cb.ax.yaxis.set_tick_params(color=color)
            cb.outline.set_edgecolor(color)
            plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=color)

        return fig

    def thumb(
        self,
        direction="Forward",
        channel="Z-Axis",
        cmap=cm.afmhot,
        **kwargs,
    ):
        """Plots an image to use as a Thumbnail

        Parameters
        ----------

            direction: {'Forward','Backward'}, default 'Forward'
                Scan direction to use
            channel: string, default 'Z-Axis'
                Name of channel to use
            cmap: colormap from matplotlib, default afmhot
                Colormap to use in false color plot

            **kwargs:
                Passed to matplotlib imshow

        Return
        ------
            figure
                matplotlib figure object

        """
        data = self.data["Image"][direction][channel]
        fig = plt.figure(figsize=(3, 3), dpi=150)
        plt.imshow(data, origin="lower", cmap=cm.afmhot)
        plt.axis("off")
        return fig

    def hist(
        self,
        direction="Forward",
        channel="Z-Axis",
        bins="sqrt",
        scale=1e9,
        cmap=cm.afmhot,
        xlabel="Z-Axis [nm]",
        fontsize=10,
        density=False,
        theme="light",
        **kwargs,
    ):

        data = np.array(
            self.data["Image"][direction][channel] * scale, dtype=float
        ).reshape(-1)
        if isinstance(bins, str):
            bins = n_bins(data, method=bins)

        fig = plt.figure(dpi=300)

        if theme == "light":
            color = "k"
            c_bg = "w"
        else:
            color = "w"
            c_bg = "k"
            fig.patch.set_facecolor(c_bg)

        n, edge, patches = plt.hist(
            data, bins=bins, density=density, color="green", **kwargs
        )
        bin_centers = 0.5 * (edge[:-1] + edge[1:])

        col = bin_centers - min(bin_centers)
        col /= max(col)

        for c, p in zip(col, patches):
            plt.setp(p, "facecolor", cmap(c))

        im = plt.gca()
        for spine in im.axes.spines.values():
            spine.set_edgecolor(color)

        plt.xlabel(xlabel, fontsize=fontsize, color=color)
        if density:
            ylabel = "$\\rho$"
        else:
            ylabel = "Counts"
        plt.ylabel(ylabel, fontsize=fontsize, color=color)
        plt.xticks(fontsize=fontsize, color=color)
        plt.yticks(fontsize=fontsize, color=color)
        im.axes.tick_params(color=color, labelcolor=color)

        im.get_children()[-1].set_facecolor(c_bg)

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
        # definition of central moment
        def Mu(x, i):
            return np.mean((x - x.mean()) ** i)

        # mean deviation
        def Sa_(x):
            return np.mean(np.abs(x - x.mean()))

        # RMS deviation
        def Sq_(x):
            return np.sqrt(Mu(x, 2))

        # Skew
        def skew_(x):
            return Mu(x, 3) / Mu(x, 2) ** (3.0 / 2.0)

        # Kurtosis
        def kurt_(x):
            return Mu(x, 4) / Mu(x, 2) ** 2 - 3

        # stat=[]
        func = [Sa_, Sq_, skew_, kurt_, np.min, np.max]
        fname = ["Sa", "Sq", "Ssk", "Sku", "Min", "Max"]
        stats = {}
        # cnames = []
        for d0, _ in self.data["Image"].groupby(level=0):
            s = {}
            for d1, df in self.data["Image"].groupby(level=1):
                # cnames.append([d0, d1])
                s[d1] = {s: f(df[d0][d1]) for s, f in zip(fname, func)}
            stats[d0] = s

        return toPandaDF(stats).unstack(level=0)


def toPandaDF(user_dict):
    df = pd.DataFrame.from_dict(
        {
            (i, j): user_dict[i][j]
            for i in user_dict.keys()
            for j in user_dict[i].keys()
        },
        orient="index",
    ).transpose()
    return df


def n_bins(x, method="sqrt"):
    n = len(x)
    if method == "freedman":  # freedman-diaconis rule
        h = 2 * stats.iqr(x) / np.cbrt(n)
        k = int(np.ceil((max(x) - min(x)) / h))
    elif method == "rice":
        k = int(np.ceil(2 * np.cbrt(n)))
    elif method == "sturges":
        k = int(np.ceil(np.log2(n)) + 1)
    elif method == "doane":
        g = stats.skew(x)
        sigma = np.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
        k = int(1 + np.log2(n) + np.log2(1 + np.abs(g) / sigma))
    elif method == "quantile":
        k = np.quantile(x, np.arange(0, 1.01, 0.025))
    else:
        k = int(np.ceil(np.sqrt(n)))
    return k


if __name__ == "__main__":
    file = "..\\example\\Calculate_Roughness\\sapphire.nid"
    from colormaps import colors

    cmap = colors("Sky").cmap

    # fig = plt.figure(dpi=300)
    afm = process(file)

    chan = "Z-Axis"

    afm.flatten(channel=chan)

    rms_roughness = afm.stats().Forward[chan].Sq
    z = afm.data.Image.Forward[chan]
    mask = z > 2 * rms_roughness
    afm.plane_level()
    afm.flatten(channel=chan, mask=mask)
    fig = afm.image(
        cbar="Z [nm]",
        title="",
        channel=chan,
        scalebar={"color": "w", "length": 0.25},
        scale=1e9,
        theme="dark",
        cmap=cmap,
    )
    plt.show()

    afm.hist(scale=1e12, xlabel="Height [pm]", theme="dark", cmap=cmap)

    print("Average Roughness: %3.2f pm" % (rms_roughness * 1e12))
