# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:04:53 2024

@author: nelson
"""

import numpy as np
import pandas as pd
import os

import colormaps


class colors:
    def __init__(self, name=None, gwy_dir=None, **kwargs):
        self.name = name
        self.gwy_dir = gwy_dir
        if not gwy_dir:
            self.gwy_dir = (
                "C:\\Program Files\\Gwyddion\\share\\gwyddion\\gradients\\"
            )
        custom = ["BlueBlackRed"]
        gwy_list = os.listdir(self.gwy_dir)

        cmaps = {}
        cmaps["Custom"] = custom
        cmaps["Gwyddion"] = gwy_list

        if name in gwy_list:
            cmap = self.gwy_cmap()
        elif name in custom:
            cmap = self.custom()
        else:
            print("Name Not Found\n")
            print("Use one of the following choices:\n")
            print({x for v in cmaps.values() for x in v})
            cmap = []
        self.cmap = cmap
        self.cmap_list = cmaps

    def gwy_cmap(self):
        cmap = pd.read_csv(
            self.gwy_dir + self.name,
            sep=" ",
            header=None,
            skiprows=1,
            index_col=0,
        )
        nodes = cmap.index.values
        colors = cmap.values

        return LinearSegmentedColormap.from_list(
            self.name, list(zip(nodes, colors))
        )

    def custom(self):
        if self.name == "BlueBlackRed":
            # Blue-Black-Red
            start = np.array(
                [0.6196078431372549, 0.792156862745098, 0.8823529411764706]
            )
            mid = np.array([0, 0, 0])
            end = np.array(
                [0.9882352941176471, 0.5725490196078431, 0.4470588235294118]
            )
            nodes = [0, 0.5, 1]

        return LinearSegmentedColormap.from_list(
            self.name, list(zip(nodes, [start, mid, end]))
        )
