# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:04:53 2024

@author: nelson
"""

import os
import pathlib

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

GRADIENTS = pathlib.Path(__file__).with_name("gradients")


def read_gradient(path):
    """[(position, (r, g, b)), ...] of a Gwyddion gradient file."""
    lines = pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or not lines[0].startswith("Gwyddion resource GwyGradient"):
        raise ValueError(f"{path} is not a Gwyddion gradient file.")
    rows = [[float(v) for v in line.split()] for line in lines[1:] if line.strip()]
    return [(row[0], tuple(row[1:4])) for row in rows]


class colors:
    """Color maps for plots: Gwyddion's gradients and BlueBlackRed.

    Parameters
    ----------
        name : string
            Name of the color map, the result is in .cmap
        gwy_dir : string, default the gradients delivered with this package
            Folder with Gwyddion gradient files
    """

    def __init__(self, name=None, gwy_dir=None, **kwargs):
        self.name = name
        self.gwy_dir = pathlib.Path(gwy_dir) if gwy_dir else GRADIENTS
        custom = ["BlueBlackRed"]
        gwy_list = sorted(os.listdir(self.gwy_dir))

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
            print(sorted({x for v in cmaps.values() for x in v}))
            cmap = []
        self.cmap = cmap
        self.cmap_list = cmaps

    def gwy_cmap(self):
        return LinearSegmentedColormap.from_list(self.name, read_gradient(self.gwy_dir / self.name))

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
