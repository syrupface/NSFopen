# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:32:30 2025

@author: ed
"""

from .read import read


class nhf_spec:
    """Force curves of an .nhf spectroscopy file, one at a time.

    read(filename).spec gives the same data with more options.
    """

    def __init__(self, filename=None):
        self.filename = filename
        self._afm = read(filename)
        if self._afm.spec is None:
            raise ValueError(f"{filename} is not a spectroscopy file.")
        self.param = self._afm.param
        self.seg_names = self._afm.spec.segments
        self.chan_names = self._afm.spec.channels()

    def data(self, segment="Advance", channel="Deflection", N=0):
        return self._afm.spec[segment][channel][N]
