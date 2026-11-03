# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:32:30 2025

@author: ed
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nanosurf as nsf


class nhf_spec:
    def __init__(self, filename=None):
        self.filename = filename
        self.param = None
        self.seg_names = None
        self.chan_names = None
        self._read()

    def _read(self):
        nhf_file = nsf.util.nhf_reader.NHFFileReader(verbose=False)
        ok = nhf_file.read(self.filename)

        image_name = ", ".join(nhf_file.measurement)

        param = nhf_file.measurement[image_name].attribute
        meas_type = param["measurement_type"]

        # size_x = nhf_file.measurement[image_name].dataset_size_x
        # size_y = nhf_file.measurement[image_name].dataset_size_y

        # len_x = nhf_file.measurement[image_name].dataset_range_x * 1e6
        # len_y = nhf_file.measurement[image_name].dataset_range_y * 1e6

        defl_sens = nhf_file.measurement[image_name].attribute[
            "spm_probe_calibration_deflection_sensitivity"
        ]

        spring_constant = nhf_file.measurement[image_name].attribute[
            "spm_probe_calibration_spring_constant"
        ]

        param["probe_k"] = spring_constant
        param["probe_sens"] = defl_sens

        segments = nhf_file.measurement[image_name].segment

        self.seg_names = list(segments.keys())
        self._segments = segments

        special_channels = [
            "channel_data_offsets",
            "number_of_datapoints_acquired",
            "deflection_min",
            "deflection_max",
        ]

        for seg in segments.keys():
            meas = segments[seg]

            chans = meas.channel.keys()
            data_chans = [x for x in chans if x not in special_channels]

            self.chan_names = list(data_chans)

        data_number = len(meas.read_channel("channel_data_offsets").dataset)
        param["number"] = data_number
        self.param = pd.Series(param)

    def data(self, segment="Advance", channel="Deflection", N=0):
        data = self._segments[segment].read_channel(channel).dataset
        offset = (
            self._segments[segment]
            .read_channel("channel_data_offsets")
            .dataset
        )
        num = (
            self._segments[segment]
            .read_channel("number_of_datapoints_acquired")
            .dataset
        )
        data_out = data[int(offset[N]) : int(offset[N]) + int(num[N])]
        return data_out


if __name__ == "__main__":
    """
    CHANGE THE FILE NAME
    """

    file = "spec_test.nhf"

    afm = nhf_spec(file)

    # list segment names
    print(afm.seg_names)

    # list channel names
    print(afm.chan_names)

    # how many force curves in dataset?
    print(r"Number: %i" % afm.param["number"])

    # get deflection (in volts) and Z-position data for 10th retract curve
    num = 9  # python indexes at 0 so 10 -> 9
    d = afm.data(segment="Retract", channel="Deflection", N=num)
    z = afm.data(segment="Retract", channel="Position Z", N=num)

    # convert deflection to force
    f = d * afm.param.probe_k * afm.param.probe_sens

    # plot 10th force curve
    plt.plot(z, f)
