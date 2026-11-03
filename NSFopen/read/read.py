# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:17:03 2018

Copyright (c) 2018 nelson<at>nanosurf.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


import os
import re
import time

import numpy as np
import pandas as pd

from .nhf import NHFFile, Spectroscopy, WaveModeNMA


def _header_text(line):
    """A line of a NID header: UTF-8, in files of old versions possibly Latin-1."""
    try:
        return line.decode("utf-8")
    except UnicodeDecodeError:
        return line.decode("iso-8859-1")


class read:
    """Opens a Nanosurf .nid or .nhf file.

    Attributes
    ----------
        data : pandas Series
            data["Image"]["Forward"]["Z-Axis"] is an image, data["Spec"]["Retract"]["Deflection"]
            holds force curves. In an .nhf file the curves are read on access, see CurveChannel.
        param : pandas Series
            The parameters of the file
        units : dict
            Unit of every channel, by (kind, segment, channel)
        spec, nma : Spectroscopy, WaveModeNMA or None
            Force curves of an .nhf spectroscopy or WaveMode NMA file
    """

    def __init__(self, filename=None, **kwargs):
        self.filename = filename
        self.data = None
        self.param = None
        self.units = {}
        self.spec = None
        self.nma = None
        self._reader = None

        if not filename:
            print("\nFile required!\n")
            return

        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        if ext == ".nid":
            self._reader = nid_read(filename=filename, **kwargs)
        elif ext == ".nhf":
            self._reader = nhf_read(filename, **kwargs)
            self.spec = self._reader.spec
            self.nma = self._reader.nma
        else:
            print("Not a valid filetype")
            return
        self.data = self._reader.data
        self.param = self._reader.param
        self.units = self._reader.units
        self.image_size = getattr(self._reader, "image_size", None)

    def close(self):
        """Releases an .nhf file; its curves can no longer be read afterwards."""
        if hasattr(self._reader, "close"):
            self._reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# frame of an .nid file -> where it goes in the data
_NID_FRAMES = {
    "Spec forward": ("Spec", "Forward"),
    "Spec backward": ("Spec", "Backward"),
    "Spec fwd pause": ("Spec", "Pause Forward"),
    "Spec bwd pause": ("Spec", "Pause Backward"),
    "Scan forward": ("Image", "Forward"),
    "Scan backward": ("Image", "Backward"),
    "2nd scan forward": ("Image", "2nd Forward"),
    "2nd scan backward": ("Image", "2nd Backward"),
    "Frequency sweep": ("Sweep", "Sweep"),
    "Frequency sweep SHO": ("Sweep", "SHO"),
    "Spectrum FFT": ("Sweep", "FFT"),
    "Spectrum Fit": ("Sweep", "Fit"),
}


class nid_read:
    def __init__(self, filename=None, dataframe=True, verbose=False):
        self.filename = filename
        self.data = None
        self.param = None

        if not filename:
            print("\nFile required!\n")
            return

        self.__dataframe = dataframe
        self.verbose = verbose
        # names of the channels that can be opened.  Can be added to.
        self.data_names = list(_NID_FRAMES)

        self.read()

    def read(self):
        t = time.time()

        # isolates header from data file
        if self.verbose:
            print("Reading Header")

        with open(self.filename, "rb") as fid:
            raw = fid.read()

        head = np.array(raw.split(b"\n\r\n\r", 1)[0].split(b"\r\n"))
        boo = head == b""
        indices = np.nonzero(boo[1:] != boo[:-1])[0] + 1
        head = np.split(head, indices)

        head = [h for h in head if len(h) > 1]

        header = {}
        for h in head:
            title = _header_text(h[0])[1:-1]
            data = h[1:]
            values = {}
            for d in data:
                val = _header_text(d).split("=")
                values[val[0]] = val[1].rstrip()
            header[title] = values

        param = {}

        def from_array(key, param):
            return float(header[key][param].split(",")[4])

        def get_values(key, param, df=False):
            string = header[key][param]

            ty, val, unit = re.search(
                r"(\S)\[(.*)\]\*\[(.*)\]", string
            ).groups()
            if ty == "D":
                val = float(val)
            if ty == "B":
                val = bool(val)
            if ty == "L":
                val = int(val)
            if ty == "V":
                val = [float(v) for v in val.split(",")]
                unit = [str(u) for u in unit.split(",")]
            out = {"Value": val, "Unit": unit}
            if df:
                if len(out) == 1:
                    return pd.DataFrame(out)
                else:
                    return pd.DataFrame(out)
            else:
                return out

        groupNames = []
        dataSet = []

        ds = "DataSet"
        grCnt = int(header[ds]["GroupCount"])
        for gr in range(grCnt):
            groupNames.append(header[ds]["Gr" + str(gr) + "-Name"])
            gc = int(header[ds]["Gr" + str(gr) + "-Count"])

            for g in range(gc):
                key = "Gr" + str(gr) + "-Ch" + str(g)
                if key in header[ds]:
                    dataSet.append(header[ds][key])

        terms = [
            "Frame",
            "Points",
            "Lines",
            "SaveBits",
            "Dim0Min",
            "Dim0Range",
            "Dim0Unit",
            "Dim1Min",
            "Dim1Range",
            "Dim1Unit",
            "Dim2Min",
            "Dim2Range",
            "Dim2Unit",
            "Dim2Name",
        ]

        types = [0, 1, 1, 1, 2, 2, 0, 2, 2, 0, 2, 2, 0, 0]

        # 0 = string, 1 = integer, 2 = float
        typ = [lambda x: str(x), lambda x: int(x), lambda x: float(x)]

        for term, ty in zip(terms, types):
            val = []
            for d in dataSet:
                h = header[d]
                if term in h:
                    val.append(typ[ty](h[term]))

            param[term] = val

        # for spectroscopy with maps
        dataPoints = []
        for d in dataSet:
            h = header[d]
            linePoints = []
            if "LineDim0Min" in h:
                for key, value in h.items():
                    if re.match(r"LineDim\d*Points", key):
                        linePoints.append(int(value))
            dataPoints.append(np.array(linePoints))
        param["LinePoints"] = dataPoints

        cantilever = {}
        key = "DataSet\\Calibration\\Cantilever"
        if key in header:
            cantilever = {
                "Manufacturer": header[key]["Manufacturer"],
                "Name": header[key]["Name"],
            }

            propcount = int(header[key]["PropCount"])
            for i in range(propcount):
                cantilever["Prop" + str(i)] = get_values(key, "Prop" + str(i))

        key = "DataSet\\Calibration\\Scanhead"
        if key in header:
            sens = from_array(key, "In5")
            max_defl = from_array(key, "In0")
            cantilever["Sensitivity"] = {"Value": sens / max_defl, "Unit": "m"}

        mapTable = {}
        key = "DataSet\\SpecInfos"
        if key in header:
            n = int(header[key]["SubSectionCount"])
            secNames = [header[key]["SubSection" + str(i)] for i in range(n)]

            specMode = header[key + "\\" + secNames[0]]["SpecMode"]

            count = int(header[key + "\\" + secNames[1]]["Count"])

            mapTable = [
                header[key + "\\" + secNames[1]][specMode[:3] + str(i)].split(
                    ";"
                )
                for i in range(count)
            ]
            mapTable = np.array(mapTable).astype(float)[0]

        specParam = {}
        modFw, modBw, pauseFw, pauseBw, rangeFw, rangeBw, stopFw = (
            [] for i in range(7)
        )
        key = "DataSet\\Parameters\\Spectroscopy"
        if key in header:
            modFw = get_values(key, "FwModTime")
            modBw = get_values(key, "BwModTime")
            pauseFw = get_values(key, "FwPauseTime")
            pauseBw = get_values(key, "BwPauseTime")
            rangeFw = get_values(key, "FwModRange")
            rangeBw = get_values(key, "BwModRange")
            stopFw = get_values(key, "FwModStopValue")

        scanParam = {}
        scanRange, scanTime, scanOffset, scanRotation = ([] for i in range(4))
        key = "DataSet\\Parameters\\Imaging"
        if key in header:
            scanRange = get_values(key, "ScanRange")
            scanTime = get_values(key, "ScanTime")
            scanOffset = get_values(key, "ScanOffset")
            scanRotation = get_values(key, "ScanRotation")

        thermal = {}
        key = "DataSet-Info"
        if "-- Thermal Tuning --" in header[key]:
            if "k" in header[key]["Q Factor:"]:
                Q = float(header[key]["Q Factor:"][:-1]) * 1e3
            else:
                Q = float(header[key]["Q Factor:"])
            Freq = header[key]["Frequency:"]
            SpringK = header[key]["Spring Constant:"]
            Peak = header[key]["Peak Value:"]

            gr = re.search(r"(\d+\.\d+)(\S*)", Freq).groups()
            Freq = {"Value": gr[0], "Unit": gr[1]}
            try:
                gr = re.search(r"(\d+\.\d+)\s(\S*)", SpringK).groups()
            except:
                gr = re.search(r"(\d)\s(\S*)", SpringK).groups()
            SpringK = {"Value": gr[0], "Unit": gr[1]}

            try:
                gr = re.search(r"(\d+\.\d+)(\S*)", Peak).groups()
            except:
                gr = re.search(r"(\d)(\S*)", Peak).groups()
            Peak = {"Value": gr[0], "Unit": gr[1]}

            thermal = {
                "Frequency": Freq,
                "Q Factor": Q,
                "Spring Constant": SpringK,
                "Peak": Peak,
            }

        required_size = int(
            sum(
                [
                    a * b * c / 8
                    for a, b, c in zip(
                        param["Points"], param["Lines"], param["SaveBits"]
                    )
                ]
            )
        )

        q = float(2 ** param["SaveBits"][0])
        z0 = float(q / 2)

        if param["SaveBits"][0] == 16:
            dt = np.int16
        elif param["SaveBits"][0] == 32:
            dt = np.int32

        if self.verbose:
            print("Reading Data")

        if required_size > len(raw):
            raise ValueError("The file is shorter than its header says (incomplete file?).")
        data_in = np.frombuffer(raw, dtype=dt, offset=len(raw) - required_size).astype(float)
        data_in = (data_in + z0) / q

        data_in = np.split(
            data_in,
            np.cumsum(
                [a * b for a, b in zip(param["Points"], param["Lines"])]
            ),
        )
        data_in.pop()

        data = []
        # reshape and rescale data
        for datain, pts, lns, zran, zmin in zip(
            data_in,
            param["Points"],
            param["Lines"],
            param["Dim2Range"],
            param["Dim2Min"],
        ):

            data.append(datain.reshape((lns, pts)).__mul__(zran).__add__(zmin))

        data_crop = []
        for numP, dSet in zip(param["LinePoints"], data):
            if numP.any():
                temp = []
                for n, d in zip(numP, dSet):
                    temp.append(d[:n])
                data_crop.append(temp)
            else:
                data_crop.append(dSet)
        data = np.empty(len(data_crop), dtype=object)    # one array per dataset, not one object per value
        for number, block in enumerate(data_crop):
            data[number] = block

        # this is the index of the channels that will be output.
        idx = [
            idx
            for idx, frame in enumerate(param["Frame"])
            if frame in self.data_names
        ]

        # only take data from approved list in name
        data = data[idx]
        frames = np.array(param["Frame"])[idx]
        channel = np.array(param["Dim2Name"])[idx]

        frame_chan = [i + ";" + j for i, j in zip(frames, channel)]

        seen = {}
        new_channel = []
        for x in frame_chan:                     # channels of the same name: Amplitude, Amplitude2, Amplitude3
            seen[x] = seen.get(x, 0) + 1
            new_channel.append(x if seen[x] == 1 else f"{x}{seen[x]}")
        channel = [x.split(";")[1] for x in new_channel]

        out = {}
        for frame, chan, dat in zip(frames, channel, data):
            if frame not in out:
                out[frame] = {}
            if chan not in out[frame]:
                out[frame][chan] = {}
            if isinstance(dat, list):
                out[frame][chan] = dat
            else:
                out[frame][chan] = dat.astype(float, copy=False)

        image = {}
        spec = {}
        spectrum = {}
        targets = {"Image": image, "Spec": spec, "Sweep": spectrum}

        for name, value in out.items():
            group, label = _NID_FRAMES[name]
            targets[group][label] = value

        # the frames of a file with spectroscopy start with the curves: the size of the image is that of an image frame
        first_image = next((i for i, frame in enumerate(param["Frame"])
                            if _NID_FRAMES.get(frame, ("",))[0] == "Image"), None)
        self.image_size = None
        if first_image is not None:
            self.image_size = (float(param["Dim0Range"][first_image]), float(param["Dim1Range"][first_image]))

        self.units = {}
        for frame, chan, unit in zip(frames, channel, np.array(param["Dim2Unit"])[idx]):
            group, label = _NID_FRAMES[frame]
            self.units[(group, label, chan)] = str(unit)

        x = dict(
            zip(
                ["min", "range", "units"],
                [param["Dim0Min"], param["Dim0Range"], param["Dim0Unit"]],
            )
        )
        y = dict(
            zip(
                ["min", "range", "units"],
                [param["Dim1Min"], param["Dim1Range"], param["Dim1Unit"]],
            )
        )
        z = dict(
            zip(
                ["min", "range", "units"],
                [param["Dim2Min"], param["Dim2Range"], param["Dim2Unit"]],
            )
        )
        scanParam = dict(
            zip(
                ["range", "time/line", "offset", "rotation"],
                [scanRange, scanTime, scanOffset, scanRotation],
            )
        )
        specParam = dict(
            zip(
                [
                    "modFw",
                    "modBw",
                    "pauseFw",
                    "pauseBw",
                    "rangeFw",
                    "rangeBw",
                    "stop-by-value",
                    "maps",
                ],
                [
                    modFw,
                    modBw,
                    pauseFw,
                    pauseBw,
                    rangeFw,
                    rangeBw,
                    stopFw,
                    mapTable,
                ],
            )
        )

        parameters = dict(
            zip(
                ["Tip", "X", "Y", "Z", "Scan", "Spec", "Tune", "HeaderDump"],
                [cantilever, x, y, z, scanParam, specParam, thermal, header],
            )
        )

        dataout = [image, spec, spectrum]
        datanames = ["Image", "Spec", "Sweep"]

        valid_data = [val for num, val in enumerate(dataout) if val]
        valid_names = [
            datanames[num] for num, val in enumerate(dataout) if val
        ]

        dataout = dict(zip(valid_names, valid_data))
        t_end = time.time() - t

        if self.verbose:
            print("Elapsed Time: %3.2f sec\n" % t_end)

        if self.__dataframe:
            # import pandas as pd
            self.data = toPandaDF(dataout).unstack(level=0)
            self.data.dropna(inplace=True)
            # self.specunits = pd.DataFrame(specUnits)
            self.param = toPandaSeries(parameters)
        else:
            self.data = dataout
            self.param = parameters


class nhf_read:
    """Opens a Nanosurf Studio .nhf file with h5py.

    Parameters
    ----------
        filename : string
        measurement : string or int, default the first one
            Which measurement of the file to open, see .measurements
        dataframe : boolean, default True
            data and param as pandas Series, else as plain dicts
    """

    def __init__(self, filename=None, measurement=None, dataframe=True, **_):
        self.filename = filename
        self.data = None
        self.param = None
        self.units = {}
        self.spec = None
        self.nma = None
        self.measurements = []
        self.__dataframe = dataframe
        self.file = NHFFile(filename)
        try:
            self.read(measurement)
        except Exception:
            self.file.close()
            raise

    def read(self, which=None):
        self.measurements = list(self.file.measurements)
        measurement = self.file.select(which)
        self.measurement = measurement
        ny, nx = measurement.shape
        size_x, size_y = measurement.size

        param = dict(measurement.attrs)
        param.update(
            {
                "file_version": self.file.version,
                "probe_k": measurement.spring_constant,
                "probe_sens": measurement.deflection_sensitivity,
                "size_x": size_x,
                "size_y": size_y,
                "points": nx,
                "lines": ny,
                "scan_up": measurement.scan_up,
            }
        )

        data = {}
        if measurement.is_spectroscopy:
            self.spec = Spectroscopy(measurement)
            param["number"] = self.spec.n_points
            segments = {name: self.spec[name] for name in self.spec.segments}
            data["Spec"] = segments
            for segment, channels in segments.items():
                for name, channel in channels.items():
                    self.units[("Spec", segment, name)] = channel.unit
        else:
            if measurement.is_wavemode_nma:
                self.nma = WaveModeNMA(measurement)
            data["Image"] = {}
            for segment in measurement.segments.values():
                channels = {}
                for name, channel in segment.channels.items():
                    if len(channel) > nx * ny or len(channel) == 0:  # the curves of NMA: use .nma
                        continue
                    values = np.full(nx * ny, np.nan)
                    values[: len(channel)] = channel.read()
                    channels[name] = values.reshape(ny, nx)
                    self.units[("Image", segment.name, name)] = channel.unit
                if channels:
                    data["Image"][segment.name] = channels

        if self.__dataframe:
            self.data = _nested_series(data)
            self.param = pd.Series(param)
        else:
            self.data = data
            self.param = param

    def close(self):
        self.file.close()


def _nested_series(data):
    """A Series with index (kind, segment, channel), holding arrays or CurveChannels as they are."""
    keys, values = [], []
    for kind, segments in data.items():
        for segment, channels in segments.items():
            for channel, value in channels.items():
                keys.append((kind, segment, channel))
                values.append(value)
    holder = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        holder[i] = value
    return pd.Series(holder, index=pd.MultiIndex.from_tuples(keys))


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


def toPandaSeries(user_dict):
    df = pd.Series(
        {
            (i, j): user_dict[i][j]
            for i in user_dict.keys()
            for j in user_dict[i].keys()
        }
    )
    return df
