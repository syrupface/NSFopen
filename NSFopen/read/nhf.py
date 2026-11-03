# -*- coding: utf-8 -*-
"""
Reader for Nanosurf Studio .nhf files (HDF5), using only h5py and numpy.

Handles file versions 1.1 and 2.x: images, spectroscopy points and grids and
WaveMode NMA. Scalar image channels are read when asked for; the force curves
of spectroscopy and NMA files stay on disk and are read curve by curve, so a
file of several GB opens immediately.

Conventions
-----------
    * All values are floats in the unit of the channel's selected signal (or the
      unit asked for). Samples that were not acquired are NaN.
    * Grid position (row, column): row 0 is the bottom line, column 0 the left.
      An image array has the same layout for an upward scan.
    * rect_axis_size and rect_axis_range are stored as [y, x] in the file.
"""

import pathlib
import re
import warnings
from collections import deque

import h5py
import numpy as np

_META_NAME = re.compile(r"^[a-z0-9_]+$")  # offsets, per-curve min and max, ...
_SKIPPED_GROUPS = ("calibration", "analysis")
_FAST_DEFLECTION = ("Fast In Deflection (16bit)", "NMA Deflection")
_FREE_WAVE = ("WaveMode Free Wave", "NMA Free Wave")
_TIME_SHIFT = ("wavemode_time_shift", "wavemode_free_wave_phase_delay")
_AMPLITUDE = ("wavemode_amplitude_correction", "wavemode_free_wave_amplitude_correction")
_FREQUENCY = ("wavemode_frequency", "wave_frequency")
_FREE_WAVE_BASELINE = "WaveMode Free Wave Baseline"
_DEFLECTION_BASELINE = "NMA Deflection Baseline"


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _plain(value):
    """An attribute as a plain Python value."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def extract_unit(text):
    """'meter (m)' -> 'm'"""
    found = re.findall(r"\(([^)]+)\)", _text(text))
    return found[0] if found else _text(text)


def grid_index(row, col, nx, snake=True):
    """Acquisition index of a grid position; works on arrays too."""
    return row * nx + np.where((row % 2 == 0) | (not snake), col, nx - 1 - col)


class _CalibrationNode:
    """One conversion of the calibration graph of a 2.x file."""

    def __init__(self, source, sink, unit, kind, par, forward):
        self.source, self.sink, self.unit = source, sink, extract_unit(unit)
        self.kind, self.par, self.forward = kind, par, forward

    def apply(self, x):
        a, b = self.par[0], self.par[1]
        if self.kind in ("linear", "linear_non_invertible"):
            return x * a + b if self.forward else (x - b) / a
        c = self.par[2]
        exponential = (self.kind == "exponential") == self.forward
        if self.kind in ("exponential", "logarithmic"):
            return b * np.exp((x - c) / a) if exponential else a * np.log(x / b) + c
        raise ValueError(f"unsupported calibration type '{self.kind}'")


def _read_calibration_nodes(h5file):
    """{source signal: [nodes]} from the calibration group, empty if there is none."""
    graph = {}
    for group in h5file.values():
        if not isinstance(group, h5py.Group) or _text(group.attrs.get("group_type", "")) != "calibration":
            continue
        for dataset in group.values():
            a = dataset.attrs
            try:
                source, sink = _text(a["signal_source_id"]), _text(a["signal_sink_id"])
                kind, way = _text(a["signal_mapping_type"]), _text(a["signal_invertibility"])
                par = [float(v) for v in a["signal_mapping_parameters"]]
                if way in ("right", "two_way"):
                    node = _CalibrationNode(source, sink, _text(a["signal_sink_unit"]), kind, par, way == "right")
                    graph.setdefault(source, []).append(node)
                if way in ("left", "two_way"):
                    node = _CalibrationNode(sink, source, _text(a["signal_source_unit"]), kind, par, True)
                    graph.setdefault(sink, []).append(node)
            except (KeyError, TypeError, ValueError):
                continue
    return graph


def _paths_from(graph, start):
    """{signal: [nodes from start to it]}, breadth first: the shortest way to every signal."""
    paths, queue = {start: []}, deque([start])
    while queue:
        signal = queue.popleft()
        for node in graph.get(signal, []):
            if node.sink not in paths:
                paths[node.sink] = paths[signal] + [node]
                queue.append(node.sink)
    return paths


class Channel:
    """One dataset of a segment. Nothing is read until read() is called."""

    def __init__(self, dataset, nhf_file):
        self.dataset = dataset
        self.file = nhf_file
        self.attrs = {k: _plain(v) for k, v in dataset.attrs.items()}
        self.name = _text(self.attrs.get("signal_name", self.attrs.get("name", "")))
        self._paths = None

    def __len__(self):
        return self.dataset.shape[0]

    @property
    def signal_id(self):
        return _text(self.attrs.get("signal_id", ""))

    @property
    def selected_signal(self):
        return _text(self.attrs.get("signal_selected", self.signal_id))

    def _graph_paths(self):
        if self._paths is None:
            self._paths = _paths_from(self.file.calibration, self.signal_id) if self.signal_id else {}
        return self._paths

    def _calibrated_unit(self):
        name = "signal_calibration_unit" if "signal_calibration_unit" in self.attrs else "base_calibration_unit"
        return extract_unit(self.attrs.get(name, ""))

    @property
    def unit(self):
        """Unit of read() without arguments."""
        path = self._graph_paths().get(self.selected_signal)
        return path[-1].unit if path else self._calibrated_unit()

    def units(self):
        """{unit: signal} this channel can be read in."""
        out = {self.unit: self.selected_signal}
        for signal, path in self._graph_paths().items():
            if path:
                out.setdefault(path[-1].unit, signal)
        out.setdefault(self._calibrated_unit(), self.signal_id)
        return out

    def _type_range(self, dtype):
        """Raw values of the calibration range. In 2.x the extreme value is reserved for 'no data'."""
        if not np.issubdtype(dtype, np.integer):
            return 0.0, 1.0
        info = np.iinfo(dtype)
        if self.file.version >= (2, 0):
            return (0.0, float(info.max) - 1.0) if info.min == 0 else (float(info.min) + 1.0, float(info.max))
        return float(info.min), float(info.max)

    def _scale_linear(self, values, dtype):
        low = self.attrs.get("signal_calibration_min", self.attrs.get("base_calibration_min"))
        high = self.attrs.get("signal_calibration_max", self.attrs.get("base_calibration_max"))
        if low is None or high is None:
            return values
        raw_low, raw_high = self._type_range(dtype)
        return (values - raw_low) * ((float(high) - float(low)) / (raw_high - raw_low)) + float(low)

    def read(self, selection=slice(None), unit=None):
        """Values as a float array.

        Parameters
        ----------
            selection : slice or array-like, default all
                Part of the dataset, as raw sample numbers
            unit : string, default None
                Read in this unit (see units()) instead of the selected signal's
        """
        raw = np.asarray(self.dataset[selection])
        dtype = raw.dtype
        invalid = None
        if np.issubdtype(dtype, np.integer):
            if self.file.version >= (2, 0):
                invalid = raw == self.dataset.fillvalue
            elif np.iinfo(dtype).min < 0:  # version 1 has no fill value: lines not acquired hold the minimum
                invalid = raw == np.iinfo(dtype).min
        values = raw.astype(np.float64)
        if unit is None:
            target = self.selected_signal
        else:
            known = self.units()
            if unit not in known:
                raise ValueError(f"Channel '{self.name}' cannot be read in '{unit}', only in {sorted(known)}.")
            target = known[unit]
        path = self._graph_paths().get(target)
        if path:
            for node in path:
                values = node.apply(values)
        elif self._calibrated_unit() != "int":
            values = self._scale_linear(values, dtype)
        if invalid is not None and invalid.any():
            values[invalid] = np.nan
        return values


class Segment:
    """A segment of a measurement ('Forward', 'Retract 1', ...)."""

    def __init__(self, name, group, nhf_file):
        self.name = name
        self.group = group
        self.file = nhf_file
        self.attrs = {k: _plain(v) for k, v in group.attrs.items()}
        self.channels = {}
        self._metadata = {}
        for dataset in group.values():
            if not isinstance(dataset, h5py.Dataset):
                continue
            channel = Channel(dataset, nhf_file)
            if not channel.name:
                continue
            if _META_NAME.match(channel.name):
                self._metadata[channel.name] = channel
            else:
                self.channels[channel.name] = channel
        self._blocks = None

    def blocks(self):
        """(start, count) arrays of every curve in the datasets of this segment."""
        if self._blocks is None:
            self._blocks = self._read_blocks()
        return self._blocks

    def _read_blocks(self):
        first = next(iter(self.channels.values()), None)
        if first is None:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        if "channel_data_offsets" in self._metadata:  # version 1
            starts = np.asarray(self._metadata["channel_data_offsets"].dataset[()], np.int64)
            counts = np.asarray(self._metadata["number_of_datapoints_acquired"].dataset[()], np.int64)
            return starts, counts
        source = first.attrs.get("dataset_block_size_source")
        if source is not None:
            for dataset in self.group.values():
                if _plain(dataset.attrs.get("dataset_block_size_id", "")) == source:
                    if dataset.shape[0] > 0:
                        counts = np.asarray(dataset[()], np.int64)
                        return np.concatenate([[0], np.cumsum(counts)[:-1]]), counts
                    size = int(dataset.attrs["dataset_block_size_max"])  # fixed size blocks: not written out
                    number = len(first) // max(size, 1)
                    return np.arange(number, dtype=np.int64) * size, np.full(number, size, np.int64)
        count = int(self.attrs.get("number_of_datapoints_acquired", len(first)))  # version 1, single spec point
        return np.zeros(1, np.int64), np.array([min(count, len(first))], np.int64)


class Measurement:
    """A measurement group: its attributes, segments and root channels."""

    def __init__(self, group, nhf_file):
        self.group = group
        self.file = nhf_file
        self.attrs = {k: _plain(v) for k, v in group.attrs.items()}
        self.name = _text(self.attrs.get("measurement_name", self.attrs.get("name", group.name)))
        self.type = _text(self.attrs.get("group_type", self.attrs.get("measurement_type", "")))
        self.channels = {}
        self.segments = {}
        for item in group.values():
            if isinstance(item, h5py.Group):
                base = _text(item.attrs.get("segment_name", item.attrs.get("name", item.name)))
                name, number = base, 2  # version 1 grids have two segments called 'Wait'
                while name in self.segments:
                    name, number = f"{base} {number}", number + 1
                self.segments[name] = Segment(name, item, nhf_file)
            else:
                channel = Channel(item, nhf_file)
                if channel.name:
                    self.channels[channel.name] = channel

    @property
    def is_image(self):
        return self.type == "image_line_based"

    @property
    def is_spectroscopy(self):
        return self.type.startswith("spectroscopy")

    @property
    def is_wavemode_nma(self):
        return self.type == "wavemode_nma"

    @property
    def shape(self):
        """(lines, points per line) of the image or grid."""
        a = self.attrs
        size = a.get("rect_axis_size")
        if size is None:
            for segment in self.segments.values():
                size = segment.attrs.get("rect_axis_size")
                if size is not None:
                    break
        if size is not None:
            return int(size[0]), int(size[1])
        if "image_points_per_line" in a:
            return int(a["image_number_of_lines"]), int(a["image_points_per_line"])
        if "grid_number_of_columns" in a:
            return int(a["grid_number_of_rows"]), int(a["grid_number_of_columns"])
        return 1, 1

    @property
    def size(self):
        """(x, y) range of the image or grid in m."""
        a = self.attrs
        if "rect_axis_range" in a:
            return float(a["rect_axis_range"][1]), float(a["rect_axis_range"][0])
        if "image_size_x" in a:
            return float(a["image_size_x"]), float(a["image_size_y"])
        if "grid_width" in a:
            return float(a["grid_width"]), float(a["grid_height"])
        return 0.0, 0.0

    @property
    def scan_up(self):
        direction = self.attrs.get("scan_direction", self.attrs.get("image_scan_direction", "up"))
        return _text(direction).lower() == "up"

    @property
    def deflection_sensitivity(self):
        return _plain(self.attrs.get("spm_probe_calibration_deflection_sensitivity", float("nan")))

    @property
    def spring_constant(self):
        return _plain(self.attrs.get("spm_probe_calibration_spring_constant", float("nan")))


class NHFFile:
    """An open .nhf file. Use as a context manager, or call close() when done."""

    def __init__(self, filename):
        self.path = pathlib.Path(filename)
        self.h5 = h5py.File(self.path, "r")
        try:
            major = int(self.h5.attrs["nsf_file_version_major"])
            minor = int(self.h5.attrs["nsf_file_version_minor"])
        except KeyError:
            self.h5.close()
            raise ValueError(f"{self.path.name} is not a Nanosurf nhf file (it has no file version).")
        self.version = (major, minor)
        self.calibration = _read_calibration_nodes(self.h5)
        self.measurements = {}
        for group in self.h5.values():
            if isinstance(group, h5py.Group) and _text(group.attrs.get("group_type", "")) not in _SKIPPED_GROUPS:
                measurement = Measurement(group, self)
                self.measurements[measurement.name] = measurement

    def select(self, measurement=None):
        """A measurement by name or number; the first one by default."""
        names = list(self.measurements)
        if not names:
            raise ValueError(f"{self.path.name} contains no measurement.")
        if measurement is None:
            return self.measurements[names[0]]
        return self.measurements[names[measurement] if isinstance(measurement, int) else measurement]

    def close(self):
        if self.h5:
            self.h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class CurveChannel:
    """A channel of a spectroscopy segment. Curves are read from the file on access.

    channel[n]        n-th curve in acquisition order (1D array)
    channel[row, col] curve at a grid position, row 0 at the bottom (1D array)
    channel.curves()  all curves in acquisition order (list of 1D arrays)
    channel.array()   all curves as (rows, columns, samples), NaN padded
    np.asarray(channel) is the same as array()
    """

    def __init__(self, channel, segment, measurement, unit=None):
        self.channel = channel
        self.segment = segment
        self.measurement = measurement
        self.n_points = segment.blocks()[0].size
        self.read_unit = unit

    @property
    def unit(self):
        return self.read_unit or self.channel.unit

    @property
    def shape(self):
        return self.measurement.shape

    def __len__(self):
        return self.n_points

    def __repr__(self):
        return f"<CurveChannel '{self.channel.name}' of '{self.segment.name}': {self.n_points} curves [{self.unit}]>"

    def to(self, unit):
        """The same channel, read in another unit."""
        return CurveChannel(self.channel, self.segment, self.measurement, unit)

    def _snake(self):
        pattern = _text(self.channel.attrs.get("signal_data_pattern", "snake"))
        return "snake" in pattern

    def _point(self, key):
        ny, nx = self.shape
        if isinstance(key, tuple):
            row, col = key
            row, col = row + ny if row < 0 else row, col + nx if col < 0 else col
            if not (0 <= row < ny and 0 <= col < nx):
                raise IndexError(f"Grid position {key} is outside the {ny} x {nx} grid.")
            key = int(grid_index(row, col, nx, self._snake()))
        key = key + self.n_points if key < 0 else key
        if not 0 <= key < self.n_points:
            raise IndexError(f"Curve {key} was not measured ({self.n_points} curves in the file).")
        return int(key)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)) or (
                isinstance(key, tuple) and len(key) == 2 and all(isinstance(k, (int, np.integer)) for k in key)):
            start, count = self.segment.blocks()
            point = self._point(int(key) if not isinstance(key, tuple) else (int(key[0]), int(key[1])))
            return self.channel.read(slice(int(start[point]), int(start[point] + count[point])), self.read_unit)
        return self.array()[key]

    def curves(self):
        start, count = self.segment.blocks()
        start, count = start[:self.n_points], count[:self.n_points]
        if not start.size:
            return []
        low, high = int(start.min()), int((start + count).max())
        values = self.channel.read(slice(low, high), self.read_unit)
        return [values[s - low:s - low + c] for s, c in zip(start, count)]

    def array(self):
        ny, nx = self.shape
        curves = self.curves()
        longest = max((c.size for c in curves), default=0)
        out = np.full((ny, nx, longest), np.nan)
        lines, columns = np.divmod(np.arange(len(curves)), nx)
        snake = self._snake()
        for line, column, curve in zip(lines, columns, curves):
            if line < ny:
                col = nx - 1 - column if snake and line % 2 else column
                out[line, col, :curve.size] = curve
        return out

    def __array__(self, dtype=None, copy=None):
        out = self.array()
        return out if dtype is None else out.astype(dtype)

    def last(self):
        """Last sample of every curve as a (rows, columns) map, NaN where nothing was measured."""
        ny, nx = self.shape
        start, count = self.segment.blocks()
        values = np.full(ny * nx, np.nan)
        for i, (s, c) in enumerate(zip(start[:self.n_points], count[:self.n_points])):
            if c > 0 and i < values.size:
                values[i] = self.channel.read(int(s + c - 1), self.read_unit)
        out = values.reshape(ny, nx)
        if self._snake():
            out[1::2] = out[1::2, ::-1]
        return out


class Spectroscopy:
    """Lazy access to the force curves of a spectroscopy point or grid.

    spec.segments, spec.channels(segment), spec.n_points, spec.shape
    spec["Retract 1"]["Deflection"][row, col]   a curve at a grid position
    spec.force_curve("Retract 1", (row, col))   (z in m, force in N)
    """

    def __init__(self, measurement):
        self.measurement = measurement
        self.shape = measurement.shape
        self.size = measurement.size
        self.sensitivity = measurement.deflection_sensitivity
        self.spring_constant = measurement.spring_constant
        self.segments = [name for name, seg in measurement.segments.items() if seg.channels]
        # version 1 grids keep their Wait segments as a single block, so the largest count is the number of curves
        self.n_points = max((measurement.segments[name].blocks()[0].size for name in self.segments), default=0)
        if self.n_points < self.shape[0] * self.shape[1]:
            warnings.warn(f"Only {self.n_points} of {self.shape[0] * self.shape[1]} grid points were measured.")

    def __getitem__(self, segment):
        seg = self.measurement.segments[segment]
        return {name: CurveChannel(channel, seg, self.measurement) for name, channel in seg.channels.items()}

    def channels(self, segment=None):
        segment = segment or self.segments[0]
        return list(self.measurement.segments[segment].channels)

    def coordinates(self):
        """(x, y) of every curve in acquisition order, in m, if the file has them."""
        for names in (("coordinates_x", "coordinates_y"), ("x_coordinates", "y_coordinates")):
            if names[0] in self.measurement.channels:
                return tuple(np.asarray(self.measurement.channels[n].dataset[()], float) for n in names)
        return None

    def deflection(self, segment, point, unit="V"):
        """Deflection of one curve in V, m or N, from whichever signals the file holds."""
        channel = self[segment]["Deflection"]
        known = channel.channel.units()
        if unit in known:
            return channel.to(unit)[point]
        in_volt = channel.to("V")[point] if "V" in known else None
        if in_volt is None:
            raise ValueError(f"Deflection is stored in {sorted(known)}, which cannot be converted to {unit}.")
        return in_volt * {"m": self.sensitivity, "N": self.sensitivity * self.spring_constant}[unit]

    def force_curve(self, segment, point):
        """(Position Z in m, force in N) of one curve, point = n or (row, col)."""
        return self[segment]["Position Z"].to("m")[point], self.deflection(segment, point, "N")


class NMACurve:
    """One advance-retract cycle of WaveMode NMA, as NumPy arrays over one wave period."""

    def __init__(self, time, z, deflection, interaction, turn, sensitivity, spring_constant):
        self.time = time  # s
        self.z = z  # m
        self.deflection = deflection  # V, raw NMA deflection
        self.interaction = interaction  # V, deflection minus free wave, without their baselines
        self.turn = turn  # sample of the closest approach
        self.sensitivity, self.spring_constant = sensitivity, spring_constant

    @property
    def force(self):
        return self.interaction * self.sensitivity * self.spring_constant

    @property
    def advance(self):
        """(z, force) up to the closest approach."""
        return self.z[:self.turn], self.force[:self.turn]

    @property
    def retract(self):
        """(z, force) from the closest approach on."""
        return self.z[self.turn:], self.force[self.turn:]


class WaveModeNMA:
    """Lazy access to the per-pixel curves of a WaveMode NMA image.

    The image holds one wave period per pixel in 'Fast In Deflection (16bit)', or
    'NMA Deflection' in older files. The interaction d - w is the deflection minus
    the free wave, each without its baseline, times the amplitude correction; older
    files store the same in 'NMA Interaction'. Z is the free wave, shifted by the
    time shift, times the deflection sensitivity. Each period splits into advance
    and retract at the minimum of Z.

    nma.curve((row, col), scan="Forward")  an NMACurve
    nma.interaction(first, stop, scan)     block of pixels as (pixels, samples)
    nma.map(func, scan)                    func(block) per pixel, as an image
    """

    def __init__(self, measurement):
        self.measurement = measurement
        self.shape = measurement.shape
        self.size = measurement.size
        self.sensitivity = measurement.deflection_sensitivity
        self.spring_constant = measurement.spring_constant
        self.scans = [name for name in ("Forward", "Backward") if name in measurement.segments]
        first = measurement.segments[self.scans[0]]
        self.fast_channel = next((n for n in _FAST_DEFLECTION if n in first.channels), None)
        wave_name = next((n for n in _FREE_WAVE if n in measurement.channels), None)
        if self.fast_channel is None or wave_name is None:
            raise ValueError("This WaveMode NMA file has no stored curves (NMA deflection and free wave).")
        wave = measurement.channels[wave_name]
        self.free_wave = wave.read(unit="V" if "V" in wave.units() else None)
        self.samples = self.free_wave.size
        a = measurement.attrs
        self.frequency = next((float(a[k]) for k in _FREQUENCY if k in a), float("nan"))
        self.sampling_rate = self.frequency * self.samples
        self.time_shift = next((float(a[k]) for k in _TIME_SHIFT if k in a), 0.0)
        self.amplitude_correction = next((float(a[k]) for k in _AMPLITUDE if k in a), 1.0)
        self.free_wave_baseline = self._read_free_wave_baseline()
        ny, nx = self.shape
        self.n_points = min([ny * nx] + [len(measurement.segments[s].channels[self.fast_channel]) // self.samples
                                         for s in self.scans])
        if self.n_points < ny * nx:
            warnings.warn(f"Only {self.n_points} of {ny * nx} pixels have NMA curves.")

    def _read_free_wave_baseline(self):
        """Level of the free wave: stored in newer files, else its mean over the baseline window of the period."""
        channels = self.measurement.channels
        if _FREE_WAVE_BASELINE in channels:
            channel = channels[_FREE_WAVE_BASELINE]
            return float(channel.read(unit="V" if "V" in channel.units() else None)[0])
        a = self.measurement.attrs
        begin, end = float(a.get("wavemode_baseline_window_begin", 0.95)), float(a.get("wavemode_baseline_window_end", 0.2))
        phase = np.arange(self.samples) / self.samples
        window = (phase >= begin) | (phase < end) if begin > end else (phase >= begin) & (phase < end)
        return float(self.free_wave[window].mean())

    def _baseline(self, first, stop, scan):
        """Deflection baseline in V of pixels first..stop-1, as (pixels, 1); zero if the file has none."""
        channels = self.measurement.segments[scan].channels
        if _DEFLECTION_BASELINE not in channels:
            return np.zeros((stop - first, 1))
        channel = channels[_DEFLECTION_BASELINE]
        return channel.read(slice(first, stop), "V" if "V" in channel.units() else None).reshape(-1, 1)

    def _point(self, point):
        if isinstance(point, tuple):
            row, col = point
            if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
                raise IndexError(f"Pixel {point} is outside the {self.shape[0]} x {self.shape[1]} image.")
            point = row * self.shape[1] + col
        if not 0 <= point < self.n_points:
            raise IndexError(f"Pixel {point} has no curve ({self.n_points} pixels in the file).")
        return int(point)

    def _fast(self, scan):
        return self.measurement.segments[scan].channels[self.fast_channel]

    def deflection(self, first, stop, scan="Forward"):
        """NMA deflection in V of pixels first..stop-1, as (pixels, samples)."""
        channel = self._fast(scan)
        selection = slice(first * self.samples, stop * self.samples)
        values = channel.read(selection, "V" if "V" in channel.units() else None)
        return values.reshape(-1, self.samples)

    def interaction(self, first, stop, scan="Forward", amplitude_correction=None, baseline=True):
        """Interaction in V of pixels first..stop-1, as (pixels, samples).

        baseline=False leaves out the baselines of deflection and free wave, so
        the result has the offset of the deflection in it.
        """
        correction = self.amplitude_correction if amplitude_correction is None else amplitude_correction
        channels = self.measurement.segments[scan].channels
        if baseline and "NMA Interaction" in channels:
            selection = slice(first * self.samples, stop * self.samples)
            return channels["NMA Interaction"].read(selection).reshape(-1, self.samples) * correction
        out = self.deflection(first, stop, scan) - self.free_wave
        if baseline:
            out = out - self._baseline(first, stop, scan) + self.free_wave_baseline
        return out * correction

    def z_wave(self, time_shift=None, sensitivity=None):
        """(Z over one period in m, sample of the closest approach)."""
        shift = self.time_shift if time_shift is None else time_shift
        sens = self.sensitivity if sensitivity is None else sensitivity
        index = np.arange(self.samples)
        wave = np.interp(index + shift * self.sampling_rate, index, self.free_wave, period=self.samples)
        return wave * sens, min(max(int(np.argmin(wave)), 1), self.samples - 1)

    def curve(self, point, scan="Forward", time_shift=None, amplitude_correction=None, sensitivity=None, baseline=True):
        """NMACurve of one pixel, point = n or (row, col); the calibration of the file unless given."""
        n = self._point(point)
        z, turn = self.z_wave(time_shift, sensitivity)
        return NMACurve(np.arange(self.samples) / self.sampling_rate, z, self.deflection(n, n + 1, scan)[0],
                        self.interaction(n, n + 1, scan, amplitude_correction, baseline)[0], turn,
                        self.sensitivity if sensitivity is None else sensitivity, self.spring_constant)

    def map(self, func=None, scan="Forward", block=4096):
        """An image with one value per pixel; block of pixels at a time, so it fits in memory.

        func takes an array (pixels, samples) of the interaction in V and returns
        one value per pixel. Default: the most negative interaction, i.e. the
        adhesion dip.
        """
        func = func or (lambda a: a.min(axis=1))
        ny, nx = self.shape
        values = np.full(ny * nx, np.nan)
        for first in range(0, self.n_points, block):
            stop = min(first + block, self.n_points)
            values[first:stop] = func(self.interaction(first, stop, scan))
        return values.reshape(ny, nx)
