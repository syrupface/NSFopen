# -*- coding: utf-8 -*-
# Part of NSFopen.process; the Nanosurf Surface Analyzer (Spectroscopy page) uses this module.
""" Spectroscopy files: force curves (force-distance) and I-V curves of Nanosurf .nid and .nhf files, read with NSFopen.
Processing of force curves (filter, baseline with optional interference sine, contact point, force and separation),
the measurements of a curve (adhesion, energies, stiffness) and the contact model fit (contact.py), and the same for
every curve of a file: a results table, its statistics and, for a grid (force map), maps of every result. No GUI code.

Conventions. x is the piezo position, oriented so that it increases towards the sample (files that store it the other
way are flipped when read); the force is repulsive > 0. The sample position delta = x - F/k is where the tip apex is
relative to the undeformed sample (the separation is -delta), the indentation is delta - delta0 with the contact point
delta0. Every segment is put in time order: the approach from far to the contact, the retract from the contact to far.
"""

import contextlib
import dataclasses
import io
import math
import pathlib
import re
import warnings
from typing import Callable, Optional

import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter

from . import contact

APPROACH, RETRACT, PAUSE, OTHER = "approach", "retract", "pause", "other"
FILTERS = [("none", "None"), ("mean", "Moving average"), ("median", "Median"), ("savgol", "Savitzky-Golay")]
BASELINES = [("none", "None"), ("offset", "Offset"), ("linear", "Offset and tilt"),
             ("sine", "Offset, tilt and interference (sine)")]
X_AXES = [("z", "Piezo Z"), ("separation", "Tip-sample separation"), ("indentation", "Indentation")]
SUFFIXES = (".nid", ".nhf")


class SpecError(Exception):
    pass


# ---- the file ----

@dataclasses.dataclass
class Segment:
    name: str
    kind: str                       # approach, retract, pause, other
    channels: list
    units: dict                     # channel -> unit


def segment_kind(name: str) -> str:
    low = name.lower()
    if "pause" in low or "wait" in low:
        return PAUSE
    if any(word in low for word in ("forward", "advance", "approach", "extend")):
        return APPROACH
    if any(word in low for word in ("backward", "retract", "withdraw")):
        return RETRACT
    return OTHER


def _header_value(text) -> float:
    """ 'D[2.72436]*[N/m]' or '2.5e-06' -> float; NaN if none. """
    match = re.search(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", str(text))
    return float(match.group()) if match else math.nan


def _section(header, name: str) -> dict:
    try:
        value = header[name]
    except (KeyError, TypeError):
        return {}
    return dict(value) if hasattr(value, "items") else {}


def is_spectroscopy_file(path) -> bool:
    """ True for a .nid or .nhf file that holds spectroscopy curves (checked without reading the curves). """
    path = pathlib.Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".nid":
            with open(path, "rb") as f:
                head = f.read(400_000).split(b"#!", 1)[0].decode("latin-1", errors="replace")
            infos = re.search(r"\[DataSet\\DataSetInfos\]\s*(.*?)\n\[", head, re.S)
            return bool(infos and re.search(r"SubSection0\s*=\s*Spec", infos.group(1)))
        if suffix == ".nhf":
            return bool(nhf_spectroscopy_measurements(path))
    except (OSError, ValueError, KeyError):
        return False
    return False


def nhf_measurements(path) -> list:
    """ [(name, type)] of the measurements of a Studio .nhf file (read with NSFopen; only the structure, no data). """
    from NSFopen.read.nhf import NHFFile
    with NHFFile(str(path)) as nhf:
        return [(name, str(m.type)) for name, m in nhf.measurements.items()]


def nhf_spectroscopy_measurements(path) -> list:
    """ Names of the spectroscopy measurements (single curves or grids) of an .nhf file. """
    return [name for name, kind in nhf_measurements(path) if kind.startswith("spectroscopy")]


class SpecFile:
    """ The curves of a spectroscopy file. data(segment, channel, n) gives a curve in SI units (as stored, not yet
        oriented or reordered: see ForceCurve). """

    _arrays = None

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.segments: list = []
        self.n_curves = 0
        self.grid: Optional[tuple] = None          # (nx, ny)
        self.positions: Optional[np.ndarray] = None   # (n, 2) x, y in m, or None
        self.spring_constant = math.nan
        self.sensitivity = math.nan                 # m/V
        self.modulated = ""                          # what the ramp moved: 'Z-Axis', 'Tip voltage', ...
        self.info: dict = {}
        self.notes: list = []
        self._ramps: dict = {}                      # segment -> (start, range) of a ramp that is not stored
        self._cache: dict = {}
        self._reader = None
        suffix = self.path.suffix.lower()
        if suffix not in SUFFIXES:
            raise SpecError(f"{self.path.name}: not a .nid or .nhf file.")
        try:
            from NSFopen.read import read
        except ImportError:
            raise SpecError("NSFopen is needed to read spectroscopy files (pip install -U NSFopen).")
        options = {}
        if suffix == ".nhf":
            try:
                measurements = nhf_measurements(self.path)
            except Exception as e:
                raise SpecError(f"Could not read {self.path.name}: {e}")
            spectroscopy = [name for name, kind in measurements if kind.startswith("spectroscopy")]
            if not spectroscopy:
                kinds = ", ".join(f"{name} ({kind or 'no type'})" for name, kind in measurements) or "none"
                raise SpecError(f"{self.path.name} holds no spectroscopy measurement (measurements: {kinds}).")
            options["measurement"] = spectroscopy[0]
            if len(spectroscopy) > 1:
                self.notes.append(f"{len(spectroscopy)} spectroscopy measurements: '{spectroscopy[0]}' is shown.")
        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self._reader = read(str(self.path), **options)
            self.notes += [str(w.message) for w in caught]
        except Exception as e:
            raise SpecError(f"Could not read {self.path.name}: {e}")
        if self._reader.data is None:
            raise SpecError(f"Could not read {self.path.name}.")
        if suffix == ".nid":
            self._load_nid()
        else:
            self._load_nhf()
        if not self.segments or self.n_curves == 0:
            raise SpecError(f"{self.path.name} holds no spectroscopy curves.")

    @classmethod
    def from_curves(cls, name: str, segments: dict, units: dict, spring_constant: float = math.nan,
                    sensitivity: float = math.nan, grid: Optional[tuple] = None, extent=(0.0, 1e-6, 0.0, 1e-6),
                    modulated: str = "") -> "SpecFile":
        """ A file made from arrays (tests, simulations): segments = {segment: {channel: [curve arrays]}},
            units = {channel: unit}. """
        self = cls.__new__(cls)
        self.path = pathlib.Path(name)
        self.segments, self.info, self.notes, self._ramps, self._reader = [], {"File": {"Name": name}}, [], {}, None
        self._cache = {}
        self.spring_constant, self.sensitivity, self.modulated = spring_constant, sensitivity, modulated
        self._arrays = segments
        for seg, channels in segments.items():
            self.segments.append(Segment(seg, segment_kind(seg), list(channels), {c: units.get(c, "") for c in channels}))
        self.n_curves = max(len(curves) for channels in segments.values() for curves in channels.values())
        self.grid = grid
        self.positions = _snake_positions(self.n_curves, grid[0], grid[1], extent) if grid else None
        return self

    # -- .nid --
    def _load_nid(self):
        data, param = self._reader.data, self._reader.param
        try:
            spec = data["Spec"]
        except KeyError:
            raise SpecError(f"{self.path.name} holds no spectroscopy curves.")
        header = param["HeaderDump"]
        info = _section(header, "DataSet-Info")
        self.info = {"File": {"Name": self.path.name, "Read with": "NSFopen"},
                     "Spectroscopy": {k: v for k, v in info.items() if not k.startswith("--")}}
        self.modulated = str(info.get("Mod. output", ""))
        names = list(dict.fromkeys(spec.index.get_level_values(0))) if hasattr(spec.index, "levels") else list(spec.index)
        for name in names:
            channels = list(dict.fromkeys(spec[name].index))
            units = {ch: self._reader.units.get(("Spec", name, ch), "") for ch in channels}
            self.segments.append(Segment(name, segment_kind(name), channels, units))
            self.n_curves = max(self.n_curves, *(len(spec[name][ch]) for ch in channels))
        cantilever = _section(header, "DataSet\\Calibration\\Cantilever")
        self.spring_constant = _header_value(cantilever.get("Prop0", "nan"))
        scanhead = _section(header, "DataSet\\Calibration\\Scanhead")
        match = re.search(r",m,([-+0-9.eE]+),", str(scanhead.get("In5", "")))
        if match:
            divider = 5.0 if scanhead.get("HeadTyp") == "DriveAFM" else 10.0
            self.sensitivity = float(match.group(1)) / divider
        self.info["Probe"] = {"Cantilever": cantilever.get("Name", ""), "Spring constant": f"{self.spring_constant:g} N/m",
                              "Deflection sensitivity": f"{self.sensitivity:g} m/V"}
        table = _section(header, "DataSet\\SpecInfos\\SpecMapTable")
        try:
            x0, x1, y0, y1, nx, ny = (float(v) for v in table["Map0"].split(";")[:6])
            nx, ny = int(nx), int(ny)
            if nx * ny > 1:
                self.grid = (nx, ny)
                self.positions = _snake_positions(self.n_curves, nx, ny, (x0, x1, y0, y1))
        except (KeyError, ValueError):
            pass
        spec_param = _section(header, "DataSet\\Parameters\\Spectroscopy")
        start = _header_value(spec_param.get("StartOffset", "0"))
        fw, bw = _header_value(spec_param.get("FwModRange", "nan")), _header_value(spec_param.get("BwModRange", "nan"))
        base = _header_value(info.get(self.modulated, "0")) if str(info.get("Relative", "")).lower() == "true" else 0.0
        if math.isfinite(base) and math.isfinite(fw):
            for segment in self.segments:
                if segment.kind == APPROACH:
                    self._ramps[segment.name] = (base + start, fw)
                elif segment.kind == RETRACT and math.isfinite(bw):
                    self._ramps[segment.name] = (base + start + fw, bw)

    # -- .nhf --
    def _load_nhf(self):
        spec = self._reader.spec
        if spec is None:
            raise SpecError(f"{self.path.name} holds no spectroscopy curves.")
        self._spec = spec
        for name in spec.segments:
            channels = spec.channels(name)
            units = {ch: self._reader.units.get(("Spec", name, ch), "") for ch in channels}
            self.segments.append(Segment(name, segment_kind(name), channels, units))
        self.n_curves = int(spec.n_points)
        self.spring_constant = float(spec.spring_constant or math.nan)
        self.sensitivity = float(spec.sensitivity or math.nan)
        ny, nx = spec.shape
        coordinates = spec.coordinates()
        if coordinates is not None:
            self.positions = np.column_stack(coordinates)[:self.n_curves]
        if nx * ny > 1:
            self.grid = (int(nx), int(ny))
            if self.positions is None:
                size_x, size_y = spec.size
                self.positions = _snake_positions(self.n_curves, nx, ny, (0, size_x, 0, size_y))
        self.info = {"File": {"Name": self.path.name, "Read with": "NSFopen"},
                     "Probe": {"Spring constant": f"{self.spring_constant:g} N/m",
                               "Deflection sensitivity": f"{self.sensitivity:g} m/V"},
                     "Measurement": {k: str(v) for k, v in dict(self._reader.param).items()
                                     if isinstance(v, (str, int, float))}}

    # -- access --
    def segment(self, name: str) -> Segment:
        for segment in self.segments:
            if segment.name == name:
                return segment
        raise SpecError(f"No segment '{name}'.")

    def segments_of(self, kind: str) -> list:
        return [s for s in self.segments if s.kind == kind]

    def unit(self, segment: str, channel: str) -> str:
        if channel == RAMP:
            return "V" if "volt" in self.modulated.lower() else "m"
        return self.segment(segment).units.get(channel, "")

    def data(self, segment: str, channel: str, n: int) -> np.ndarray:
        """ One curve of one channel (SI units as stored; deflection of .nhf files in N if it can be converted). """
        key = (segment, channel, n)
        if key in self._cache:
            return self._cache[key]
        if not 0 <= n < self.n_curves:
            raise SpecError(f"Curve {n + 1} does not exist ({self.n_curves} curves).")
        if channel == RAMP:
            length = len(self.data(segment, self.segment(segment).channels[0], n))
            start, span = self._ramps.get(segment, (0.0, float(length - 1)))
            values = start + np.linspace(0.0, span, length)
        elif getattr(self, "_arrays", None) is not None:
            values = np.asarray(self._arrays[segment][channel][n], dtype=float)
        elif self.path.suffix.lower() == ".nid":
            values = np.asarray(self._reader.data["Spec"][segment][channel][n], dtype=float)
        else:
            if channel.lower().startswith("deflection"):
                try:
                    values = np.asarray(self._spec.deflection(segment, n, "N"), dtype=float)
                except Exception:
                    values = np.asarray(self._spec[segment][channel][n], dtype=float)
            else:
                values = np.asarray(self._spec[segment][channel][n], dtype=float)
        if len(self._cache) > 4000:
            self._cache.clear()
        self._cache[key] = values
        return values

    def x_channels(self, segment: str) -> list:
        channels = list(self.segment(segment).channels)
        return channels + ([RAMP] if segment in self._ramps else [])

    def default_channels(self) -> tuple:
        """ (x channel, y channel, kind) guessed from the channel names: kind 'force' or 'iv' or 'other'. """
        segment = (self.segments_of(APPROACH) or self.segments)[0]
        names = self.x_channels(segment.name)
        lower = {n.lower(): n for n in names}
        current = next((n for n in names if "current" in n.lower() or segment.units.get(n) == "A"), None)
        voltage = next((n for n in names if n != current and ("volt" in n.lower() or "bias" in n.lower())), None)
        if current and ("volt" in self.modulated.lower() or voltage):
            return (voltage or (RAMP if RAMP in names else names[0])), current, "iv"
        z = next((lower[n] for n in ("z-axis sensor", "z-axissensor", "position z", "z-axis", "z") if n in lower), None)
        deflection = next((n for n in names if n.lower().startswith("deflection")), None)
        if deflection:
            return (z or (RAMP if RAMP in names else names[0])), deflection, "force"
        return names[0], names[-1], "other"

    def close(self):
        if self._reader is not None and hasattr(self._reader, "close"):
            self._reader.close()


RAMP = "Ramp (from the settings)"          # x channel built from the ramp of the file when it is not stored


def _snake_positions(n: int, nx: int, ny: int, extent: tuple) -> np.ndarray:
    """ Positions of the curves of a grid measured line by line, every other line backward (row 0 = first line). """
    x0, x1, y0, y1 = extent
    xs = np.linspace(x0, x1, nx) if nx > 1 else np.array([0.5 * (x0 + x1)])
    ys = np.linspace(y0, y1, ny) if ny > 1 else np.array([0.5 * (y0 + y1)])
    index = np.arange(n)
    row, col = index // nx, index % nx
    col = np.where(row % 2 == 1, nx - 1 - col, col)
    row = np.minimum(row, ny - 1)
    return np.column_stack([xs[col], ys[row]])


def grid_cell(n: int, nx: int) -> tuple:
    """ (row, col) of curve n of a snake grid. """
    row, col = divmod(int(n), nx)
    return row, (nx - 1 - col if row % 2 else col)


def grid_index(row: int, col: int, nx: int) -> int:
    return row * nx + (nx - 1 - col if row % 2 else col)


def to_map(values, n: int, grid: tuple) -> np.ndarray:
    """ Values in curve order -> (ny, nx) map, row 0 = first line (y0); missing curves NaN. """
    nx, ny = grid
    out = np.full((ny, nx), np.nan)
    for i, value in enumerate(values[:n]):
        row, col = grid_cell(i, nx)
        if row < ny:
            out[row, col] = value
    return out


def valid_channels(spec: "SpecFile", x: str, y: str, default_x: str, default_y: str) -> tuple:
    names = set()
    for segment in spec.segments:
        names.update(spec.x_channels(segment.name))
    return (x if x in names else default_x), (y if y in names else default_y)


def subset(spec: "SpecFile", step: int = 1, region: Optional[tuple] = None) -> list:
    """ Curves to analyze: every step-th line and column of a grid (every step-th curve otherwise), within region =
        (first row, last row, first column, last column) of the grid if given. A coarse map first, then a region at
        full resolution, takes a fraction of the time of the whole map. """
    step = max(int(step), 1)
    if spec.grid is None:
        return list(range(0, spec.n_curves, step))
    nx, ny = spec.grid
    r0, r1, c0, c1 = region if region is not None else (0, ny - 1, 0, nx - 1)
    out = []
    for row in range(max(r0, 0), min(r1, ny - 1) + 1, step):
        for col in range(max(c0, 0), min(c1, nx - 1) + 1, step):
            n = grid_index(row, col, nx)
            if n < spec.n_curves:
                out.append(n)
    return sorted(out)


def full_grid_origin(spec: "SpecFile") -> tuple:
    """ (x, y) of the lower left corner of a map (half a step outside its first curve), the origin of the positions
        when they are shown from 0 like the axes of images. """
    if spec.positions is None:
        return 0.0, 0.0
    x0, y0 = spec.positions.min(axis=0)
    if spec.grid is None:
        return float(x0), float(y0)
    x1, y1 = spec.positions.max(axis=0)
    nx, ny = spec.grid
    hx = (x1 - x0) / (nx - 1) / 2 if nx > 1 else 0.0
    hy = (y1 - y0) / (ny - 1) / 2 if ny > 1 else 0.0
    return float(x0 - hx), float(y0 - hy)


@dataclasses.dataclass
class MapView:
    """ The part of a grid that was analyzed: the lines and columns shown (a coarse grid for a step, a region). """
    rows: np.ndarray                  # grid lines shown, increasing
    cols: np.ndarray                  # grid columns shown, increasing
    nx: int                           # columns of the whole grid
    positions: np.ndarray             # positions of all curves

    @property
    def grid(self) -> tuple:
        return len(self.cols), len(self.rows)

    def index(self, r: int, c: int) -> int:
        return grid_index(int(self.rows[r]), int(self.cols[c]), self.nx)

    def cell(self, n: int) -> Optional[tuple]:
        row, col = grid_cell(n, self.nx)
        r, c = np.searchsorted(self.rows, row), np.searchsorted(self.cols, col)
        if r < len(self.rows) and c < len(self.cols) and self.rows[r] == row and self.cols[c] == col:
            return int(r), int(c)
        return None

    def to_map(self, per_curve: np.ndarray) -> np.ndarray:
        """ Values given per curve (index = curve - 1) as the map of the view, row 0 = lowest line. """
        out = np.full((len(self.rows), len(self.cols)), np.nan)
        for r in range(len(self.rows)):
            for c in range(len(self.cols)):
                n = self.index(r, c)
                if n < len(per_curve):
                    out[r, c] = per_curve[n]
        return out

    def cell_positions(self) -> np.ndarray:
        corners = [self.index(0, 0), self.index(len(self.rows) - 1, len(self.cols) - 1)]
        return self.positions[[n for n in corners if n < len(self.positions)]]

    def extent(self) -> tuple:
        """ (x0, x1, y0, y1): positions of the first and the last curve shown. """
        first, last = self.cell_positions()[[0, -1]]
        return float(first[0]), float(last[0]), float(first[1]), float(last[1])


def map_view(spec: "SpecFile", indices) -> Optional[MapView]:
    if spec.grid is None or spec.positions is None:
        return None
    cells = [grid_cell(n, spec.grid[0]) for n in indices]
    if not cells:
        return None
    rows = np.unique([r for r, _ in cells])
    cols = np.unique([c for _, c in cells])
    return MapView(rows, cols, spec.grid[0], spec.positions)


# ---- processing ----

def smooth(y: np.ndarray, kind: str, window: int) -> np.ndarray:
    """ Filter of a curve: moving average, median or Savitzky-Golay (2nd order) over an odd window of points. """
    y = np.asarray(y, dtype=float)
    window = int(window) | 1
    if kind == "none" or window < 3 or y.size < 3:
        return y
    window = min(window, y.size - (1 - y.size % 2))
    if window < 3:
        return y
    if kind == "mean":
        return ndimage.uniform_filter1d(y, window, mode="nearest")
    if kind == "median":
        return ndimage.median_filter(y, size=window, mode="nearest")
    if kind == "savgol":
        return savgol_filter(y, window, min(2, window - 1), mode="interp")
    raise SpecError(f"Unknown filter '{kind}'.")


@dataclasses.dataclass
class Baseline:
    kind: str
    offset: float = 0.0
    slope: float = 0.0
    amplitude: float = 0.0
    period: float = math.nan
    phase: float = 0.0
    region: tuple = (math.nan, math.nan)        # x range used

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        value = self.offset + self.slope * x
        if self.amplitude and math.isfinite(self.period):
            value = value + self.amplitude * np.sin(2.0 * math.pi * x / self.period + self.phase)
        return value


def fit_baseline(x: np.ndarray, y: np.ndarray, kind: str, region: tuple, period: Optional[float] = None) -> Baseline:
    """ Baseline of the part of a curve far from the sample: the points with region[0] <= x <= region[1].
        kind: none, offset, linear, sine (offset, tilt and a sine of the interference of the laser reflected by the
        sample, whose period in x is about half the wavelength; searched between 4 points and the region length if
        not given). """
    if kind == "none":
        return Baseline("none")
    use = (x >= min(region)) & (x <= max(region)) & np.isfinite(y)
    if use.sum() < 4:
        raise SpecError("The baseline region holds too few points: widen it.")
    xr, yr = x[use], y[use]
    if kind == "offset":
        return Baseline(kind, float(np.median(yr)), region=region)
    slope, offset = np.polyfit(xr, yr, 1)
    if kind == "linear":
        return Baseline(kind, float(offset), float(slope), region=region)
    if kind != "sine":
        raise SpecError(f"Unknown baseline '{kind}'.")
    center = float(xr.mean())
    length = float(xr.max() - xr.min())
    step = length / max(xr.size - 1, 1)

    def solve(p):
        phase = 2.0 * math.pi * (xr - center) / p
        design = np.column_stack([np.ones_like(xr), xr - center, np.sin(phase), np.cos(phase)])
        coef, *_ = np.linalg.lstsq(design, yr, rcond=None)
        return coef, float(np.sum((design @ coef - yr) ** 2))

    if period is None:
        candidates = np.geomspace(max(4.0 * step, length / 40.0), length * 1.5, 160)
        costs = [solve(p)[1] for p in candidates]
        best = int(np.argmin(costs))
        lo, hi = candidates[max(best - 1, 0)], candidates[min(best + 1, len(candidates) - 1)]
        fine = np.linspace(lo, hi, 60)
        period = float(fine[int(np.argmin([solve(p)[1] for p in fine]))])
    coef, _ = solve(period)
    c0, c1, s, c = (float(v) for v in coef)
    amplitude = math.hypot(s, c)
    phase = math.atan2(c, s) - 2.0 * math.pi * center / period
    return Baseline(kind, c0 - c1 * center, c1, amplitude, period, phase, region)


@dataclasses.dataclass
class ForceSettings:
    x_channel: str = ""                     # empty: guessed
    y_channel: str = ""
    spring_constant: float = math.nan       # N/m; NaN: from the file
    sensitivity: float = math.nan           # m/V; NaN: from the file
    filter: str = "none"
    window: int = 7
    baseline: str = "linear"
    baseline_from: float = 0.0              # % of the approach range, from the far end
    baseline_to: float = 40.0
    model: str = "hertz"
    shape: str = "sphere"
    radius: float = 10e-9
    half_angle: float = 20.0
    poisson: float = 0.5
    fit_segment: str = "auto"               # auto (model default), approach, retract
    fit_max_force: float = 100.0            # %: approach fits use the points up to this share of the maximum force
    op_lower: float = 20.0                  # %: Oliver-Pharr fit range of the unloading curve
    op_upper: float = 95.0
    thickness_mode: str = "off"             # bottom effect: off (thick sample), fixed, map (from the contact height)
    thickness: float = 5e-6                 # m: the sample thickness (fixed)
    substrate: float = 0.0                  # m: height (contact) of the substrate (map: thickness = height - this)
    bonded: bool = True                     # the sample is bonded to the substrate

    def tip(self, thickness: float = math.inf) -> contact.Tip:
        return contact.Tip(self.shape, self.radius, self.half_angle, self.poisson, thickness, bool(self.bonded))

    def thickness_at(self, contact_x: float) -> float:
        """ The sample thickness for the bottom-effect correction at a curve whose contact is at the piezo position
            contact_x: inf (off), the fixed value, or (map) its height above the substrate, i.e.
            -contact_x - substrate. NaN: the curve has no contact point (map); <= 0: on the substrate. """
        if self.thickness_mode == "fixed":
            return float(self.thickness)
        if self.thickness_mode == "map":
            return -float(contact_x) - float(self.substrate) if math.isfinite(contact_x) else math.nan
        return math.inf


@dataclasses.dataclass
class SegmentData:
    x: np.ndarray                  # piezo position towards the sample, m
    force: np.ndarray              # N, baseline removed
    raw: np.ndarray                # N, filtered, before the baseline
    baseline: Baseline
    delta: np.ndarray              # x - F/k, m


@dataclasses.dataclass
class ForceCurve:
    index: int
    approach: Optional[SegmentData]
    retract: Optional[SegmentData]
    spring_constant: float
    contact_x: float = math.nan    # piezo position of the contact, m
    contact_delta: float = math.nan
    values: dict = dataclasses.field(default_factory=dict)       # name -> (value, unit)
    fit: Optional[contact.FitResult] = None
    fit_segment: str = ""
    error: str = ""


def _to_force(y: np.ndarray, unit: str, k: float, sensitivity: float) -> np.ndarray:
    if unit == "N":
        return y
    if unit == "m":
        return y * k
    if unit == "V":
        if not math.isfinite(sensitivity):
            raise SpecError("The deflection is in V: set the deflection sensitivity.")
        return y * sensitivity * k
    raise SpecError(f"The deflection unit '{unit}' cannot be converted to a force.")


def _orientation(spec: SpecFile, settings: ForceSettings, approach: str, x_channel: str, y_channel: str) -> float:
    """ +1 if x grows towards the sample, -1 if it shrinks: from the first curve, whose end with the larger force
        is the one near the sample. """
    x = spec.data(approach, x_channel, 0)
    y = spec.data(approach, y_channel, 0)
    n = min(x.size, y.size)
    if n < 4:
        return 1.0
    x, y = x[:n], y[:n]
    tenth = max(n // 10, 1)
    near_last = np.nanmean(y[-tenth:]) > np.nanmean(y[:tenth])
    step = x[-1] - x[0]
    return math.copysign(1.0, step if near_last else -step) if step else 1.0


class ForceAnalysis:
    """ Processing and fit of the force curves of a file with one settings object. """

    def __init__(self, spec: SpecFile, settings: Optional[ForceSettings] = None):
        self.spec = spec
        self.settings = settings or ForceSettings()
        self._sign = None

    @property
    def spring_constant(self) -> float:
        k = self.settings.spring_constant
        return k if math.isfinite(k) and k > 0 else self.spec.spring_constant

    @property
    def sensitivity(self) -> float:
        s = self.settings.sensitivity
        return s if math.isfinite(s) and s > 0 else self.spec.sensitivity

    def channels(self) -> tuple:
        """ The x and y channel of the settings, or the guessed ones if the file does not have them (settings saved
            with another file, e.g. 'Position Z' of Studio files for a .nid file). """
        x, y, _ = self.spec.default_channels()
        return valid_channels(self.spec, self.settings.x_channel, self.settings.y_channel, x, y)

    def segment_names(self) -> tuple:
        approach = self.spec.segments_of(APPROACH)
        retract = self.spec.segments_of(RETRACT)
        return (approach[0].name if approach else None), (retract[0].name if retract else None)

    def _segment(self, name: str, n: int, sign: float, far_first: bool, k: float, period=None) -> SegmentData:
        s = self.settings
        x_channel, y_channel = self.channels()
        x = spec_x = self.spec.data(name, x_channel, n) * sign
        y = self.spec.data(name, y_channel, n)
        m = min(spec_x.size, y.size)
        x, y = x[:m], y[:m]
        force = _to_force(y, self.spec.unit(name, y_channel), k, self.sensitivity)
        if m >= 2 and (x[0] > x[-1]) == far_first:              # time order: approach far -> near, retract near -> far
            x, force = x[::-1], force[::-1]
        good = np.isfinite(x) & np.isfinite(force)
        x, force = x[good], force[good]
        raw = smooth(force, s.filter, s.window)
        if x.size < 4:
            raise SpecError("The curve has too few points.")
        x_far, x_near = float(x.min()), float(x.max())
        span = x_near - x_far
        region = (x_far + span * s.baseline_from / 100.0, x_far + span * s.baseline_to / 100.0)
        baseline = fit_baseline(x, raw, s.baseline, region, period)
        corrected = raw - baseline(x)
        return SegmentData(x, corrected, raw, baseline, x - corrected / k)

    def curve(self, n: int, fit: bool = True) -> ForceCurve:
        """ The processed curve n (0-based) with its measurements and, if fit, the model fit. """
        k = self.spring_constant
        if not (math.isfinite(k) and k > 0):
            raise SpecError("Set the spring constant: the file has none.")
        approach_name, retract_name = self.segment_names()
        if approach_name is None and retract_name is None:
            raise SpecError("The file has no approach or retract segment.")
        x_channel, y_channel = self.channels()
        if self._sign is None:
            self._sign = _orientation(self.spec, self.settings, approach_name or retract_name, x_channel, y_channel)
        approach = self._segment(approach_name, n, self._sign, True, k) if approach_name else None
        period = approach.baseline.period if (approach and self.settings.baseline == "sine") else None
        retract = None
        if retract_name:
            try:
                retract = self._segment(retract_name, n, self._sign, False, k,
                                        period if period and math.isfinite(period) else None)
            except SpecError:
                if approach is None:
                    raise
        curve = ForceCurve(n, approach, retract, k)
        _measure(curve)
        if fit:
            self.fit(curve)
        return curve

    def fit(self, curve: ForceCurve):
        s = self.settings
        thickness = s.thickness_at(curve.contact_x)
        if s.thickness_mode != "off" and s.model not in contact.BOTTOM_MODELS:
            curve.error = (f"The bottom-effect correction is for Hertz, Sneddon and DMT, not "
                           f"{dict(contact.MODELS)[s.model]}: set the sample to thick.")
            return
        if s.thickness_mode == "map" and not math.isfinite(thickness):
            curve.error = "No contact point on the approach curve: the sample thickness is not known."
            return
        if thickness <= 0:                       # on (or below) the substrate: nothing to correct
            thickness = math.inf
        tip = s.tip(thickness)
        segment = contact.MODEL_SEGMENT[s.model] if s.fit_segment == "auto" else s.fit_segment
        curve.fit_segment = segment
        data = curve.approach if segment == APPROACH else curve.retract
        if data is None:
            curve.error = f"The file has no {segment} segment for the fit."
            return
        try:
            if s.model in ("hertz", "sneddon"):
                delta, force = _contact_part(data, segment, curve.contact_delta, s.fit_max_force)
                curve.fit = contact.fit_elastic(delta, force, tip, s.model)
            elif s.model == "dmt":
                delta, force = _retract_to_pull_off(data)
                curve.fit = contact.fit_elastic(delta, force, tip, "dmt")
            elif s.model == "jkr":
                delta, force = _retract_through_pull_off(data)
                curve.fit = contact.fit_jkr(delta, force, tip)
            elif s.model == "maugis":
                delta, force = _retract_through_pull_off(data)
                curve.fit = contact.fit_maugis(delta, force, tip)
            elif s.model == "oliver_pharr":
                zero = curve.contact_delta
                if not math.isfinite(zero):
                    raise contact.FitError("No contact point on the approach curve (needed for the depth).")
                curve.fit = contact.fit_oliver_pharr(data.delta, data.force, tip, zero, s.op_upper / 100.0,
                                                     s.op_lower / 100.0)
            else:
                raise contact.FitError(f"Unknown model '{s.model}'.")
        except contact.FitError as e:
            curve.error = str(e)
            curve.fit = None
            return
        curve.values.update(curve.fit.values())

    def run(self, indices=None, progress: Optional[Callable[[int, int], bool]] = None) -> "Results":
        """ Every curve (or the given ones): their values. progress(done, total) returning False stops. """
        indices = list(range(self.spec.n_curves) if indices is None else indices)
        rows, curves_failed = [], 0
        for done, n in enumerate(indices, start=1):
            try:
                curve = self.curve(n)
                row = {name: value for name, (value, _) in curve.values.items()}
                row["error"] = curve.error
                units = {name: unit for name, (_, unit) in curve.values.items()}
            except SpecError as e:
                row, units = {"error": str(e)}, {}
                curves_failed += 1
            row["curve"] = n + 1
            rows.append((row, units))
            if progress is not None and progress(done, len(indices)) is False:
                break
        units = {}
        for _, u in rows:
            units.update(u)
        return Results([r for r, _ in rows], units, self.spec)


def _contact_part(data: SegmentData, segment: str, contact_delta: float, max_force_percent: float) -> tuple:
    """ Approach points for an elastic fit: from before the contact (one contact-range back, so the contact point is
        free) up to the given share of the maximum force. """
    order = np.arange(data.delta.size) if segment == APPROACH else np.arange(data.delta.size)[::-1]
    delta, force = data.delta[order], data.force[order]
    top = int(np.argmax(force))
    delta, force = delta[:top + 1], force[:top + 1]
    limit = max_force_percent / 100.0 * force.max()
    keep = force <= limit + 1e-30
    if math.isfinite(contact_delta):
        reach = float(delta.max() - contact_delta)
        keep &= delta >= contact_delta - max(reach, 0.0)
    return delta[keep], force[keep]


def _retract_to_pull_off(data: SegmentData) -> tuple:
    start, stop = int(np.argmax(data.force)), int(np.argmin(data.force))
    if stop <= start + 3:
        stop = data.force.size - 1
    return data.delta[start:stop + 1], data.force[start:stop + 1]


def _retract_through_pull_off(data: SegmentData) -> tuple:
    start, low = int(np.argmax(data.force)), int(np.argmin(data.force))
    extra = max((low - start) // 3, 5)
    return data.delta[start:min(low + extra, data.force.size)], data.force[start:min(low + extra, data.force.size)]


def _crossing(x: np.ndarray, y: np.ndarray, index: int) -> float:
    """ x where y crosses zero between index - 1 and index (linear interpolation). """
    if index <= 0:
        return float(x[0])
    y0, y1 = y[index - 1], y[index]
    t = 0.0 if y1 == y0 else -y0 / (y1 - y0)
    return float(x[index - 1] + min(max(t, 0.0), 1.0) * (x[index] - x[index - 1]))


def _measure(curve: ForceCurve):
    """ Values of a curve that need no model. """
    values = curve.values
    a, r = curve.approach, curve.retract
    if a is not None:
        top = int(np.argmax(a.force))
        values["max force"] = (float(a.force[top]), "N")
        below = np.nonzero(a.force[:top + 1] <= 0)[0]
        if below.size and below[-1] < top:
            i = int(below[-1]) + 1
            curve.contact_x = _crossing(a.x, a.force, i)
            curve.contact_delta = _crossing(a.delta, a.force, i)
            values["contact point (piezo)"] = (curve.contact_x, "m")
            values["height (contact)"] = (-curve.contact_x, "m")
            values["max indentation"] = (float(a.delta[top] - curve.contact_delta), "m")
        values["snap-in force"] = (max(-float(a.force[:top + 1].min()), 0.0), "N")
        upper = a.force >= 0.5 * a.force[top]
        if upper.sum() >= 3:
            values["slope (top half)"] = (float(np.polyfit(a.x[upper], a.force[upper], 1)[0]), "N/m")
    if r is not None:
        low = int(np.argmin(r.force))
        values["adhesion force"] = (max(-float(r.force[low]), 0.0), "N")
        if math.isfinite(curve.contact_x):
            values["pull-off distance"] = (float(curve.contact_x - r.x[low]), "m")    # piezo retraction past contact
        attractive = np.minimum(r.force, 0.0)
        order = np.argsort(r.delta)
        values["adhesion energy"] = (float(-np.trapezoid(attractive[order], r.delta[order])), "J")
    if a is not None and r is not None:
        work_in = float(np.trapezoid(a.force[np.argsort(a.delta)], np.sort(a.delta)))
        work_out = float(np.trapezoid(r.force[np.argsort(r.delta)], np.sort(r.delta)))
        values["dissipated energy"] = (work_in - work_out, "J")


# ---- results of many curves ----

REGION = "model region"                # 0: the settings of all curves, 1, 2, ...: the regions fitted with other settings
MODEL_FIELDS = ("model", "shape", "radius", "half_angle", "poisson", "fit_segment", "fit_max_force", "op_lower",
                "op_upper", "thickness_mode", "thickness", "substrate", "bonded")            # what a model region sets; the processing (filter, baseline, k) is the same for all


def model_settings(settings: ForceSettings) -> dict:
    return {name: getattr(settings, name) for name in MODEL_FIELDS}


def complete_model(model: dict) -> dict:
    """ A model dict with every MODEL_FIELDS entry (older files lack the newer ones: their defaults). """
    defaults = model_settings(ForceSettings())
    return {name: model.get(name, defaults[name]) for name in MODEL_FIELDS}


def describe_model(model: dict) -> str:
    """ 'JKR, sphere R 20 nm, ν 0.5' (with ', thin 3 µm, bonded' or ', thin from the height' for the bottom
        effect) """
    from . import contact as _contact
    model = complete_model(model)
    shape = dict(_contact.SHAPES).get(model["shape"], model["shape"]).lower()
    geometry = []
    if model["shape"] in ("sphere", "hyperboloid", "punch"):
        geometry.append(f"R {format_value(model['radius'], 'm', 3)}")
    if model["shape"] in ("cone", "pyramid", "hyperboloid"):
        geometry.append(f"{model['half_angle']:g}°")
    text = f"{dict(_contact.MODELS)[model['model']]}, {shape} {' '.join(geometry)}, ν {model['poisson']:g}"
    if model["thickness_mode"] != "off":
        thin = (f"thin {format_value(model['thickness'], 'm', 3)}" if model["thickness_mode"] == "fixed"
                else f"thin from the height (substrate {format_value(model['substrate'], 'm', 3)})")
        text += f", {thin}, {'bonded' if model['bonded'] else 'not bonded'}"
    return text.replace("  ", " ")


def substrate_height(analysis: "ForceAnalysis", indices) -> float:
    """ Median height (contact) of the given curves, e.g. the masked ones on the substrate: the 'substrate' of the
        bottom-effect correction from the height. NaN if none has a contact point. """
    heights = []
    for n in indices:
        try:
            curve = analysis.curve(int(n), fit=False)
        except SpecError:
            continue
        if math.isfinite(curve.contact_x):
            heights.append(-curve.contact_x)
    return float(np.median(heights)) if heights else math.nan

class Results:
    """ The values of many curves: rows (dicts) in curve order, statistics and maps. """

    def __init__(self, rows: list, units: dict, spec: Optional[SpecFile] = None):
        self.rows = rows
        self.units = units
        self.spec = spec
        self.min_r2 = 0.0               # quality filter: curves whose fit has a lower R² (or failed) are left out

    def has_fits(self) -> bool:
        return any("R²" in row for row in self.rows)

    def left_out(self, row: dict) -> bool:
        """ True if the quality filter leaves the curve out: its fit failed or has R² below min_r2. """
        if self.min_r2 <= 0 or not self.has_fits():
            return False
        r2 = row.get("R²", math.nan)
        return not (isinstance(r2, float) and math.isfinite(r2) and r2 >= self.min_r2)

    def count_left_out(self) -> int:
        return sum(1 for row in self.rows if self.left_out(row))

    def merge(self, other: "Results", region: int, label: str):
        """ The rows of other (curves fitted with the settings of a region) replace the rows of the same curves. """
        by_curve = {row["curve"]: i for i, row in enumerate(self.rows)}
        for row in other.rows:
            row = dict(row, **{REGION: float(region), "model": label})
            if row["curve"] in by_curve:
                self.rows[by_curve[row["curve"]]] = row
            else:
                self.rows.append(row)
        self.rows.sort(key=lambda r: r["curve"])
        self.units.update(other.units)
        self.units[REGION] = ""
        self._view = None

    def names(self) -> list:
        seen = []
        for row in self.rows:
            for name in row:
                if name not in ("curve", "error", "model") and name not in seen:
                    seen.append(name)
        return seen

    def column(self, name: str) -> np.ndarray:
        """ A value of every row; NaN where missing or left out by the quality filter. """
        return np.array([math.nan if self.left_out(row) and name != REGION else row.get(name, math.nan)
                         for row in self.rows], dtype=float)

    def statistics(self, selection: Optional[np.ndarray] = None) -> list:
        """ [(name, unit, N, mean, std, median, min, max)] over the curves where the value is finite; selection: a
            boolean per row (e.g. the curves inside a mask), None = all. """
        out = []
        for name in self.names():
            if name == REGION:                                   # a label, not a measurement
                continue
            values = self.column(name)
            if selection is not None:
                values = values[np.asarray(selection, bool)[:values.size]]
            values = values[np.isfinite(values)]
            if values.size:
                out.append((name, self.units.get(name, ""), int(values.size), float(values.mean()),
                            float(values.std(ddof=1)) if values.size > 1 else 0.0, float(np.median(values)),
                            float(values.min()), float(values.max())))
            else:
                out.append((name, self.units.get(name, ""), 0) + (math.nan,) * 5)
        return out

    def view(self) -> Optional[MapView]:
        """ The analyzed part of the grid (None if the file is not a map). """
        if self.spec is None:
            return None
        if getattr(self, "_view", None) is None:
            self._view = map_view(self.spec, [row["curve"] - 1 for row in self.rows])
        return self._view

    def per_curve(self, name: str) -> np.ndarray:
        """ A value for every curve of the file (NaN for the curves not analyzed). """
        values = np.full(self.spec.n_curves if self.spec is not None else len(self.rows), np.nan)
        for row in self.rows:
            values[row["curve"] - 1] = math.nan if self.left_out(row) and name != REGION else row.get(name, math.nan)
        return values

    def map(self, name: str) -> Optional[np.ndarray]:
        """ The map of a value over the analyzed part of the grid (row 0 = lowest line). """
        view = self.view()
        return view.to_map(self.per_curve(name)) if view is not None else None

    def failed(self) -> int:
        return sum(1 for row in self.rows if row.get("error"))

    def selection(self, curve_mask: Optional[np.ndarray], region: str) -> Optional[np.ndarray]:
        """ Boolean per row for a region ('all', 'masked', 'unmasked') of a mask given per curve (index = curve - 1). """
        if region.startswith("region:"):
            number = float(region.split(":", 1)[1])
            return np.array([row.get(REGION, 0.0) == number for row in self.rows])
        if region == "all" or curve_mask is None:
            return None
        inside = np.array([bool(curve_mask[row["curve"] - 1]) if row["curve"] - 1 < len(curve_mask) else False
                           for row in self.rows])
        return inside if region == "masked" else ~inside

    def write_csv(self, path, curve_mask: Optional[np.ndarray] = None):
        import csv
        names = self.names()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["curve"] + (["x [m]", "y [m]"] if self.spec is not None and self.spec.positions is not None
                                  else [])
            masked = ["masked"] if curve_mask is not None else []
            models = any("model" in row for row in self.rows)
            masked += ["model"] if models else []
            quality = self.min_r2 > 0 and self.has_fits()
            masked += [f"left out (R² < {self.min_r2:g})"] if quality else []
            writer.writerow(header + masked + [f"{n} [{self.units[n]}]" if self.units.get(n) else n for n in names]
                            + ["error"])
            for row in self.rows:
                position = []
                if self.spec is not None and self.spec.positions is not None:
                    position = [f"{v:.6g}" for v in self.spec.positions[row["curve"] - 1]]
                flag = [int(bool(curve_mask[row["curve"] - 1]))] if curve_mask is not None else []
                flag += [row.get("model", "")] if models else []
                flag += [int(self.left_out(row))] if quality else []
                writer.writerow([row["curve"]] + position + flag +
                                [f"{row[n]:.6g}" if isinstance(row.get(n), float) and math.isfinite(row[n])
                                 else "" for n in names] + [row.get("error", "")])


def average_curve(analysis: "ForceAnalysis", indices: list, segment: str = APPROACH, points: int = 400) -> tuple:
    """ Mean and standard deviation of the force of several curves on a common grid of the separation relative to
        each curve's contact point (curves without contact point are left out): (delta, mean, std, used). """
    curves = []
    for n in indices:
        try:
            c = analysis.curve(n, fit=False)
        except SpecError:
            continue
        data = c.approach if segment == APPROACH else c.retract
        if data is None or not math.isfinite(c.contact_delta):
            continue
        curves.append((data.delta - c.contact_delta, data.force))
    if not curves:
        raise SpecError("No curve with a contact point to average.")
    low = max(float(d.min()) for d, _ in curves)
    high = min(float(d.max()) for d, _ in curves)
    if not high > low:
        raise SpecError("The curves do not overlap.")
    grid = np.linspace(low, high, points)
    stack = np.array([np.interp(grid, d[np.argsort(d)], f[np.argsort(d)]) for d, f in curves])
    return grid, stack.mean(axis=0), stack.std(axis=0), len(curves)


_PREFIXES = [(1e-18, "a"), (1e-15, "f"), (1e-12, "p"), (1e-9, "n"), (1e-6, "µ"), (1e-3, "m"), (1.0, ""), (1e3, "k"), (1e6, "M"),
             (1e9, "G"), (1e12, "T")]


def format_value(value, unit: str, digits: int = 4) -> str:
    """ A value with an SI prefix for any unit (5 GPa, 250 aJ, 12 pA); plain for no unit, % and °. """
    if value is None or not isinstance(value, (int, float, np.floating, np.integer)) or not math.isfinite(value):
        return "n/a"
    if unit in ("", "%", "°"):
        return f"{value:.{digits}g}{(' ' + unit) if unit else ''}"
    if unit == "m²":
        for scale, label in ((1e-18, "nm²"), (1e-12, "µm²"), (1e-6, "mm²")):
            if abs(value) < scale * 1e5 or label == "mm²":
                return f"{value / scale:.{digits}g} {label}"
    magnitude = abs(value)
    scale, prefix = 1.0, ""
    if magnitude > 0:
        for s, p in _PREFIXES:
            if magnitude >= s * 0.999:
                scale, prefix = s, p
    return f"{value / scale:.{digits}g} {prefix}{unit}"
