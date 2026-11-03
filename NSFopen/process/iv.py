# -*- coding: utf-8 -*-
# Part of NSFopen.process; the Nanosurf Surface Analyzer (Spectroscopy page) uses this module.
""" I-V spectroscopy (current against bias voltage): smoothing, the derivative dI/dV, the normalized conductance
(dI/dV)/(I/V), and the values of a curve: resistance and conductance at zero bias, the current offset, the gap around
zero bias where the current stays below a threshold, and the rectification ratio. No GUI code; SI units. """

import dataclasses
import math

import numpy as np
from scipy.signal import savgol_filter

from . import spectroscopy

PLOTS = [("iv", "I(V)"), ("didv", "dI/dV"), ("normalized", "(dI/dV) / (I/V)"), ("log", "log |I|"),
         ("fn", "Fowler-Nordheim: ln(I/V²) vs 1/V")]


@dataclasses.dataclass
class IVSettings:
    x_channel: str = ""
    y_channel: str = ""
    filter: str = "none"
    window: int = 7
    derivative_window: int = 11          # points of the Savitzky-Golay derivative
    zero_window: float = 0.1             # V: ± range around 0 V for the zero-bias conductance
    gap_threshold: float = 10e-12        # A: |I| below this counts as gap
    offset: bool = False                 # subtract the current at 0 V (amplifier offset)
    segments: str = "both"               # forward, backward, both (mean of both on a common voltage grid)


@dataclasses.dataclass
class IVCurve:
    index: int
    voltage: np.ndarray
    current: np.ndarray
    didv: np.ndarray
    values: dict = dataclasses.field(default_factory=dict)
    error: str = ""

    def plot(self, kind: str) -> tuple:
        """ (x, y, x label, y label, y unit) of one of PLOTS. """
        v, i = self.voltage, self.current
        if kind == "iv":
            return v, i, "bias", "current", "A"
        if kind == "didv":
            return v, self.didv, "bias", "dI/dV", "S"
        with np.errstate(divide="ignore", invalid="ignore"):
            if kind == "normalized":
                conductance = i / v
                broad = np.sqrt(conductance ** 2 + (np.nanmax(np.abs(conductance)) * 1e-3) ** 2)
                return v, self.didv / broad, "bias", "(dI/dV)/(I/V)", ""
            if kind == "log":
                return v, np.log10(np.abs(i)), "bias", "log10 |I / A|", ""
            if kind == "fn":
                keep = (np.abs(v) > 1e-6) & (np.abs(i) > 0)
                return 1.0 / v[keep], np.log(np.abs(i[keep]) / v[keep] ** 2), "1/V", "ln(|I|/V²)", ""
        raise ValueError(kind)


def derivative(v: np.ndarray, i: np.ndarray, window: int) -> np.ndarray:
    """ dI/dV by a Savitzky-Golay derivative on a uniform voltage grid (the points are resampled if needed). """
    order = np.argsort(v)
    v, i = v[order], i[order]
    window = max(int(window) | 1, 5)
    if v.size < window:
        return np.gradient(i, v)
    grid = np.linspace(v[0], v[-1], v.size)
    step = (grid[-1] - grid[0]) / (grid.size - 1) if grid.size > 1 else 1.0
    di = savgol_filter(np.interp(grid, v, i), window, 2, deriv=1, delta=step if step else 1.0)
    return np.interp(v, grid, di)


def analyze(voltage, current, settings: IVSettings) -> dict:
    """ Values of one I-V curve (sorted by voltage). """
    v, i = np.asarray(voltage, float), np.asarray(current, float)
    values = {}
    if v.size >= 2:
        slope, intercept = np.polyfit(v, i, 1)
        values["resistance (linear fit)"] = (1.0 / slope if slope else math.inf, "Ω")
    near = np.abs(v) <= settings.zero_window
    if near.sum() >= 3:
        g0, i0 = np.polyfit(v[near], i[near], 1)
        values["zero-bias conductance"] = (float(g0), "S")
        values["zero-bias resistance"] = (1.0 / g0 if g0 else math.inf, "Ω")
        values["current at 0 V"] = (float(i0), "A")
    small = np.abs(i) < settings.gap_threshold
    if not small.any() and v.min() < 0 < v.max():
        values["gap width"] = (0.0, "V")
    elif v.min() < 0 < v.max():
        # the connected run of small current that contains 0 V
        order = np.argsort(v)
        vs, ss = v[order], small[order]
        zero = int(np.argmin(np.abs(vs)))
        if ss[zero]:
            lo = zero
            while lo > 0 and ss[lo - 1]:
                lo -= 1
            hi = zero
            while hi < vs.size - 1 and ss[hi + 1]:
                hi += 1
            values["gap width"] = (float(vs[hi] - vs[lo]), "V")
            values["gap from"], values["gap to"] = (float(vs[lo]), "V"), (float(vs[hi]), "V")
        else:
            values["gap width"] = (0.0, "V")
    if v.min() < 0 < v.max():
        top = min(v.max(), -v.min())
        plus, minus = np.interp(top, v, i), np.interp(-top, v, i)
        if minus != 0:
            values["rectification |I(+V)/I(-V)|"] = (abs(plus / minus), "")
    values["max |current|"] = (float(np.abs(i).max()), "A")
    return values


class IVAnalysis:
    def __init__(self, spec: spectroscopy.SpecFile, settings: IVSettings = None):
        self.spec = spec
        self.settings = settings or IVSettings()

    def channels(self) -> tuple:
        x, y, _ = self.spec.default_channels()
        return spectroscopy.valid_channels(self.spec, self.settings.x_channel, self.settings.y_channel, x, y)

    def _segment(self, name: str, n: int) -> tuple:
        x_channel, y_channel = self.channels()
        v = self.spec.data(name, x_channel, n)
        i = self.spec.data(name, y_channel, n)
        m = min(v.size, i.size)
        v, i = v[:m], spectroscopy.smooth(i[:m], self.settings.filter, self.settings.window)
        good = np.isfinite(v) & np.isfinite(i)
        order = np.argsort(v[good])
        return v[good][order], i[good][order]

    def curve(self, n: int) -> IVCurve:
        s = self.settings
        forward = self.spec.segments_of(spectroscopy.APPROACH)
        backward = self.spec.segments_of(spectroscopy.RETRACT)
        names = {"forward": forward[:1], "backward": backward[:1], "both": forward[:1] + backward[:1]}[s.segments]
        if not names:
            names = self.spec.segments[:1]
        parts = [self._segment(seg.name, n) for seg in names]
        if len(parts) == 1:
            v, i = parts[0]
        else:
            low = max(p[0].min() for p in parts)
            high = min(p[0].max() for p in parts)
            if not high > low:
                v, i = parts[0]
            else:
                v = np.linspace(low, high, max(p[0].size for p in parts))
                i = np.mean([np.interp(v, pv, pi) for pv, pi in parts], axis=0)
        if v.size < 3:
            raise spectroscopy.SpecError("The curve has too few points.")
        if s.offset and v.min() <= 0 <= v.max():
            i = i - np.interp(0.0, v, i)
        curve = IVCurve(n, v, i, derivative(v, i, s.derivative_window))
        curve.values = analyze(v, i, s)
        return curve

    def run(self, indices=None, progress=None) -> spectroscopy.Results:
        indices = list(range(self.spec.n_curves) if indices is None else indices)
        rows, units = [], {}
        for done, n in enumerate(indices, start=1):
            try:
                c = self.curve(n)
                row = {name: value for name, (value, _) in c.values.items()}
                units.update({name: unit for name, (_, unit) in c.values.items()})
                row["error"] = ""
            except spectroscopy.SpecError as e:
                row = {"error": str(e)}
            row["curve"] = n + 1
            rows.append(row)
            if progress is not None and progress(done, len(indices)) is False:
                break
        return spectroscopy.Results(rows, units, self.spec)
