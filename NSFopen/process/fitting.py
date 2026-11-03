# -*- coding: utf-8 -*-
# Ported from the Nanosurf Surface Analyzer (NanosurfImg), where it is covered by tests.
""" Fits of functions to histograms (height distributions): one to three Gaussians, a Lorentzian, a pseudo-Voigt.
No GUI code. The fit is a weighted least squares on the counts (Poisson weights), so the uncertainties are meaningful
for the counts of a histogram; for an image they are a lower limit because neighboring pixels are correlated. """

import dataclasses
import math
from typing import Optional

import numpy as np
from scipy import ndimage, optimize, signal

from . import ops

MODELS = {"gauss": "Gaussian", "gauss2": "2 Gaussians", "gauss3": "3 Gaussians", "lorentz": "Lorentzian",
          "voigt": "Pseudo-Voigt"}
FWHM_GAUSS = 2.0 * math.sqrt(2.0 * math.log(2.0))


def _gauss(x, area, center, sigma):
    return area / (sigma * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _lorentz(x, area, center, gamma):
    return area / math.pi * gamma / ((x - center) ** 2 + gamma ** 2)


def _voigt(x, area, center, fwhm, eta):
    """ Pseudo-Voigt: eta Lorentzian + (1 - eta) Gaussian of the same FWHM and area. """
    return eta * _lorentz(x, area, center, fwhm / 2) + (1 - eta) * _gauss(x, area, center, fwhm / FWHM_GAUSS)


@dataclasses.dataclass
class Peak:
    center: float
    center_error: float
    width: float                    # sigma (Gaussian), gamma = HWHM (Lorentzian), FWHM (pseudo-Voigt)
    width_error: float
    fwhm: float
    area: float                     # fraction of the counts in the fit range
    area_error: float
    eta: Optional[float] = None     # Lorentzian share of a pseudo-Voigt


@dataclasses.dataclass
class HistogramFit:
    model: str
    peaks: list                     # sorted by center
    low: float                      # fit range
    high: float
    x: np.ndarray                   # bin centers of the fitted histogram
    counts: np.ndarray              # counts per bin
    bin_width: float
    reduced_chi2: float
    params: np.ndarray              # internal (normalized) parameters, for curve()
    _scale: tuple = (0.0, 1.0)      # (offset, scale) of the normalized x axis

    def curve(self, x: np.ndarray, component: Optional[int] = None) -> np.ndarray:
        """ Expected counts per bin of the fit (or of one peak) at the values x. """
        offset, scale = self._scale
        u = (np.asarray(x, float) - offset) / scale
        values = _evaluate(self.model, u, self.params, component)
        return values * self.counts.sum() * self.bin_width / scale

    def summary(self, scale: float = 1.0, unit: str = "") -> str:
        """ Text of the result; values multiplied by scale and labeled with unit. """
        width_name = {"lorentz": "HWHM", "voigt": "FWHM"}.get(self.model, "σ")
        lines = [f"{MODELS[self.model]}, reduced χ² {self.reduced_chi2:.3g}"]
        for i, p in enumerate(self.peaks, start=1):
            text = (f"{i}: center {p.center * scale:.4g} ± {p.center_error * scale:.2g} {unit}, "
                    f"{width_name} {p.width * scale:.4g} ± {p.width_error * scale:.2g}")
            if self.model != "voigt":
                text += f", FWHM {p.fwhm * scale:.4g}"
            if len(self.peaks) > 1:
                text += f", share {100 * p.area:.1f} %"
            if p.eta is not None:
                text += f", Lorentz share {p.eta:.2f}"
            lines.append(text)
        if len(self.peaks) > 1:
            gaps = [self.peaks[i + 1].center - self.peaks[i].center for i in range(len(self.peaks) - 1)]
            errors = [math.hypot(self.peaks[i + 1].center_error, self.peaks[i].center_error)
                      for i in range(len(self.peaks) - 1)]
            lines.append("separation " + ", ".join(f"{g * scale:.4g} ± {e * scale:.2g}" for g, e in zip(gaps, errors))
                         + f" {unit}")
        return "\n".join(lines)


def _count(model: str) -> int:
    return {"gauss": 1, "gauss2": 2, "gauss3": 3}.get(model, 1)


def _evaluate(model: str, u: np.ndarray, params: np.ndarray, component: Optional[int] = None) -> np.ndarray:
    if model.startswith("gauss"):
        parts = [_gauss(u, *params[3 * i:3 * i + 3]) for i in range(_count(model))]
    elif model == "lorentz":
        parts = [_lorentz(u, *params)]
    else:
        parts = [_voigt(u, *params)]
    return parts[component] if component is not None else np.sum(parts, axis=0)


def _initial_peaks(u: np.ndarray, counts: np.ndarray, n: int) -> list:
    """ (center, width, share) of n starting peaks: the most prominent maxima of the lightly smoothed histogram with
        their half widths (peak_widths) and shares (height x width), completed by quantiles of the distribution if
        there are fewer maxima than peaks. """
    total = counts.sum()
    du = u[1] - u[0]
    smooth = ndimage.gaussian_filter1d(counts.astype(float), 1.0)
    padded = np.concatenate([[0.0], smooth, [0.0]])           # a peak at the edge of the range counts too
    found, props = signal.find_peaks(padded, prominence=0.01 * smooth.max())
    order = np.argsort(-props["prominences"])[:n] if len(found) else []
    widths = signal.peak_widths(padded, found[order], rel_height=0.5)[0] if len(order) else []
    chosen = [int(found[i]) - 1 for i in order]
    peaks = [(float(u[c]), max(float(w) * du / FWHM_GAUSS, du / 2)) for c, w in zip(chosen, widths)]
    mean = (u * counts).sum() / total
    spread = math.sqrt(max(((u - mean) ** 2 * counts).sum() / total, 1e-12))
    cumulative = np.cumsum(counts) / total
    k = 0
    while len(peaks) < n:                                  # fill up with quantiles
        q = (k + 0.5) / n
        center = float(u[min(int(np.searchsorted(cumulative, q)), len(u) - 1)])
        if all(abs(center - c) > du for c, _ in peaks):
            peaks.append((center, spread / n))
        k += 1
        if k > 4 * n:
            peaks.append((float(u[len(u) // 2]), spread / n))
    density = smooth / (total * du)
    result = []
    for center, width in sorted(peaks):
        index = min(max(int(round((center - u[0]) / du)), 0), len(u) - 1)
        result.append((center, width, float(density[index] * width * math.sqrt(2 * math.pi))))
    shares = sum(r[2] for r in result)
    return [(c, w, a / shares if shares > 1.0 else max(a, 0.01)) for c, w, a in result]


def fit_histogram(values: np.ndarray, model: str = "gauss", bins: int = 100, low: Optional[float] = None,
                  high: Optional[float] = None) -> HistogramFit:
    """ Fits a model to the histogram of the finite values between low and high (default: all). """
    if model not in MODELS:
        raise ops.OpError(f"Unknown model '{model}'.")
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size < 20:
        raise ops.OpError("Too few values for a fit.")
    low = float(values.min()) if low is None else float(low)
    high = float(values.max()) if high is None else float(high)
    if not high > low:
        raise ops.OpError("The fit range is empty (the upper limit must be above the lower one).")
    values = values[(values >= low) & (values <= high)]
    bins = int(bins)
    n_params = {"gauss": 3, "gauss2": 6, "gauss3": 9, "lorentz": 3, "voigt": 4}[model]
    if values.size < 20 or bins < n_params + 3:
        raise ops.OpError("Too few values or bins in the fit range.")
    counts, edges = np.histogram(values, bins=bins, range=(low, high))
    x = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    offset, scale = float(values.mean()), float(values.std()) or (high - low)
    u = (x - offset) / scale
    du = width / scale
    total = counts.sum()
    y = counts / (total * du)                           # density in normalized units: the areas are shares
    sigma_y = np.sqrt(np.maximum(counts, 1)) / (total * du)
    guesses = _initial_peaks(u, counts, _count(model))
    lo_u, hi_u = (low - offset) / scale, (high - offset) / scale
    if model.startswith("gauss"):
        p0, lower, upper = [], [], []
        for center, w, share in guesses:
            p0 += [share, center, max(w, du / 2)]
            lower += [0.0, lo_u, du / 4]
            upper += [2.0, hi_u, 2 * (hi_u - lo_u)]
    elif model == "lorentz":
        center, w, share = guesses[0]
        p0, lower, upper = [share, center, max(w, du / 2)], [0.0, lo_u, du / 4], [2.0, hi_u, 2 * (hi_u - lo_u)]
    else:
        center, w, share = guesses[0]
        p0 = [share, center, max(w * FWHM_GAUSS, du), 0.5]
        lower, upper = [0.0, lo_u, du / 2, 0.0], [2.0, hi_u, 4 * (hi_u - lo_u), 1.0]
    p0 = np.clip(p0, np.array(lower) + 1e-9, np.array(upper) - 1e-9)

    def model_fn(uu, *params):
        return _evaluate(model, uu, np.asarray(params))

    try:
        params, cov = optimize.curve_fit(model_fn, u, y, p0=p0, sigma=sigma_y, absolute_sigma=True,
                                         bounds=(lower, upper), maxfev=20000)
    except (RuntimeError, ValueError) as e:
        raise ops.OpError(f"The fit did not converge: {e}")
    errors = np.sqrt(np.clip(np.diag(cov), 0, None)) if np.all(np.isfinite(cov)) else np.full(len(params), np.nan)
    residual = (y - model_fn(u, *params)) / sigma_y
    dof = max(len(u) - len(params), 1)
    peaks = []
    if model.startswith("gauss"):
        for i in range(_count(model)):
            a, c, s = params[3 * i:3 * i + 3]
            ea, ec, es = errors[3 * i:3 * i + 3]
            peaks.append(Peak(c * scale + offset, ec * scale, s * scale, es * scale, s * scale * FWHM_GAUSS, a, ea))
    elif model == "lorentz":
        a, c, g = params
        ea, ec, eg = errors
        peaks.append(Peak(c * scale + offset, ec * scale, g * scale, eg * scale, 2 * g * scale, a, ea))
    else:
        a, c, f, eta = params
        ea, ec, ef, _ = errors
        peaks.append(Peak(c * scale + offset, ec * scale, f * scale, ef * scale, f * scale, a, ea, float(eta)))
    order = np.argsort([p.center for p in peaks])
    result = HistogramFit(model, [peaks[i] for i in order], low, high, x, counts, width,
                          float((residual ** 2).sum() / dof), np.asarray(params), (offset, scale))
    if model.startswith("gauss"):                      # internal parameters in the order of the sorted peaks
        result.params = np.concatenate([params[3 * i:3 * i + 3] for i in order])
    return result
