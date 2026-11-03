# -*- coding: utf-8 -*-
# Ported from the Nanosurf Surface Analyzer (NanosurfImg), where it is covered by tests.
""" Fits of the radially averaged PSD: where the spectrum changes regime, and the effective lateral resolution.

Models (f = spatial frequency in 1/m, fitted on log10(PSD) so that every decade counts the same, each bin weighted by
the square root of the number of Fourier coefficients in it):

- 'abc_noise': k-correlation (ABC) model plus a white noise floor, PSD = A / (1 + (B f)^2)^((C + 1) / 2) + N.
  The plateau at low f turns into the power law f^-(C+1) at the corner f_c = 1 / B (correlation length
  xi = B / 2 pi); the power law meets the noise floor N at f_n: features with a wavelength shorter than 1 / f_n are
  below the noise, so 1 / f_n is the effective resolution (the smallest resolved period).
- 'abc': the same without the noise floor (a clean image, or the noise cut off with From / To).
- 'broken': two power laws joined at f_t (a straight line on each side in log-log). Model free: f_t is where the
  slope changes, e.g. from the surface's power law to the flatter noise, or from the plateau to the power law.

No GUI code. """

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import optimize

from . import ops

MODELS = {"abc_noise": "ABC + noise floor", "abc": "ABC (k-correlation)", "broken": "Two power laws"}


@dataclass
class PsdFit:
    model: str
    f: np.ndarray                   # the fitted frequencies (1/m)
    params: dict                    # model parameters (SI)
    low: float                      # fitted range (1/m)
    high: float
    residual: float                 # rms of log10(data / model): 0.1 = 26 % scatter
    results: dict = field(default_factory=dict)     # derived quantities (SI), see summary

    def curve(self, f: np.ndarray, part: str = "all") -> np.ndarray:
        """ The model at f; part 'signal' leaves out the noise floor. """
        p = self.params
        if self.model == "broken":
            x = np.log10(f / p["f_t"])
            slope = np.where(x < 0, p["slope_low"], p["slope_high"])
            return p["P_t"] * 10.0 ** (slope * x)
        signal = p["A"] / (1.0 + (p["B"] * f) ** 2) ** ((p["C"] + 1.0) / 2.0)
        return signal + (p["N"] if self.model == "abc_noise" and part == "all" else 0.0)

    def summary(self, unit: str = "m") -> str:
        """ The result as text (lengths in nm or µm). """
        r, p = self.results, self.params

        def length(value: float) -> str:
            if not math.isfinite(value):
                return "-"
            return f"{value * 1e9:.4g} nm" if value < 1e-6 else f"{value * 1e6:.4g} µm"

        lines = [f"{MODELS[self.model]}, fitted {self.low * 1e-6:.3g}-{self.high * 1e-6:.3g} 1/µm, "
                 f"scatter {100 * (10 ** self.residual - 1):.0f} %"]
        if self.residual > 0.3:
            lines.append("Warning: the model describes this spectrum poorly (periodic structures give peaks; "
                         "narrow From / To, or use Two power laws)")
        if self.model == "broken":
            lines.append(f"Transition at {self.f_text(p['f_t'])} (period {length(1 / p['f_t'])}); slopes "
                         f"{p['slope_low']:.2f} below, {p['slope_high']:.2f} above")
            if math.isfinite(r.get("resolution", math.nan)):
                lines.append(f"Effective resolution: period {length(r['resolution'])} (features down to about "
                             f"{length(r['resolution'] / 2)}); finer detail is in the flatter part (noise)")
            return "\n".join(lines)
        hurst = r.get("hurst", math.nan)
        if r["f_c"] < self.low:
            lines.append(f"No plateau in the range: the corner lies below {self.f_text(self.low)} (the correlation "
                         f"length is longer than about {length(1 / (2 * math.pi * self.low))}; scan a larger area)")
        elif r["f_c"] > self.high:
            lines.append(f"No power law in the range: the corner lies above {self.f_text(self.high)}")
        else:
            lines.append(f"Corner (plateau to power law) at {self.f_text(r['f_c'])}: period {length(1 / r['f_c'])}, "
                         f"correlation length ξ = {length(r['xi'])}")
        lines.append(f"Power law slope {-(p['C'] + 1):.2f}" + (f", Hurst exponent H = {hurst:.2f}"
                                                                if 0 < hurst < 1 else ""))
        if unit == "m" and math.isfinite(r.get("rms", math.nan)) and r["f_c"] >= self.low:
            lines.append(f"Roughness of the model surface: {r['rms'] * 1e9:.3g} nm rms")
        if self.model == "abc_noise":
            noise = r.get("noise_rms", math.nan)
            noise_text = f" ({noise * 1e9:.3g} nm rms over the image)" if unit == "m" else ""
            if r.get("resolution") == math.inf:
                lines.append("The noise floor lies above the signal everywhere: no resolved structure")
            elif math.isfinite(r.get("f_n", math.nan)):
                lines.append(f"Signal meets the noise floor{noise_text} at {self.f_text(r['f_n'])}")
                lines.append(f"Effective resolution: period {length(r['resolution'])} (features down to about "
                             f"{length(r['resolution'] / 2)})")
            else:
                lines.append(f"The signal stays above the noise floor{noise_text} up to the pixel limit: the "
                             f"resolution is set by the pixel size (period {length(r['resolution'])})")
        return "\n".join(lines)

    @staticmethod
    def f_text(f: float) -> str:
        return f"{f * 1e-6:.3g} 1/µm"


def _weights(counts: Optional[np.ndarray], n: int) -> np.ndarray:
    if counts is None:
        return np.ones(n)
    w = np.sqrt(np.maximum(np.asarray(counts, float), 1.0))
    return w / w.mean()


def _fit_abc(x: np.ndarray, y: np.ndarray, w: np.ndarray, noise: bool) -> tuple:
    """ Least squares in log10 space; parameters log10 A, log10 B, C, log10 N. """
    a0 = np.median(y[:3])
    n0 = np.median(y[-max(3, len(y) // 10):]) if noise else -np.inf
    level = np.log10(10 ** a0 / 10.0)                   # where the PSD has dropped tenfold
    below = np.nonzero(y < level)[0]
    b0 = -x[below[0]] if below.size else -x[len(x) // 2]

    def model(q):
        signal = q[0] - (q[2] + 1.0) / 2.0 * np.log10(1.0 + 10.0 ** (2 * (q[1] + x)))
        return np.logaddexp(signal * np.log(10), q[3] * np.log(10)) / np.log(10) if noise else signal

    best = None
    for c0 in (1.5, 2.5, 4.0):
        for shift in (-0.5, 0.0, 0.5):
            start = [a0, b0 + shift, c0] + ([n0 - 0.3] if noise else [])
            lower = [a0 - 6, -x[-1] - 3, 0.05] + ([n0 - 8] if noise else [])
            upper = [a0 + 6, -x[0] + 3, 12.0] + ([y.max() + 1] if noise else [])
            start = np.clip(start, np.array(lower) + 1e-9, np.array(upper) - 1e-9)
            try:
                result = optimize.least_squares(lambda q: w * (model(q) - y), start, bounds=(lower, upper),
                                                x_scale="jac")
            except ValueError:
                continue
            if best is None or result.cost < best.cost:
                best = result
    if best is None:
        raise ops.OpError("The fit did not converge.")
    q = best.x
    params = {"A": 10.0 ** q[0], "B": 10.0 ** q[1], "C": float(q[2]), "N": 10.0 ** q[3] if noise else 0.0}
    residual = float(np.sqrt(np.mean((model(q) - y) ** 2)))
    return params, residual


def _fit_broken(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple:
    """ Two straight lines in log-log, continuous at x_t: grid search for x_t, then refined. """

    def lines(xt):
        d = x - xt
        lo, hi = np.minimum(d, 0.0), np.maximum(d, 0.0)
        design = np.column_stack([np.ones_like(x), lo, hi]) * w[:, None]
        coef, *_ = np.linalg.lstsq(design, w * y, rcond=None)
        return coef, float(np.sum((design @ coef - w * y) ** 2))

    candidates = x[2:-2]
    if candidates.size < 1:
        raise ops.OpError("Too few points for two power laws: widen the range.")
    costs = [lines(xt)[1] for xt in candidates]
    i = int(np.argmin(costs))
    lo_, hi_ = candidates[max(i - 1, 0)], candidates[min(i + 1, candidates.size - 1)]
    xt = optimize.minimize_scalar(lambda v: lines(v)[1], bounds=(lo_, hi_), method="bounded").x \
        if hi_ > lo_ else candidates[i]
    coef, _ = lines(xt)
    model = coef[0] + coef[1] * np.minimum(x - xt, 0) + coef[2] * np.maximum(x - xt, 0)
    params = {"f_t": 10.0 ** xt, "P_t": 10.0 ** coef[0], "slope_low": float(coef[1]), "slope_high": float(coef[2])}
    return params, float(np.sqrt(np.mean((model - y) ** 2)))


def fit_psd(f: np.ndarray, psd: np.ndarray, model: str = "abc_noise", low: Optional[float] = None,
            high: Optional[float] = None, counts: Optional[np.ndarray] = None, pixel: Optional[float] = None) -> PsdFit:
    """ Fits a model (MODELS) to the PSD between low and high (1/m; None: all). `pixel` (m) is the pixel size, the
        resolution limit when the signal never meets the noise. Raises ops.OpError. """
    if model not in MODELS:
        raise ops.OpError(f"Unknown model '{model}'.")
    f, psd = np.asarray(f, float), np.asarray(psd, float)
    use = np.isfinite(f) & np.isfinite(psd) & (f > 0) & (psd > 0)
    if low is not None:
        use &= f >= low
    if high is not None:
        use &= f <= high
    needed = {"abc_noise": 8, "abc": 6, "broken": 6}[model]
    if use.sum() < needed:
        raise ops.OpError(f"Only {int(use.sum())} PSD points in the range: at least {needed} are needed.")
    x, y = np.log10(f[use]), np.log10(psd[use])
    w = _weights(None if counts is None else np.asarray(counts)[use], x.size)
    if model == "broken":
        params, residual = _fit_broken(x, y, w)
    else:
        params, residual = _fit_abc(x, y, w, model == "abc_noise")
    result = PsdFit(model, f[use], params, float(f[use].min()), float(f[use].max()), residual)
    nyquist = 0.5 / pixel if pixel else float(f.max())
    if model == "broken":
        flatter = params["slope_high"] > params["slope_low"]
        result.results["resolution"] = 1.0 / params["f_t"] if flatter else math.nan
        return result
    A, B, C, N = params["A"], params["B"], params["C"], params["N"]
    r = result.results
    r["f_c"] = 1.0 / B
    r["xi"] = B / (2 * math.pi)
    r["hurst"] = (C - 1.0) / 2.0
    r["rms"] = math.sqrt(2 * math.pi * A / ((C - 1.0) * B * B)) if C > 1 else math.nan
    if model == "abc_noise":
        if pixel:
            r["noise_rms"] = math.sqrt(N) / pixel        # white noise over the (1/pixel)² frequency square
        if N < A:
            # A / (1 + (B f)^2)^((C+1)/2) = N
            fn = math.sqrt(max((A / N) ** (2.0 / (C + 1.0)) - 1.0, 0.0)) / B
        else:
            fn = 0.0
        if 0 < fn < nyquist:
            r["f_n"], r["resolution"] = fn, 1.0 / fn
        elif fn == 0:
            r["f_n"], r["resolution"] = math.nan, math.inf
        else:
            r["f_n"], r["resolution"] = math.nan, 1.0 / nyquist
    return result
