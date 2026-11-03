# -*- coding: utf-8 -*-
# Ported from the Nanosurf Surface Analyzer (NanosurfImg), where it is covered by tests.
""" Roughness and waviness of a line profile after ISO 21920 (the successor of ISO 4287 / 4288), with the Gaussian
profile filter of ISO 16610-21.

The profile is split into
    primary profile P   = the measured profile minus its form (a line or a 2nd order polynomial), optionally
                          smoothed by the short cutoff λs,
    waviness W          = P filtered by the Gaussian low pass with the cutoff λc (long wavelengths),
    roughness R         = P - W (wavelengths shorter than λc).
The Gaussian weighting function is s(x) = exp(-π (x / (α λc))²) / (α λc) with α = sqrt(ln 2 / π), so that a sine of
wavelength λc keeps 50 % of its amplitude. Near the ends the filter sees only half its window, so the parameters are
evaluated without λc / 2 at each end (the run-up and run-out of a stylus instrument): the evaluation length is the
profile length minus λc. The filtered profiles are still computed over the whole line (renormalized weights at the
ends) for the display. Profiles shorter than three cutoffs are evaluated over their whole length, with a warning.

AFM profiles are much shorter than the ISO 4288 table assumes (0.08 to 8 mm cutoffs): the automatic cutoff is the
largest value of the series ..., 0.25, 0.8, 2.5, 8, 25, 80 µm, ... (steps of about 3.16) for which the evaluation
length holds five cutoffs (the default number of sections of ISO 21920), i.e. the profile holds six. Rz is the mean of the peak-to-valley heights of these
sections; the other parameters use the whole evaluation length. """

import dataclasses
import math
from typing import Optional

import numpy as np

from . import ops

ALPHA = math.sqrt(math.log(2.0) / math.pi)
CUTOFF_SERIES = tuple(m * 10.0 ** e for e in range(-8, -2) for m in (0.8, 2.5)) + (8e-3,)   # ... 0.8 µm ... 8 mm
SHORT_RATIO = 300.0                                  # λc / λs (ISO 21920-3 default)
FORMS = (("line", "Line (tilt)"), ("poly2", "2nd order (tilt and bow)"), ("none", "None"))
SECTIONS = 5                                         # sections of the evaluation length (ISO 21920 default)

HELP = {
    "a": "arithmetic mean of the absolute heights",
    "q": "root mean square of the heights",
    "p": "highest peak above the mean line",
    "v": "deepest valley below the mean line (positive)",
    "z": "mean of the largest peak-to-valley height of the sections (five sections of one cutoff each)",
    "t": "largest peak-to-valley height of the whole evaluation length",
    "sk": "skewness: < 0 plateau with valleys (e.g. worn, honed), > 0 peaks on a flat base",
    "ku": "kurtosis: 3 for a Gaussian surface, > 3 sharp peaks or valleys, < 3 rounded",
    "dq": "root mean square slope of the profile",
    "Sm": "mean width of the profile elements (a peak and the following valley; height discrimination 10 % of "
          "the peak-to-valley height, width discrimination 1 % of the cutoff)",
    "mr": "material ratio: the part of the profile at or above the level half the total height below the highest "
          "peak",
    "dc": "height difference between the levels where the material ratio is 20 % and 80 %",
}


@dataclasses.dataclass
class Settings:
    cutoff: float = 0.0              # λc in m, 0 = automatic
    short: str = "auto"              # λs: 'auto' (λc / 300, if larger than the point spacing) or 'none'
    form: str = "line"


@dataclasses.dataclass
class Result:
    x: np.ndarray                    # distance in m (the whole line)
    form: np.ndarray                 # the form that was removed (in the unit of the profile)
    primary: np.ndarray              # P (form removed, λs applied)
    waviness: np.ndarray             # W
    roughness: np.ndarray            # R
    cutoff: float                    # λc used, m
    short_cutoff: float              # λs used, m (0: none)
    automatic: bool                  # λc chosen automatically
    unit: str                        # unit of the heights
    evaluation: tuple                # (start, end) in m of the evaluation length
    params: dict                     # name -> (value, kind): kind 'height' (profile unit), 'length' (m), 'slope',
                                     # '%', '' (no unit)
    notes: list


def auto_cutoff(length: float) -> float:
    """ The largest cutoff of the series for which the profile holds six cutoffs: five for the evaluation and
        half a cutoff at each end (length / 6 if none fits). """
    fitting = [c for c in CUTOFF_SERIES if (SECTIONS + 1) * c <= length * (1 + 1e-9)]
    return fitting[-1] if fitting else length / (SECTIONS + 1)


def gaussian_filter(z: np.ndarray, dx: float, cutoff: float) -> np.ndarray:
    """ The Gaussian mean line (ISO 16610-21) of an evenly sampled profile; NaN points are left out, the weights
        are renormalized near the ends and gaps. """
    z = np.asarray(z, float)
    valid = np.isfinite(z)
    half = int(math.ceil(cutoff / dx))
    x = np.arange(-half, half + 1) * dx
    weights = np.exp(-math.pi * (x / (ALPHA * cutoff)) ** 2)
    numerator = np.convolve(np.where(valid, z, 0.0), weights, mode="full")[half:half + z.size]
    denominator = np.convolve(valid.astype(float), weights, mode="full")[half:half + z.size]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def _remove_form(x: np.ndarray, z: np.ndarray, form: str) -> np.ndarray:
    ok = np.isfinite(z)
    if form == "none" or ok.sum() < 3:
        return np.full_like(z, np.nanmean(z) if ok.any() else 0.0)
    order = 2 if form == "poly2" else 1
    centre = x[ok].mean()
    coefficients = np.polyfit(x[ok] - centre, z[ok], order)
    return np.polyval(coefficients, x - centre)


def profile_elements(r: np.ndarray, x: np.ndarray, height_limit: float, width_limit: float) -> list:
    """ Widths of the profile elements (a peak above and the following valley below the mean line, or the reverse)
        after ISO 21920: parts whose extreme is smaller than height_limit or that are narrower than width_limit
        are merged with their neighbors. Incomplete parts at the two ends are not counted. """
    ok = np.isfinite(r)
    r, x = r[ok], x[ok]
    if r.size < 3:
        return []
    sign = np.where(r >= 0, 1, -1)
    cuts = np.nonzero(np.diff(sign))[0]
    # crossing positions, linearly interpolated
    crossings = [x[i] + (x[i + 1] - x[i]) * r[i] / (r[i] - r[i + 1]) if r[i] != r[i + 1] else x[i] for i in cuts]
    bounds = [x[0]] + crossings + [x[-1]]
    index = [0] + [i + 1 for i in cuts] + [r.size]
    parts = []                                       # [start, end, sign, extreme |value|]
    for k in range(len(index) - 1):
        piece = r[index[k]:index[k + 1]]
        if piece.size:
            parts.append([bounds[k], bounds[k + 1], int(np.sign(piece[np.argmax(np.abs(piece))]) or 1),
                          float(np.max(np.abs(piece)))])
    changed = True
    while changed and len(parts) > 2:
        changed = False
        inner = range(1, len(parts) - 1)
        weak = [k for k in inner if parts[k][3] < height_limit or parts[k][1] - parts[k][0] < width_limit]
        if not weak:
            break
        k = min(weak, key=lambda i: parts[i][3])     # the weakest part goes first
        before, after = parts[k - 1], parts[k + 1]
        merged = [before[0], after[1], before[2], max(before[3], after[3])]
        parts[k - 1:k + 2] = [merged]
        changed = True
    inner = parts[1:-1]                              # complete parts only
    return [inner[i + 1][1] - inner[i][0] for i in range(0, len(inner) - 1, 2)]


def _family(prefix: str, r: np.ndarray, x: np.ndarray, dx: float, cutoff: float, sections: bool) -> dict:
    ok = np.isfinite(r)
    v = r[ok]
    if v.size < 3:
        return {}
    q = float(np.sqrt(np.mean(v ** 2)))
    out = {f"{prefix}a": (float(np.mean(np.abs(v))), "height"), f"{prefix}q": (q, "height"),
           f"{prefix}p": (float(v.max()), "height"), f"{prefix}v": (float(-v.min()), "height")}
    if sections:
        count = max(1, int(round((x[-1] - x[0]) / cutoff)))
        edges = np.linspace(x[0], x[-1], count + 1)
        heights = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            part = r[(x >= lo) & (x <= hi) & ok]
            if part.size:
                heights.append(float(part.max() - part.min()))
        out[f"{prefix}z"] = (float(np.mean(heights)) if heights else math.nan, "height")
    out[f"{prefix}t"] = (float(v.max() - v.min()), "height")
    out[f"{prefix}sk"] = (float(np.mean(v ** 3) / q ** 3) if q > 0 else math.nan, "")
    out[f"{prefix}ku"] = (float(np.mean(v ** 4) / q ** 4) if q > 0 else math.nan, "")
    return out


def analyze(distance: np.ndarray, heights: np.ndarray, unit: str, settings: Optional[Settings] = None) -> Result:
    """ P, W and R of a profile (distance in m, evenly spaced) with their parameters. Raises ops.OpError. """
    settings = settings or Settings()
    x = np.asarray(distance, float)
    z = np.asarray(heights, float)
    finite = np.nonzero(np.isfinite(z))[0]
    if finite.size:                                  # parts of the line outside the image do not count
        x, z = x[finite[0]:finite[-1] + 1], z[finite[0]:finite[-1] + 1]
    if x.size < 16 or np.isfinite(z).sum() < 16:
        raise ops.OpError("The profile is too short for a roughness analysis (at least 16 points).")
    dx = float(np.median(np.diff(x)))
    length = float(x[-1] - x[0])
    notes = []
    automatic = settings.cutoff <= 0
    cutoff = auto_cutoff(length) if automatic else float(settings.cutoff)
    if cutoff < 5 * dx:
        raise ops.OpError(f"The cutoff λc must be at least five point spacings ({5 * dx * 1e6:.3g} µm).")
    if length < 2 * cutoff:
        notes.append("The profile is shorter than two cutoffs: the waviness and Rz are not reliable.")
    if length >= 3 * cutoff:
        start, end = x[0] + cutoff / 2, x[-1] - cutoff / 2
    else:
        start, end = x[0], x[-1]
        notes.append("Shorter than three cutoffs: evaluated over the whole line, the ends included.")
    form = _remove_form(x, z, settings.form)
    primary = z - form
    short = 0.0
    if settings.short == "auto":
        short = cutoff / SHORT_RATIO
        if short < 2 * dx:
            notes.append(f"λs = λc / {SHORT_RATIO:g} is below two point spacings: not applied (the sampling "
                         "already limits the short wavelengths).")
            short = 0.0
        else:
            primary = gaussian_filter(primary, dx, short)
    waviness = gaussian_filter(primary, dx, cutoff)
    roughness = primary - waviness
    inside = (x >= start - 1e-12) & (x <= end + 1e-12)
    xe, re_, we, pe = x[inside], roughness[inside], waviness[inside], primary[inside]
    params = {}
    params.update(_family("R", re_, xe, dx, cutoff, True))
    slope = np.gradient(roughness, dx)[inside]
    params["Rdq"] = (float(np.sqrt(np.nanmean(slope ** 2))), "slope")
    widths = profile_elements(re_, xe, 0.1 * params.get("Rz", (0.0,))[0], 0.01 * cutoff)
    params["RSm"] = (float(np.mean(widths)) if widths else math.nan, "length")
    valid = re_[np.isfinite(re_)]
    top, total = float(valid.max()), float(valid.max() - valid.min())
    params["Rmr"] = (100.0 * float(np.mean(valid >= top - 0.5 * total)), "%")
    ordered = np.sort(valid)[::-1]
    level = lambda ratio: ordered[min(len(ordered) - 1, int(ratio * len(ordered)))]       # noqa: E731
    params["Rdc"] = (float(level(0.2) - level(0.8)), "height")
    wav = {k: v for k, v in _family("W", we - np.nanmean(we), xe, dx, cutoff, False).items()
           if k in ("Wa", "Wq", "Wp", "Wv", "Wt")}
    params.update(wav)
    w_widths = profile_elements(we - np.nanmean(we), xe, 0.1 * wav.get("Wt", (0.0,))[0], 0.01 * cutoff)
    params["WSm"] = (float(np.mean(w_widths)) if w_widths else math.nan, "length")
    prim = _family("P", pe - np.nanmean(pe), xe, dx, cutoff, False)
    params.update({k: v for k, v in prim.items() if k in ("Pa", "Pq", "Pt", "Psk", "Pku")})
    return Result(x, form, primary, waviness, roughness, cutoff, short, automatic, unit, (float(start), float(end)),
                  params, notes)


def help_text(name: str) -> str:
    family = {"R": "Roughness", "W": "Waviness", "P": "Primary profile"}.get(name[:1], "")
    return f"{family}: {HELP.get(name[1:], '')}".strip(": ")
