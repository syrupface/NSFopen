# -*- coding: utf-8 -*-
# Ported from the Nanosurf Surface Analyzer (NanosurfImg), where it is covered by tests.
""" Arithmetic of two channels, e.g. lateral force images from the forward and backward scans: the friction part
changes sign with the scan direction and the topographic crosstalk does not, so (F - B) / 2 is the lateral force
(friction) and (F + B) / 2 the crosstalk and offset. A - B, A + B or their halves,
with channel B shifted by a (sub-pixel) offset that compensates the displacement of the tip between the forward and
the backward scan. The offset can be measured by cross-correlating two channels that show the same features (usually
the topography forward and backward). No GUI code. """

import math
import re

import numpy as np
from scipy import fft as sfft
from scipy import ndimage

from . import ops

MIN_SIGNIFICANCE = 6.0
LATERAL = re.compile(r"lateral|friction|torsion", re.IGNORECASE)    # lateral force channels ('Friction force' in .nid)

OPERATIONS = {"diff": "A - B", "sum": "A + B", "half_diff": "(A - B) / 2", "mean": "(A + B) / 2", "a": "A only",
              "b": "B (shifted) only"}


def shifted(data: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
    """ data moved by (shift_x, shift_y) pixels (x to the right, y down the rows), linear interpolation; pixels that
        come from outside the image are NaN. """
    if shift_x == 0 and shift_y == 0:
        return data.copy()
    finite = np.isfinite(data)
    filled = np.where(finite, data, np.nanmean(data) if finite.any() else 0.0)
    moved = ndimage.shift(filled, (shift_y, shift_x), order=1, mode="constant", cval=np.nan, prefilter=False)
    valid = ndimage.shift(finite.astype(float), (shift_y, shift_x), order=1, mode="constant", cval=0.0,
                          prefilter=False)
    return np.where(valid > 0.999, moved, np.nan)


def combine(a: ops.State, b: ops.State, operation: str, shift_x: float = 0.0, shift_y: float = 0.0) -> ops.State:
    """ The combination of two channels of the same size; channel B is shifted by (shift_x, shift_y) pixels first.
        The result has the geometry and unit of A, and no mask. """
    if operation not in OPERATIONS:
        raise ops.OpError(f"Unknown operation '{operation}'.")
    if a.data.shape != b.data.shape:
        raise ops.OpError("The two channels have different sizes (was one of them cropped or resampled?).")
    if operation in ("diff", "sum", "half_diff", "mean") and a.unit != b.unit:
        raise ops.OpError(f"The channels have different units ({a.unit or '-'} and {b.unit or '-'}).")
    limit = max(a.data.shape) / 2
    if abs(shift_x) > limit or abs(shift_y) > limit:
        raise ops.OpError("The shift is larger than half the image.")
    bb = shifted(b.data, shift_x, shift_y)
    data = {"diff": lambda: a.data - bb, "sum": lambda: a.data + bb, "half_diff": lambda: 0.5 * (a.data - bb),
            "mean": lambda: 0.5 * (a.data + bb), "a": lambda: a.data.copy(), "b": lambda: bb}[operation]()
    return ops.State(data, np.zeros(data.shape, bool), a.size_x, a.size_y, a.unit if operation != "b" else b.unit)


def _prepared(data: np.ndarray) -> np.ndarray:
    """ Every line with its mean and slope removed (line offsets and tilt must not dominate the correlation), NaN
        filled, and a Hann window. """
    ny, nx = data.shape
    x = np.arange(nx) - (nx - 1) / 2
    out = np.where(np.isfinite(data), data, np.nan)
    with np.errstate(all="ignore"):
        means = np.nanmean(out, axis=1)
        out = out - means[:, None]
        slopes = np.nansum(out * x, axis=1) / np.sum(x * x)
    out = out - np.nan_to_num(slopes)[:, None] * x
    out = np.nan_to_num(out)
    return out * np.outer(np.hanning(ny), np.hanning(nx))


def find_shift(a: np.ndarray, b: np.ndarray, max_shift: float = 0.25) -> tuple:
    """ (shift_x, shift_y, correlation) that moves b onto a: the peak of the normalized cross-correlation, refined to
        a fraction of a pixel by a parabola through its neighbors; shifts up to max_shift of the image size. The peak
        must stand MIN_SIGNIFICANCE robust standard deviations above the rest of the correlation map (chance peaks of
        unrelated images reach about 4), else OpError. """
    if a.shape != b.shape:
        raise ops.OpError("The two channels have different sizes.")
    fa, fb = _prepared(a), _prepared(b)
    if not fa.any() or not fb.any():
        raise ops.OpError("A channel is empty or constant.")
    ny, nx = a.shape
    size = (sfft.next_fast_len(2 * ny), sfft.next_fast_len(2 * nx))
    corr = sfft.irfft2(sfft.rfft2(fa, size) * np.conj(sfft.rfft2(fb, size)), size)
    corr = corr / math.sqrt((fa ** 2).sum() * (fb ** 2).sum())
    corr = np.fft.fftshift(corr)
    cy, cx = size[0] // 2, size[1] // 2
    ry, rx = max(1, int(ny * max_shift)), max(1, int(nx * max_shift))
    window = corr[cy - ry:cy + ry + 1, cx - rx:cx + rx + 1]
    iy, ix = np.unravel_index(int(np.argmax(window)), window.shape)

    def refine(values, i):
        if 0 < i < len(values) - 1:
            left, mid, right = values[i - 1], values[i], values[i + 1]
            denominator = left - 2 * mid + right
            if denominator < 0:
                return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
        return 0.0

    score = float(window[iy, ix])
    rest = window.copy()                                  # the map away from the peak: how high do chance peaks go?
    rest[max(iy - 8, 0):iy + 9, max(ix - 8, 0):ix + 9] = np.nan
    center = np.nanmedian(rest)
    spread = 1.4826 * np.nanmedian(np.abs(rest - center))
    significance = (score - center) / spread if spread > 0 else math.inf
    if significance < MIN_SIGNIFICANCE or iy in (0, window.shape[0] - 1) or ix in (0, window.shape[1] - 1):
        raise ops.OpError(f"No reliable shift found (the best match stands only {significance:.1f} σ above chance, "
                          f"correlation {score:.2f}). Use a channel pair with clear common features (topography "
                          "forward / backward), or set the shift by hand.")
    dy = iy - ry + refine(window[:, ix], iy)
    dx = ix - rx + refine(window[iy, :], ix)
    return float(dx), float(dy), score
