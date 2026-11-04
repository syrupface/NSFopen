# -*- coding: utf-8 -*-
# Ported from the Nanosurf Surface Analyzer (NanosurfImg), where it is covered by tests.
""" Image processing operations. Every operation is a function of a State (data, mask, pixel geometry) and plain
parameters, registered in OPS by name, so a sequence of operations can be undone, saved as a recipe and replayed on other
images (pipeline.py). Pixel coordinates are (x = column, y = row) with row 0 at the top of the image.

Two kinds of mask parameters exist:
  fit_on   (fits, statistics of a level): 'all' pixels, 'exclude' the masked pixels, or 'only' the masked pixels
  apply_to (filters, glitch removal): the result replaces 'all' pixels, the 'masked' ones, or the 'unmasked' ones """

import dataclasses
import math
from typing import Callable, Optional

import numpy as np
from matplotlib.path import Path as _Path
from scipy import ndimage, sparse
from scipy import fft as sfft
from scipy.sparse.linalg import spsolve

FIT_ON = ("all", "exclude", "only")
APPLY_TO = ("all", "masked", "unmasked")
COMBINE = ("new", "add", "subtract", "xor", "intersect")


class OpError(Exception):
    """ An operation cannot be applied; the message is meant for the user. """


@dataclasses.dataclass
class State:
    data: np.ndarray                # 2D float
    mask: np.ndarray                # 2D bool, same shape
    size_x: float                   # physical width in m
    size_y: float                   # physical height in m
    unit: str

    @property
    def dx(self) -> float:
        return self.size_x / self.data.shape[1]

    @property
    def dy(self) -> float:
        return self.size_y / self.data.shape[0]

    def copy(self) -> "State":
        return State(self.data.copy(), self.mask.copy(), self.size_x, self.size_y, self.unit)


@dataclasses.dataclass(frozen=True)
class OpSpec:
    name: str
    label: str
    func: Callable
    defaults: dict


OPS: dict = {}


def _op(name: str, label: str, **defaults):
    def register(func):
        OPS[name] = OpSpec(name, label, func, defaults)
        return func
    return register


def apply_op(state: State, name: str, params: dict, resolve: Optional[Callable] = None) -> State:
    """ A new State with the operation applied. resolve(channel key) returns the current data of another channel of
        the same file (for operations that use a second channel); the input state is not changed. """
    spec = OPS.get(name)
    if spec is None:
        raise OpError(f"Unknown operation '{name}'.")
    unknown = set(params) - set(spec.defaults)
    if unknown:
        raise OpError(f"Operation '{name}' has no parameter {', '.join(sorted(unknown))}.")
    kwargs = {**spec.defaults, **params}
    new = state.copy()
    result = spec.func(new, resolve, **kwargs)
    result = new if result is None else result
    if result.data.shape != result.mask.shape:
        raise OpError("Internal error: data and mask differ in size.")
    return result


# ---- helpers ----

def _check(value: str, allowed: tuple, what: str):
    if value not in allowed:
        raise OpError(f"{what} must be one of {', '.join(allowed)} (not '{value}').")


def _fit_pixels(state: State, fit_on: str) -> np.ndarray:
    _check(fit_on, FIT_ON, "fit_on")
    use = np.isfinite(state.data)
    if fit_on == "exclude":
        use &= ~state.mask
    elif fit_on == "only":
        use &= state.mask
    return use


def _region(state: State, apply_to: str) -> np.ndarray:
    _check(apply_to, APPLY_TO, "apply_to")
    return {"all": np.ones_like(state.mask), "masked": state.mask, "unmasked": ~state.mask}[apply_to]


def _blend(state: State, new: np.ndarray, apply_to: str):
    state.data = np.where(_region(state, apply_to), new, state.data)


def _combine(old: np.ndarray, new: np.ndarray, mode: str) -> np.ndarray:
    _check(mode, COMBINE, "combine")
    return {"new": lambda: new, "add": lambda: old | new, "subtract": lambda: old & ~new,
            "xor": lambda: old ^ new, "intersect": lambda: old & new}[mode]()


def _filled(data: np.ndarray) -> np.ndarray:
    """ Copy with the NaN pixels replaced by the median, for filters that cannot handle NaN. """
    finite = np.isfinite(data)
    if finite.all():
        return data
    return np.where(finite, data, np.median(data[finite]) if finite.any() else 0.0)


def _terms(x: np.ndarray, y: np.ndarray, order: int, order_y: int = -1) -> np.ndarray:
    """ Columns x^i y^j with i + j <= order; with an own order in y (order_y >= 0): i <= order, j <= order_y and
        i + j <= the larger of the two (for equal orders the same as one total degree). """
    if order_y < 0 or order_y == order:
        return np.stack([x ** i * y ** (d - i) for d in range(order + 1) for i in range(d + 1)], axis=1)
    top = max(order, order_y)
    return np.stack([x ** i * y ** j for d in range(top + 1) for i in range(min(d, order) + 1)
                     for j in [d - i] if j <= order_y], axis=1)


def _solve(design: np.ndarray, values: np.ndarray) -> np.ndarray:
    normal = design.T @ design
    try:
        return np.linalg.solve(normal, design.T @ values)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(design, values, rcond=None)[0]


def _normalized_axes(ny: int, nx: int) -> tuple:
    x = (np.arange(nx) - (nx - 1) / 2) / (nx / 2)
    y = (np.arange(ny) - (ny - 1) / 2) / (ny / 2)
    return np.meshgrid(x, y)


# ---- leveling ----

@_op("level.poly2d", "Polynomial level", order=1, fit_on="all", clip=0.0, order_y=-1)
def level_poly2d(state, resolve, order, fit_on, clip, order_y=-1):
    """ Subtracts a 2D polynomial of the given total degree (1 = plane); order_y >= 0 gives the y direction its own
        order (e.g. 1 in x and 2 in y: a bow along the slow scan axis), order is then the order in x. clip > 0
        repeats the fit without the pixels further than clip standard deviations from it (particles, holes). """
    if not 0 <= int(order) <= 8 or not -1 <= int(order_y) <= 8:
        raise OpError("The polynomial order must be between 0 and 8.")
    use0 = _fit_pixels(state, fit_on)
    x, y = _normalized_axes(*state.data.shape)
    design = _terms(x.ravel(), y.ravel(), int(order), int(order_y))
    z = np.nan_to_num(state.data.ravel())
    use = use0.ravel()
    n_terms = design.shape[1]
    for _ in range(6 if clip > 0 else 1):
        if use.sum() < n_terms:
            raise OpError("Too few pixels for the fit: check the mask and the fit selection.")
        coeff = _solve(design[use], z[use])
        if clip <= 0:
            break
        residual = z - design @ coeff
        sigma = residual[use].std()
        new_use = use0.ravel() & (np.abs(residual) <= clip * sigma)
        if (new_use == use).all():
            break
        use = new_use
    state.data = state.data - (design @ coeff).reshape(state.data.shape)


@_op("level.plane", "Plane level", fit_on="all", clip=0.0)
def level_plane(state, resolve, fit_on, clip):
    return level_poly2d(state, resolve, 1, fit_on, clip)


@_op("level.lines", "Line level", order=1, direction="rows", method="poly", fit_on="all", clip=0.0)
def level_lines(state, resolve, order, direction, method, fit_on, clip):
    """ Subtracts a polynomial (method 'poly') or the median (method 'median') from every row or column. """
    _check(direction, ("rows", "columns"), "direction")
    _check(method, ("poly", "median"), "method")
    if not 0 <= int(order) <= 8:
        raise OpError("The polynomial order must be between 0 and 8.")
    data, use = state.data, _fit_pixels(state, fit_on)
    if direction == "columns":
        data, use = data.T, use.T
    out = data.copy()
    axis = np.linspace(-1.0, 1.0, data.shape[1])
    if method == "poly" and clip <= 0:                   # one solve for all rows that use the same pixels
        whole = use.all(axis=1)
        if whole.any() and data.shape[1] > int(order) + 1:
            design = np.polynomial.polynomial.polyvander(axis, int(order))
            coeff = np.linalg.lstsq(design, data[whole].T, rcond=None)[0]
            out[whole] = data[whole] - (design @ coeff).T
    else:
        whole = np.zeros(data.shape[0], bool)
    for i in np.flatnonzero(~whole):
        sel = use[i]
        if method == "median":
            if sel.any():
                out[i] = data[i] - np.median(data[i][sel])
            continue
        if sel.sum() <= order + 1:
            continue
        keep = sel.copy()
        for _ in range(6 if clip > 0 else 1):
            coeff = np.polynomial.polynomial.polyfit(axis[keep], data[i][keep], int(order))
            if clip <= 0:
                break
            residual = data[i] - np.polynomial.polynomial.polyval(axis, coeff)
            new_keep = sel & (np.abs(residual) <= clip * residual[keep].std())
            if new_keep.sum() <= order + 1 or (new_keep == keep).all():
                break
            keep = new_keep
        out[i] = data[i] - np.polynomial.polynomial.polyval(axis, coeff)
    state.data = out.T if direction == "columns" else out


@_op("level.align_rows", "Align rows", method="mod", direction="rows", fit_on="all")
def level_align_rows(state, resolve, method, direction, fit_on):
    """ Removes offsets between neighboring scan lines. 'median'/'mean': every line gets the median/mean of the image;
        'mod': the offset of a line is the median of its difference to the previous line (robust against features). """
    _check(method, ("median", "mean", "mod"), "method")
    _check(direction, ("rows", "columns"), "direction")
    data = state.data.T if direction == "columns" else state.data
    use = _fit_pixels(state, fit_on)
    use = use.T if direction == "columns" else use
    work = np.where(use, data, np.nan)
    with np.errstate(all="ignore"):
        if method == "mod":
            diff = np.nanmedian(work[1:] - work[:-1], axis=1)
            offsets = np.concatenate([[0.0], np.cumsum(np.nan_to_num(diff))])
        elif method == "median":
            offsets = np.nan_to_num(np.nanmedian(work, axis=1))
        else:
            offsets = np.nan_to_num(np.nanmean(work, axis=1))
    offsets -= np.mean(offsets)
    out = data - offsets[:, None]
    state.data = out.T if direction == "columns" else out


@_op("level.flatten", "Flatten", fit_on="all")
def level_flatten(state, resolve, fit_on):
    """ The usual first step for topography: the scan lines are aligned (median of the differences to the neighbor
        line, robust against features), then a plane is subtracted. fit_on as for the levels (exclude masked particles
        to level the substrate only). """
    level_align_rows(state, resolve, "mod", "rows", fit_on)
    return level_poly2d(state, resolve, 1, fit_on, 0.0)


@_op("level.shift", "Shift values", mode="min", fit_on="all")
def level_shift(state, resolve, mode, fit_on):
    """ Shifts the data so that the minimum, mean or median of the selected pixels is zero. """
    _check(mode, ("min", "mean", "median"), "mode")
    values = state.data[_fit_pixels(state, fit_on)]
    if not values.size:
        raise OpError("No pixels selected.")
    state.data = state.data - {"min": np.min, "mean": np.mean, "median": np.median}[mode](values)


def _plane_from_slopes(state: State, gx: float, gy: float) -> np.ndarray:
    """ The plane with slopes gx = dz/dx, gy = dz/dy (physical units, y down the rows) through zero at the center. """
    ny, nx = state.data.shape
    x = (np.arange(nx) - (nx - 1) / 2) * state.dx
    y = (np.arange(ny) - (ny - 1) / 2) * state.dy
    return gx * x[None, :] + gy * y[:, None]


@_op("level.facet", "Facet level", fit_on="all")
def level_facet(state, resolve, fit_on):
    """ Subtracts the plane that makes the most common local slope horizontal (the flat facets or terraces), found by
        a mean shift on the local gradients. Better than a plane fit when large particles or steps are present. """
    use = _fit_pixels(state, fit_on)
    data = _filled(state.data)
    for _ in range(3):                       # repeated: the gradients are measured on the leveled image again
        smooth = ndimage.uniform_filter(data, 3, mode="nearest")
        gy, gx = np.gradient(smooth, state.dy, state.dx)
        ok = use & np.isfinite(gx) & np.isfinite(gy)
        ok[[0, -1], :] = False
        ok[:, [0, -1]] = False
        if ok.sum() < 16:
            raise OpError("Too few pixels for the facet level: check the mask and the fit selection.")
        g = np.c_[gx[ok], gy[ok]]
        center = np.median(g, axis=0)
        spread = 1.4826 * np.median(np.abs(g - center), axis=0)
        width = max(float(np.hypot(*spread)), 1e-30)
        for _ in range(50):                  # mean shift towards the densest slope
            weight = np.exp(-np.sum((g - center) ** 2, axis=1) / (2 * (0.5 * width) ** 2))
            new = (g * weight[:, None]).sum(axis=0) / max(weight.sum(), 1e-300)
            if np.hypot(*(new - center)) < 1e-3 * width:
                center = new
                break
            center = new
        plane = _plane_from_slopes(state, center[0], center[1])
        data = data - plane
        state.data = state.data - plane
        if np.hypot(*center) < 1e-3 * width:
            break


@_op("level.three_point", "Three-point level", points=(), size=3)
def level_three_point(state, resolve, points, size=3):
    """ Subtracts the plane through three pixels (x, y), e.g. three points on the substrate, each averaged over
        size x size pixels (odd; 1 = the single pixel); their height becomes zero. """
    if len(points) != 3:
        raise OpError("Click three points.")
    size = int(size)
    if size < 1 or size > 101:
        raise OpError("The point size must be between 1 and 101 pixels.")
    half = size // 2
    ny, nx = state.data.shape
    rows = []
    for x, y in points:
        x, y = int(round(x)), int(round(y))
        if not (0 <= x < nx and 0 <= y < ny):
            raise OpError("A point lies outside the image.")
        window = state.data[max(y - half, 0):y + half + 1, max(x - half, 0):x + half + 1]
        if not np.isfinite(window).any():
            raise OpError("A point lies on invalid data.")
        rows.append((x * state.dx, y * state.dy, float(np.nanmean(window))))
    a = np.array([[x, y, 1.0] for x, y, _ in rows])
    if abs(np.linalg.det(a)) < 1e-6 * (state.size_x * state.size_y):
        raise OpError("The three points lie on a line: choose points that span a triangle.")
    gx, gy, c = np.linalg.solve(a, [z for _, _, z in rows])
    xx = np.arange(nx)[None, :] * state.dx
    yy = np.arange(ny)[:, None] * state.dy
    state.data = state.data - (gx * xx + gy * yy + c)


def _shrink_factor(size_px: float, limit: float) -> int:  # noqa: E302
    return max(1, int(math.ceil(size_px / limit)))


def _coarse(data: np.ndarray, factor: int) -> np.ndarray:
    ny, nx = data.shape
    return resample_array(data, max(4, nx // factor), max(4, ny // factor), "linear") if factor > 1 else data


def _fine(background: np.ndarray, shape: tuple) -> np.ndarray:
    return resample_array(background, shape[1], shape[0], "linear") if background.shape != shape else background


@_op("level.median", "Median background", size=31)
def level_median(state, resolve, size):
    """ Subtracts a median filter with a large window (pixels) as background: features smaller than the window stay.
        Large windows are computed on a reduced image (at most 15 pixels per window) and interpolated back. """
    size = int(size)
    if size < 3:
        raise OpError("The window must be at least 3 pixels.")
    factor = _shrink_factor(size, 15)
    coarse = _coarse(_filled(state.data), factor)
    background = ndimage.median_filter(coarse, size=max(3, int(round(size / factor)) | 1), mode="reflect")
    state.data = state.data - _fine(background, state.data.shape)


def _parabola(radius: float, pitch: float, limit: int, span: float) -> np.ndarray:
    """ 1D structuring element -t^2 / (2 radius): the bottom of a ball of that radius (heights << radius in AFM data,
        so the paraboloid and the sphere touch the surface at the same places). It is cut where it has dropped by more
        than span (the height range of the data): further out it can never touch the surface. """
    reach = math.sqrt(2.0 * radius * max(span, 0.0)) if span > 0 else pitch
    r = max(1, min(int(math.ceil(reach / pitch)), int(radius / pitch) + 1, limit))
    t = np.arange(-r, r + 1) * pitch
    return -t ** 2 / (2.0 * radius)


def _rolled(data: np.ndarray, radius: float, dx: float, dy: float, axes: tuple) -> np.ndarray:
    """ Opening of data by a paraboloid (axes 'x' and 'y': a ball) or a parabolic cylinder (one axis: an arc along the
        rows or the columns). The paraboloid is separable, so erosion and dilation are done axis by axis, exactly and
        at full resolution. The image is continued past its border by an odd reflection (the slope goes on). """
    ny, nx = data.shape
    span = float(np.ptp(data))
    elements = {"x": _parabola(radius, dx, 2 * nx, span)[None, :], "y": _parabola(radius, dy, 2 * ny, span)[:, None]}
    py = elements["y"].shape[0] // 2 if "y" in axes else 0
    px = elements["x"].shape[1] // 2 if "x" in axes else 0
    out = np.pad(data, ((py, py), (px, px)), mode="reflect", reflect_type="odd")
    for axis in axes:
        out = ndimage.grey_erosion(out, structure=elements[axis], mode="nearest")
    for axis in axes:
        out = ndimage.grey_dilation(out, structure=elements[axis], mode="nearest")
    return out[py:py + ny, px:px + nx]


@_op("level.terraces", "Terrace level", smoothing=2.0, low=1.0, high=3.0, broaden=2, min_area=80, degree=3,
     edges="auto")
def level_terraces(state, resolve, smoothing, low, high, broaden, min_area, degree, edges):
    """ Subtracts the background (tilt, bow) of a stepped surface found by the terrace analysis (terraces.py): the
        terraces become flat and the steps stay, unlike a polynomial fit, which tilts a staircase. """
    from . import terraces
    if state.unit != "m":
        raise OpError("Terrace leveling needs a height channel (heights in m).")
    settings = terraces.TerraceSettings(smoothing=float(smoothing), low=float(low), high=float(high),
                                        broaden=int(broaden), min_area=int(min_area), degree=int(degree), edges=edges)
    result = terraces.terraces(state, settings)
    state.data = state.data - result.background
    return state


@_op("level.rolling_ball", "Rolling ball background", radius=5e-6, shape="sphere", direction="both", invert=False)
def level_rolling_ball(state, resolve, radius, shape, direction, invert):
    """ Rolls a ball (shape 'sphere') or a cylinder ('arc', along the rows, the columns, or both one after the other)
        of the given radius (m) under the surface and subtracts its envelope. Bumps whose top is more curved than the
        ball (radius of curvature about width^2 / (8 height)) stay, the broader background is removed. invert: rolled
        over the top, for holes and pits. Height data (m) only. """
    _check(shape, ("sphere", "arc"), "shape")
    _check(direction, ("rows", "columns", "both"), "direction")
    if state.unit != "m":
        raise OpError("The rolling ball needs height data in m.")
    if radius <= 0:
        raise OpError("The radius must be positive.")
    sign = -1.0 if invert else 1.0
    data = sign * _filled(state.data)
    if shape == "sphere":
        background = _rolled(data, radius, state.dx, state.dy, ("x", "y"))
    else:
        background = data
        for axis in (("x", "y") if direction == "both" else ("x",) if direction == "rows" else ("y",)):
            background = _rolled(background, radius, state.dx, state.dy, (axis,))
    state.data = state.data - sign * background


# ---- rescaling ----

@_op("rescale.scale", "Scale values", factor=1.0, offset=0.0)
def rescale_scale(state, resolve, factor, offset):
    """ data * factor + offset """
    state.data = state.data * factor + offset


def _reference_pixels(state: State, resolve, reference: str, fit_on: str) -> tuple:
    if not reference:
        raise OpError("Select a reference channel.")
    if resolve is None:
        raise OpError("Other channels are not available here.")
    ref = np.asarray(resolve(reference), dtype=float)
    if ref.shape != state.data.shape:
        raise OpError("The reference channel has a different size (was it cropped?).")
    use = _fit_pixels(state, fit_on) & np.isfinite(ref)
    if use.sum() < 8:
        raise OpError("Too few pixels for the fit.")
    if np.std(state.data[use]) == 0 or np.std(ref[use]) == 0:
        raise OpError("The channel or the reference is constant.")
    return ref, use


def reference_gain(state: State, reference: np.ndarray, fit_on: str = "all") -> dict:
    """ {'gain', 'offset', 'gain_std', 'r'}: reference ~ gain * data + offset by least squares ('gain') and by the
        ratio of the standard deviations ('gain_std'), and the correlation coefficient r. If r is well below 1 the
        channels are dominated by noise: the least squares gain is then too small and the ratio too large. """
    use = _fit_pixels(state, fit_on) & np.isfinite(reference)
    x, y = state.data[use], reference[use]
    if x.size < 8 or x.std() == 0 or y.std() == 0:
        raise OpError("Too few or constant pixels for the fit.")
    r = float(np.corrcoef(x, y)[0, 1])
    gain, gain_std = r * y.std() / x.std(), np.sign(r) * y.std() / x.std()
    return {"gain": float(gain), "offset": float(y.mean() - gain * x.mean()), "gain_std": float(gain_std), "r": r}


@_op("rescale.z_to_reference", "Rescale to reference", reference="", order=1, method="ols", fit_on="all")
def rescale_z_to_reference(state, resolve, reference, order, method, fit_on):
    """ Maps this channel onto a reference channel of the same file, e.g. the open-loop Z-Axis onto the closed-loop
        Z-Axis Sensor. Level both channels first (same steps) so that tilt and drift do not enter the fit.
        order 1 is gain and offset; method 'ols' is least squares, 'std' matches the standard deviations (see
        reference_gain for when to use which); higher orders are least squares polynomials. """
    _check(method, ("ols", "std"), "method")
    if not 1 <= int(order) <= 5:
        raise OpError("The order must be between 1 and 5.")
    ref, use = _reference_pixels(state, resolve, reference, fit_on)
    if int(order) == 1:
        fit = reference_gain(state, ref, fit_on)
        gain = fit["gain"] if method == "ols" else fit["gain_std"]
        offset = ref[use].mean() - gain * state.data[use].mean()
        state.data = gain * state.data + offset
        return
    scale, centre = np.std(state.data[use]), np.mean(state.data[use])
    coeff = np.polynomial.polynomial.polyfit((state.data[use] - centre) / scale, ref[use], int(order))
    state.data = np.polynomial.polynomial.polyval((state.data - centre) / scale, coeff)


@_op("rescale.xy", "Rescale lateral size", factor_x=1.0, factor_y=1.0)
def rescale_xy(state, resolve, factor_x, factor_y):
    """ Multiplies the physical image size (calibration of the lateral scale of an open-loop scan). """
    if factor_x <= 0 or factor_y <= 0:
        raise OpError("The factors must be positive.")
    state.size_x *= factor_x
    state.size_y *= factor_y


# ---- geometry ----

@_op("geometry.crop", "Crop", x0=0, y0=0, x1=0, y1=0)
def geometry_crop(state, resolve, x0, y0, x1, y1):
    """ Keeps the pixels x0 <= x < x1, y0 <= y < y1. """
    ny, nx = state.data.shape
    x0, x1 = max(0, int(round(min(x0, x1)))), min(nx, int(round(max(x0, x1))))
    y0, y1 = max(0, int(round(min(y0, y1)))), min(ny, int(round(max(y0, y1))))
    if x1 - x0 < 4 or y1 - y0 < 4:
        raise OpError("The cropped image would be smaller than 4 x 4 pixels.")
    dx, dy = state.dx, state.dy
    state.data, state.mask = state.data[y0:y1, x0:x1].copy(), state.mask[y0:y1, x0:x1].copy()
    state.size_x, state.size_y = (x1 - x0) * dx, (y1 - y0) * dy


@_op("geometry.flip", "Flip", axis="vertical")
def geometry_flip(state, resolve, axis):
    _check(axis, ("vertical", "horizontal"), "axis")
    index = (slice(None, None, -1), slice(None)) if axis == "vertical" else (slice(None), slice(None, None, -1))
    state.data, state.mask = state.data[index].copy(), state.mask[index].copy()


ROTATIONS = {"cw": -1, "ccw": 1, "180": 2}           # np.rot90 turns: the picture as shown (row 0 at the top)


@_op("geometry.rotate", "Rotate", turn="cw")
def geometry_rotate(state, resolve, turn):
    """ Turns the image by 90° clockwise ('cw'), counterclockwise ('ccw') or by 180° ('180'); width and height (and
        non-square pixel sizes) swap for the quarter turns. """
    _check(turn, tuple(ROTATIONS), "turn")
    k = ROTATIONS[turn]
    state.data, state.mask = np.rot90(state.data, k).copy(), np.rot90(state.mask, k).copy()
    if turn != "180":
        state.size_x, state.size_y = state.size_y, state.size_x


RESAMPLE_METHODS = ("linear", "cubic", "nearest")


def resample_array(data: np.ndarray, nx: int, ny: int, method: str = "linear") -> np.ndarray:
    """ data interpolated onto nx x ny pixels covering the same area (pixel centers at (i + 0.5) * old / new - 0.5).
        NaN pixels stay NaN where the nearest original pixel is NaN. """
    _check(method, RESAMPLE_METHODS, "method")
    old_ny, old_nx = data.shape
    ys = (np.arange(ny) + 0.5) * old_ny / ny - 0.5
    xs = (np.arange(nx) + 0.5) * old_nx / nx - 0.5
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    order = {"nearest": 0, "linear": 1, "cubic": 3}[method]
    finite = np.isfinite(data)
    if finite.all() or not finite.any():
        source = _filled(data)
    else:       # NaN pixels take the value of the nearest valid pixel, so that they do not bias their neighbors
        index = ndimage.distance_transform_edt(~finite, return_distances=False, return_indices=True)
        source = data[tuple(index)]
    out = ndimage.map_coordinates(source, [yy, xx], order=order, mode="nearest", prefilter=order > 1)
    if not finite.all():
        nearest = ndimage.map_coordinates(finite.astype(float), [yy, xx], order=0, mode="nearest") > 0.5
        out = np.where(nearest, out, np.nan)
    return out


@_op("geometry.resample", "Resample pixels", nx=256, ny=256, method="linear")
def geometry_resample(state, resolve, nx, ny, method):
    """ Changes the number of pixels (and so the pixel size) while the image keeps its physical size. The mask is
        resampled with the nearest pixel. """
    nx, ny = int(nx), int(ny)
    if not (4 <= nx <= 8192 and 4 <= ny <= 8192):
        raise OpError("The number of pixels must be between 4 and 8192 per side.")
    state.data = resample_array(state.data, nx, ny, method)
    state.mask = resample_array(state.mask.astype(float), nx, ny, "nearest") > 0.5


@_op("geometry.square_pixels", "Square pixels", method="linear")
def geometry_square_pixels(state, resolve, method):
    """ Resamples the axis with the coarser pixels to the pixel size of the finer one (same image size), so that the
        pixels are square. Nothing changes if they already are (within 0.1 %). """
    ny, nx = state.data.shape
    step = min(state.dx, state.dy)
    new_nx, new_ny = max(4, int(round(state.size_x / step))), max(4, int(round(state.size_y / step)))
    if abs(state.dx / state.dy - 1) < 1e-3 or (new_nx, new_ny) == (nx, ny):
        return
    if max(new_nx, new_ny) > 8192:
        raise OpError("Square pixels would need more than 8192 pixels per side.")
    state.data = resample_array(state.data, new_nx, new_ny, method)
    state.mask = resample_array(state.mask.astype(float), new_nx, new_ny, "nearest") > 0.5


# ---- filters ----

def _footprint(size: int, shape: str = "disc") -> np.ndarray:
    """ Structuring element of `size` pixels across: a disc (default) or a square. """
    size = int(size)
    if size < 1 or size > 201:
        raise OpError("The size must be between 1 and 201 pixels.")
    if shape == "square":
        return np.ones((size, size), bool)
    r = (size - 1) / 2
    yy, xx = np.mgrid[0:size, 0:size] - r
    return xx ** 2 + yy ** 2 <= (r + 0.5) ** 2 - 0.25 + 1e-9 if size > 2 else np.ones((size, size), bool)


def _keep_nan(state, new):
    return np.where(np.isfinite(state.data), new, np.nan)


FILTER_KINDS = {
    "mean": "Mean (average)", "median": "Median", "gaussian": "Gaussian", "conservative": "Conservative denoise",
    "kuwahara": "Kuwahara (edge preserving)", "minimum": "Minimum (erosion)", "maximum": "Maximum (dilation)",
    "opening": "Opening", "closing": "Closing", "sharpen": "Sharpen (unsharp mask)"}


@_op("filter.basic", "Filter", kind="mean", size=3, sigma=1.0, amount=1.0, shape="disc", apply_to="all")
def filter_basic(state, resolve, kind, size, sigma, amount, shape, apply_to):
    """ Local filters, all ignoring NaN pixels:
        mean: average over the footprint; median; gaussian: sigma in pixels; conservative: a pixel outside the range of
        its neighbours is clipped to that range (removes single spikes, keeps everything else); kuwahara: mean of the
        least varying of the four quadrants of a size x size window (smooths, keeps edges); minimum / maximum: grey
        erosion / dilation; opening: erosion then dilation (removes peaks narrower than the footprint); closing: the
        reverse (fills pits); sharpen: adds amount x (image - Gaussian(sigma)). size is the footprint diameter in
        pixels, shape 'disc' or 'square'. """
    _check(kind, FILTER_KINDS, "kind")
    _check(shape, ("disc", "square"), "shape")
    data = _filled(state.data)
    if kind == "gaussian":
        return filter_gaussian(state, resolve, sigma, apply_to)
    if kind == "sharpen":
        if sigma <= 0:
            raise OpError("Sigma must be positive.")
        blurred = ndimage.gaussian_filter(data, sigma, mode="reflect")
        _blend(state, _keep_nan(state, data + float(amount) * (data - blurred)), apply_to)
        return
    size = int(size)
    if size < 2 and kind != "conservative":
        raise OpError("The size must be at least 2 pixels.")
    footprint = _footprint(max(size, 3) if kind == "conservative" else size, shape)
    if kind == "mean":
        finite = np.isfinite(state.data)
        weights = footprint.astype(float)
        total = ndimage.convolve(np.where(finite, state.data, 0.0), weights, mode="reflect")
        count = ndimage.convolve(finite.astype(float), weights, mode="reflect")
        with np.errstate(all="ignore"):
            new = total / count
    elif kind == "median":
        new = ndimage.median_filter(data, footprint=footprint, mode="reflect")
    elif kind == "conservative":
        ring = footprint.copy()
        ring[ring.shape[0] // 2, ring.shape[1] // 2] = False
        low = ndimage.minimum_filter(data, footprint=ring, mode="reflect")
        high = ndimage.maximum_filter(data, footprint=ring, mode="reflect")
        new = np.clip(data, low, high)
    elif kind == "kuwahara":
        new = _kuwahara(data, size)
    elif kind == "minimum":
        new = ndimage.grey_erosion(data, footprint=footprint, mode="reflect")
    elif kind == "maximum":
        new = ndimage.grey_dilation(data, footprint=footprint, mode="reflect")
    elif kind == "opening":
        new = ndimage.grey_opening(data, footprint=footprint, mode="reflect")
    else:
        new = ndimage.grey_closing(data, footprint=footprint, mode="reflect")
    _blend(state, _keep_nan(state, new), apply_to)


def _kuwahara(data: np.ndarray, size: int) -> np.ndarray:
    """ Each pixel gets the mean of the quadrant (of a size x size window, the pixel in its corner) with the smallest
        variance. """
    q = max(2, (int(size) + 1) // 2)
    mean = ndimage.uniform_filter(data, q, mode="reflect")
    square = ndimage.uniform_filter(data * data, q, mode="reflect")
    var = np.maximum(square - mean * mean, 0.0)
    # uniform_filter with even/odd q is centered at a known offset; shift so that each quadrant has the pixel in a corner
    a, b = q // 2, (q - 1) // 2
    offsets = [(-b, -b), (-b, a), (a, -b), (a, a)]
    ny, nx = data.shape
    pad = q
    mp, vp = np.pad(mean, pad, mode="edge"), np.pad(var, pad, mode="edge")
    means = np.stack([mp[pad + dy:pad + dy + ny, pad + dx:pad + dx + nx] for dy, dx in offsets])
    variances = np.stack([vp[pad + dy:pad + dy + ny, pad + dx:pad + dx + nx] for dy, dx in offsets])
    best = np.argmin(variances, axis=0)
    return np.take_along_axis(means, best[None], 0)[0]


@_op("filter.median", "Median filter", size=3, apply_to="all")
def filter_median(state, resolve, size, apply_to):
    size = int(size)
    if size < 2:
        raise OpError("The size must be at least 2 pixels.")
    finite = np.isfinite(state.data)
    new = ndimage.median_filter(_filled(state.data), size=size, mode="reflect")
    _blend(state, np.where(finite, new, np.nan), apply_to)


@_op("filter.gaussian", "Gaussian filter", sigma=1.0, apply_to="all")
def filter_gaussian(state, resolve, sigma, apply_to):
    """ Gaussian smoothing with the standard deviation sigma in pixels (NaN pixels are ignored). """
    if sigma <= 0:
        raise OpError("Sigma must be positive.")
    finite = np.isfinite(state.data)
    weight = ndimage.gaussian_filter(finite.astype(float), sigma, mode="reflect")
    value = ndimage.gaussian_filter(np.where(finite, state.data, 0.0), sigma, mode="reflect")
    with np.errstate(all="ignore"):
        new = np.where(finite, value / weight, np.nan)
    _blend(state, new, apply_to)


def _frequency_grid(shape: tuple, dx: float, dy: float) -> tuple:
    """ (kx, ky) in 1/m of the half-plane layout of scipy.fft.rfft2. """
    return np.meshgrid(sfft.rfftfreq(shape[1], dx), sfft.fftfreq(shape[0], dy))


def _fft_filtered(state: State, transfer: Callable) -> np.ndarray:
    """ Applies transfer(kx, ky) to the spectrum of the mirror-padded image (reduces edge ringing) and crops. """
    data = _filled(state.data)
    pad = min(data.shape) // 4
    padded = np.pad(data, pad, mode="reflect")
    kx, ky = _frequency_grid(padded.shape, state.dx, state.dy)
    spectrum = sfft.rfft2(padded)
    out = sfft.irfft2(spectrum * transfer(kx, ky), s=padded.shape)[pad:pad + data.shape[0], pad:pad + data.shape[1]]
    return np.where(np.isfinite(state.data), out + (data.mean() - out.mean()), np.nan)    # the mean is kept


@_op("filter.fft", "FFT filter", kind="lowpass", cutoff=50e-9, cutoff2=500e-9, apply_to="all")
def filter_fft(state, resolve, kind, cutoff, cutoff2, apply_to):
    """ Butterworth filter in the spatial frequency domain with cutoffs given as wavelengths in m.
        lowpass: keeps wavelengths longer than cutoff; highpass: keeps wavelengths shorter than cutoff;
        bandpass / bandstop: keeps / removes wavelengths between cutoff (short) and cutoff2 (long). """
    _check(kind, ("lowpass", "highpass", "bandpass", "bandstop"), "kind")
    if cutoff <= 0 or (kind.startswith("band") and cutoff2 <= cutoff):
        raise OpError("The cutoff wavelengths must be positive and cutoff2 larger than cutoff.")

    def low(f, wavelength):
        return 1.0 / (1.0 + (f * wavelength) ** 8)           # passes f < 1 / wavelength

    def transfer(kx, ky):
        f = np.hypot(kx, ky)
        if kind == "lowpass":
            return low(f, cutoff)
        if kind == "highpass":
            return 1.0 - low(f, cutoff)
        band = low(f, cutoff) * (1.0 - low(f, cutoff2))
        return band if kind == "bandpass" else 1.0 - band

    _blend(state, _fft_filtered(state, transfer), apply_to)


@_op("filter.notch", "FFT notch", points=(), radius=2e5, apply_to="all")
def filter_notch(state, resolve, points, radius, apply_to):
    """ Removes spots of the spectrum: points are (kx, ky) in 1/m, radius is the Gaussian width in 1/m. """
    if not len(points):
        raise OpError("Select at least one spot of the spectrum.")
    if radius <= 0:
        raise OpError("The radius must be positive.")

    def transfer(kx, ky):
        h = np.ones_like(kx)
        for px, py in points:
            for sign in (1.0, -1.0):
                h *= 1.0 - np.exp(-((kx - sign * px) ** 2 + (ky - sign * py) ** 2) / (2.0 * radius ** 2))
        return h

    _blend(state, _fft_filtered(state, transfer), apply_to)


# ---- glitch correction ----

def _robust_sigma(values: np.ndarray) -> float:
    return 1.4826 * float(np.median(np.abs(values - np.median(values)))) if values.size else 0.0


@_op("glitch.spikes", "Remove spikes", size=5, threshold=5.0, passes=1, apply_to="all")
def glitch_spikes(state, resolve, size, threshold, passes, apply_to):
    """ Replaces pixels that differ from the local median by more than threshold robust standard deviations. """
    size, region = int(size), _region(state, apply_to)
    if size < 3 or threshold <= 0:
        raise OpError("The size must be at least 3 and the threshold positive.")
    data = state.data.copy()
    for _ in range(max(1, int(passes))):
        local = ndimage.median_filter(_filled(data), size=size, mode="reflect")
        residual = data - local
        with np.errstate(invalid="ignore"):
            spikes = (np.abs(residual) > threshold * _robust_sigma(residual[np.isfinite(residual)])) & region
        if not spikes.any():
            break
        data = np.where(spikes, local, data)
    state.data = data


def detect_bad_lines(data: np.ndarray, threshold: float, max_width: int = 3) -> list:
    """ Rows whose offset jumps against the rows around them, also in clusters of up to max_width adjacent rows.
        The offset profile of the rows is built from the median differences of neighboring rows (robust against
        features); its running median over 2 max_width + 3 rows is the expected level, and rows further from it than
        threshold robust standard deviations are bad. Bad rows do not mark their neighbors. """
    ny = data.shape[0]
    if ny < 3:
        return []
    max_width = max(1, int(max_width))
    with np.errstate(all="ignore"):
        steps = np.nan_to_num(np.nanmedian(data[1:] - data[:-1], axis=1))
    level = np.concatenate([[0.0], np.cumsum(steps)])
    window = min(2 * max_width + 3, ny if ny % 2 else ny - 1)
    expected = ndimage.median_filter(level, size=window, mode="nearest")
    residual = level - expected
    sigma = _robust_sigma(residual)
    if sigma == 0:
        sigma = _robust_sigma(steps) / math.sqrt(2.0)
    if sigma == 0:
        return []
    return [int(i) for i in np.flatnonzero(np.abs(residual) > threshold * sigma)]


def _replace_rows(data: np.ndarray, rows: list) -> np.ndarray:
    ny = data.shape[0]
    bad = np.zeros(ny, bool)
    bad[[r for r in rows if 0 <= r < ny]] = True
    if bad.all():
        raise OpError("Cannot replace all lines.")
    good = np.flatnonzero(~bad)
    out = data.copy()
    for i in np.flatnonzero(bad):
        below, above = good[good < i], good[good > i]
        if below.size and above.size:
            a, b = below[-1], above[0]
            w = (i - a) / (b - a)
            out[i] = (1 - w) * data[a] + w * data[b]
        else:
            out[i] = data[below[-1] if below.size else above[0]]
    return out


@_op("glitch.lines", "Replace lines", lines=(), direction="rows")
def glitch_lines(state, resolve, lines, direction):
    """ Replaces the listed rows (or columns) by interpolation between the nearest intact ones. """
    _check(direction, ("rows", "columns"), "direction")
    if not len(lines):
        raise OpError("No lines selected.")
    transposed = direction == "columns"
    data = state.data.T if transposed else state.data
    out = _replace_rows(data, [int(i) for i in lines])
    state.data = out.T if transposed else out


@_op("glitch.auto_lines", "Repair bad lines", threshold=6.0, direction="rows", max_width=3)
def glitch_auto_lines(state, resolve, threshold, direction, max_width):
    """ Finds rows (or columns) that jump against the rows around them, also several adjacent ones (see
        detect_bad_lines), and interpolates them. """
    _check(direction, ("rows", "columns"), "direction")
    transposed = direction == "columns"
    data = state.data.T if transposed else state.data
    out = _replace_rows(data, detect_bad_lines(data, threshold, max_width))
    state.data = out.T if transposed else out


@_op("glitch.fill_mask", "Fill masked pixels", clear_mask=False)
def glitch_fill_mask(state, resolve, clear_mask):
    """ Replaces the masked pixels by the solution of Laplace's equation with the surrounding pixels as boundary. """
    _laplace_fill(state, state.mask | ~np.isfinite(state.data))
    if clear_mask:
        state.mask[:] = False


def _laplace_fill(state: State, unknown: np.ndarray):
    """ The unknown pixels become the solution of Laplace's equation with the other pixels as boundary. """
    if not unknown.any():
        return
    if unknown.all():
        raise OpError("The whole image is masked.")
    ny, nx = state.data.shape
    index = -np.ones((ny, nx), int)
    index[unknown] = np.arange(unknown.sum())
    n = int(unknown.sum())
    rows, cols, values, rhs = [], [], [], np.zeros(n)
    degree = np.zeros(n)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ys, xs = np.nonzero(unknown)
        ny2, nx2 = ys + dy, xs + dx
        inside = (ny2 >= 0) & (ny2 < ny) & (nx2 >= 0) & (nx2 < nx)
        ys, xs, ny2, nx2 = ys[inside], xs[inside], ny2[inside], nx2[inside]
        me, other = index[ys, xs], index[ny2, nx2]
        np.add.at(degree, me, 1)
        known = other < 0
        np.add.at(rhs, me[known], state.data[ny2[known], nx2[known]])
        rows.append(me[~known]), cols.append(other[~known]), values.append(-np.ones((~known).sum()))
    rows.append(np.arange(n)), cols.append(np.arange(n)), values.append(degree)
    matrix = sparse.csc_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    solved = spsolve(matrix, rhs)
    state.data[unknown] = solved


SCAR_SIGNS = ("both", "positive", "negative")


def find_scars(data: np.ndarray, threshold: float = 3.0, min_length: int = 16, max_width: int = 2,
               sign: str = "both") -> np.ndarray:
    """ Mask of scars: short pieces of scan lines (up to max_width rows thick, at least min_length pixels long) that
        are higher ('positive') or lower ('negative') than both the row above and the row below by more than threshold
        times the robust standard deviation of the differences between neighboring rows. """
    _check(sign, SCAR_SIGNS, "sign")
    ny, nx = data.shape
    z = _filled(data)
    diff = z[1:] - z[:-1]
    sigma = _robust_sigma(diff[np.isfinite(diff)].ravel()) / math.sqrt(2.0)
    found = np.zeros((ny, nx), bool)
    if sigma == 0 or ny < 3:
        return found
    limit = threshold * sigma
    for width in range(1, max(1, int(max_width)) + 1):
        if ny < width + 2:
            break
        above, below = z[:ny - width - 1], z[width + 1:]
        band = np.stack([z[1 + k:ny - width + k] for k in range(width)])       # rows of the candidate band
        low = np.minimum(band - above, band - below).min(axis=0)
        high = np.maximum(band - above, band - below).max(axis=0)
        hit = np.zeros(above.shape, bool)
        if sign in ("both", "positive"):
            hit |= low > limit
        if sign in ("both", "negative"):
            hit |= high < -limit
        for k in range(width):
            found[1 + k:ny - width + k] |= hit
    labels, count = ndimage.label(found, structure=[[0, 0, 0], [1, 1, 1], [0, 0, 0]])     # runs along the rows
    if count:
        lengths = ndimage.sum(found, labels, np.arange(1, count + 1))
        found = np.isin(labels, 1 + np.flatnonzero(lengths >= int(min_length)))
    return found


@_op("mask.scars", "Mask scars", threshold=3.0, min_length=16, max_width=2, sign="both", combine="add")
def mask_scars(state, resolve, threshold, min_length, max_width, sign, combine):
    """ Masks scars (see find_scars), to inspect them or to fill them later. """
    state.mask = _combine(state.mask, find_scars(state.data, threshold, min_length, max_width, sign), combine)


@_op("glitch.scars", "Remove scars", threshold=3.0, min_length=16, max_width=2, sign="both")
def glitch_scars(state, resolve, threshold, min_length, max_width, sign):
    """ Finds scars (see find_scars) and fills them from the surrounding pixels; the mask is not changed. """
    _laplace_fill(state, find_scars(state.data, threshold, min_length, max_width, sign) & np.isfinite(state.data))


@_op("glitch.step_lines", "Correct steps in lines", threshold=6.0, min_length=20, direction="rows")
def glitch_step_lines(state, resolve, threshold, min_length, direction):
    """ Height jumps in the middle of a scan line (e.g. a tip change): every row is compared with the mean of the row
        above and the row below; jumps of that difference larger than threshold robust standard deviations split the
        row into segments, and every segment is shifted to the level of the longest one. Segments shorter than
        min_length pixels are not split off. Run Align rows first. """
    _check(direction, ("rows", "columns"), "direction")
    transposed = direction == "columns"
    z = _filled(state.data.T if transposed else state.data)
    ny, nx = z.shape
    length = max(3, int(min_length))
    if ny < 3 or nx < 2 * length:
        raise OpError("The image is too small for this window.")
    reference = np.empty_like(z)
    reference[1:-1] = 0.5 * (z[:-2] + z[2:])
    reference[0], reference[-1] = z[1], z[-2]
    d = z - reference
    half = length // 2
    total = np.concatenate([np.zeros((ny, 1)), np.cumsum(d, axis=1)], axis=1)
    x = np.arange(half, nx - half + 1)                                          # step positions (between pixels)
    left = (total[:, x] - total[:, x - half]) / half
    right = (total[:, x + half] - total[:, x]) / half
    jump = right - left
    sigma = _robust_sigma(jump.ravel())
    out = z.copy()
    if sigma == 0:
        return
    for row in range(ny):
        strength = np.abs(jump[row])
        candidates = np.flatnonzero(strength > threshold * sigma)
        cuts = []
        for index in candidates[np.argsort(-strength[candidates])]:           # strongest first, at least
            position = int(x[index])                                            # min_length apart
            if all(abs(position - c) >= length for c in cuts) and length <= position <= nx - length:
                cuts.append(position)
        if not cuts:
            continue
        edges = [0] + sorted(cuts) + [nx]
        segments = [(a, b) for a, b in zip(edges[:-1], edges[1:])]
        main = max(segments, key=lambda ab: ab[1] - ab[0])
        level = np.median(d[row, main[0]:main[1]])
        for a, b in segments:
            if (a, b) != main:
                out[row, a:b] -= np.median(d[row, a:b]) - level
    out = np.where(np.isfinite(state.data.T if transposed else state.data), out, np.nan)
    state.data = out.T if transposed else out


# ---- masks ----

def _slope_degrees(state: State) -> np.ndarray:
    gy, gx = np.gradient(_filled(state.data), state.dy, state.dx)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


@_op("mask.clear", "Clear mask")
def mask_clear(state, resolve):
    state.mask[:] = False


@_op("mask.invert", "Invert mask")
def mask_invert(state, resolve):
    state.mask = ~state.mask


@_op("mask.threshold", "Mask by threshold", source="self", mode="above", low=0.0, high=0.0, unit="percentile",
     combine="new")
def mask_threshold(state, resolve, source, mode, low, high, unit, combine):
    """ Masks the pixels of a channel ('self', another channel of the file, or 'slope' in degrees) that are 'above'
        low, 'below' high, 'between' low and high or 'outside' them. Limits are absolute values in the unit of the
        channel (degrees for slope), percentiles (0-100) or multiples of the standard deviation around the mean. """
    _check(mode, ("above", "below", "between", "outside"), "mode")
    _check(unit, ("absolute", "percentile", "sigma"), "unit")
    values = _source_values(state, resolve, source)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise OpError("The channel has no valid pixels.")
    if unit == "percentile":
        low, high = np.percentile(finite, [low, high])
    elif unit == "sigma":
        low, high = finite.mean() + low * finite.std(), finite.mean() + high * finite.std()
    with np.errstate(invalid="ignore"):
        new = {"above": values >= low, "below": values <= high, "between": (values >= low) & (values <= high),
               "outside": (values < low) | (values > high)}[mode]
    state.mask = _combine(state.mask, new & np.isfinite(values), combine)


def _source_values(state: State, resolve, source: str) -> np.ndarray:
    """ The values a mask is computed from: 'self', 'slope' (degrees) or another channel of the file. """
    if source == "self":
        return state.data
    if source == "slope":
        return _slope_degrees(state)
    if resolve is None:
        raise OpError("Other channels are not available here.")
    values = np.asarray(resolve(source), dtype=float)
    if values.shape != state.data.shape:
        raise OpError("The source channel has a different size (was it cropped?).")
    return values


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """ Otsu's threshold: the value that splits the histogram into two classes with the largest variance between
        them (the smallest variance within them). """
    finite = values[np.isfinite(values)]
    if finite.size < 2 or finite.min() == finite.max():
        raise OpError("The values are constant: no threshold can be found.")
    low, high = np.percentile(finite, [0.05, 99.95])        # isolated outliers do not stretch the histogram
    counts, edges = np.histogram(np.clip(finite, low, high), bins=bins, range=(low, high))
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight = np.cumsum(counts)
    total = weight[-1]
    mean = np.cumsum(counts * centers)
    with np.errstate(all="ignore"):
        between = (mean[-1] * weight / total - mean) ** 2 / (weight * (total - weight))
    between[~np.isfinite(between)] = -1
    return float(edges[int(np.argmax(between)) + 1])


@_op("mask.otsu", "Mask by Otsu threshold", source="self", side="above", combine="new")
def mask_otsu(state, resolve, source, side, combine):
    """ Masks the pixels above (or below) Otsu's threshold of a channel ('self', 'slope' or another channel): the
        automatic split of a two-level image, e.g. particles on a flat substrate. Level the image first. """
    _check(side, ("above", "below"), "side")
    values = _source_values(state, resolve, source)
    threshold = otsu_threshold(values)
    with np.errstate(invalid="ignore"):
        new = (values > threshold) if side == "above" else (values <= threshold)
    state.mask = _combine(state.mask, new & np.isfinite(values), combine)


def watershed_segments(data: np.ndarray, smoothing: float = 1.0, separation: float = 3.0,
                       prominence: float = 0.0) -> tuple:
    """ Segments an image into hills: (labels 1..n with 0 on the 1-pixel boundaries between them, smoothed values).
        Every local maximum (at least `separation` pixels from a higher one) floods its basin downhill until the basins
        meet. Hills whose top rises less than `prominence` (in the unit of the data) above the pass to a neighbor
        are merged into that neighbor, so noise does not make grains. """
    finite = np.isfinite(data)
    if finite.sum() < 4:
        raise OpError("Too few valid pixels.")
    values = _filled(data)
    if smoothing > 0:
        values = ndimage.gaussian_filter(values, smoothing)
    low, high = float(values.min()), float(values.max())
    if high <= low:
        raise OpError("The image is flat: no hills to separate.")
    size = max(3, int(2 * round(separation) + 1))
    peaks = values == ndimage.maximum_filter(values, size=size)
    markers, count = ndimage.label(peaks, structure=np.ones((3, 3)))
    if count == 0:
        raise OpError("No maxima found.")
    # flooding from the maxima downhill: scipy's watershed works on increasing uint16 levels, so invert the heights
    order = np.argsort(-values, axis=None, kind="stable")        # by rank, not by value: no artificial plateaus
    ranks = np.empty(values.size, np.int64)
    ranks[order] = np.arange(values.size)
    levels = (ranks * (65535.0 / max(values.size - 1, 1))).astype(np.uint16).reshape(values.shape)
    labels = ndimage.watershed_ift(levels, markers.astype(np.int32), structure=np.ones((3, 3), int))
    if prominence > 0 and count > 1:
        labels = _merge_low_hills(values, labels, count, prominence)
    labels = _contiguous(labels, values)
    # one pixel of every pair of 8-neighbors in different basins becomes a line: then no two grains touch, also
    # not at a corner, and the grain analysis (8-connectivity) keeps them apart
    boundary = np.zeros(labels.shape, bool)
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:-1, :-1] |= labels[:-1, :-1] != labels[1:, 1:]
    boundary[:-1, 1:] |= labels[:-1, 1:] != labels[1:, :-1]
    labels = np.where(boundary | ~finite, 0, labels)
    return _contiguous(labels, values, drop=True), values


def _contiguous(labels: np.ndarray, values: np.ndarray, drop: bool = False) -> np.ndarray:
    """ Every basin as one connected piece: the piece that holds its top stays, stray pieces (ties in the flooding)
        go to the nearest other basin, or with drop become background (slivers cut off by the boundary lines). """
    labels = labels.copy()
    stray = np.zeros(labels.shape, bool)
    for number, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        inside = labels[box] == number
        parts, count = ndimage.label(inside, structure=np.ones((3, 3)))
        if count <= 1:
            continue
        top = np.unravel_index(np.argmax(np.where(inside, values[box], -np.inf)), inside.shape)
        stray[box] |= inside & (parts != parts[top])
    if stray.any() and drop:
        labels[stray] = 0
    elif stray.any():
        _, (iy, ix) = ndimage.distance_transform_edt(stray, return_indices=True)
        labels[stray] = labels[iy[stray], ix[stray]]
    return labels


def _merge_low_hills(values: np.ndarray, labels: np.ndarray, count: int, prominence: float) -> np.ndarray:
    """ Merges basins by persistence: neighbors are joined at their pass (the highest point of their common border),
        highest passes first; the lower of two hills dies there if its top is less than `prominence` above the pass. """
    peak = ndimage.maximum(values, labels, np.arange(count + 1))
    peak = np.asarray(peak, float)
    passes = {}
    for a_lab, b_lab, a_val, b_val in ((labels[:, :-1], labels[:, 1:], values[:, :-1], values[:, 1:]),
                                       (labels[:-1], labels[1:], values[:-1], values[1:])):
        differ = a_lab != b_lab
        la, lb = a_lab[differ], b_lab[differ]
        height = np.minimum(a_val[differ], b_val[differ])
        low_label, high_label = np.minimum(la, lb), np.maximum(la, lb)
        order = np.lexsort((height, high_label, low_label))
        for i in order:
            key = (int(low_label[i]), int(high_label[i]))
            if height[i] > passes.get(key, -np.inf):
                passes[key] = float(height[i])
    parent = np.arange(count + 1)
    top = peak.copy()                                    # the highest peak of each merged group

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (a, b), height in sorted(passes.items(), key=lambda item: -item[1]):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        lower, higher = (ra, rb) if top[ra] < top[rb] else (rb, ra)
        if top[lower] - height < prominence:             # not a hill of its own: part of the higher one
            parent[lower] = higher
    roots = np.array([find(i) for i in range(count + 1)])
    return roots[labels]


@_op("mask.watershed", "Mask by watershed", smoothing=1.0, separation=3.0, prominence=2.0, level=0.0,
     pits=False, combine="new")
def mask_watershed(state, resolve, smoothing, separation, prominence, level, pits, combine):
    """ Masks the hills (pits=True: the pits) of the image as separate grains: watershed segmentation from the
        maxima, 1-pixel unmasked lines between neighboring grains. prominence: smallest rise of a grain above the
        pass to its neighbor, in % of the height range (1-99 %); level: keep only the upper part of every grain, in %
        between its lowest and highest point (0: the whole basin). For touching grains (polycrystalline films,
        close-packed particles), where a threshold merges everything. Level the image first. """
    data = -state.data if pits else state.data
    finite = data[np.isfinite(data)]
    if finite.size < 4:
        raise OpError("Too few valid pixels.")
    span = float(np.percentile(finite, 99) - np.percentile(finite, 1)) or float(np.ptp(finite)) or 1.0
    if smoothing < 0 or separation < 1 or prominence < 0 or not 0 <= level < 100:
        raise OpError("Smoothing >= 0, distance >= 1 pixel, prominence >= 0 and the level 0 to 99 %.")
    labels, values = watershed_segments(data, smoothing, separation, prominence / 100.0 * span)
    new = labels > 0
    if level > 0 and labels.max() > 0:
        index = np.arange(int(labels.max()) + 1)
        low = np.asarray(ndimage.minimum(values, labels, index), float)
        high = np.asarray(ndimage.maximum(values, labels, index), float)
        cut = low + level / 100.0 * (high - low)
        new &= values >= cut[labels]
        parts, _ = ndimage.label(new, structure=np.ones((3, 3)))     # keep the part of each grain with its top
        tops = new & (values >= high[labels])
        new &= np.isin(parts, np.unique(parts[tops]))
    state.mask = _combine(state.mask, new, combine)


@_op("mask.outliers", "Mask outliers", sigma=3.0, window=0, side="both", combine="add")
def mask_outliers(state, resolve, sigma, window, side, combine):
    """ Masks pixels further than sigma robust standard deviations from the median of the image (window 0) or from
        the median of a window x window neighborhood (local outliers such as dust, spikes, holes). """
    _check(side, ("both", "above", "below"), "side")
    if sigma <= 0:
        raise OpError("Sigma must be positive.")
    data = state.data
    window = int(window)
    if window >= 3:
        reference = ndimage.median_filter(_filled(data), size=window, mode="reflect")
    elif window == 0:
        finite = data[np.isfinite(data)]
        reference = np.median(finite) if finite.size else 0.0
    else:
        raise OpError("The window must be 0 (whole image) or at least 3 pixels.")
    deviation = data - reference
    limit = sigma * _robust_sigma(deviation[np.isfinite(deviation)])
    if limit == 0:
        raise OpError("The image has no spread: nothing can be an outlier.")
    with np.errstate(invalid="ignore"):
        new = {"both": np.abs(deviation) > limit, "above": deviation > limit, "below": deviation < -limit}[side]
    state.mask = _combine(state.mask, new & np.isfinite(data), combine)


def rasterize(kind: str, points: list, shape: tuple) -> np.ndarray:
    """ Boolean image of a rectangle / ellipse (two opposite corners) or a polygon / freehand outline. """
    ny, nx = shape
    if kind in ("rect", "ellipse"):
        if len(points) < 2:
            raise OpError("A rectangle or ellipse needs two corner points.")
        (xa, ya), (xb, yb) = points[0], points[-1]
        x0, x1, y0, y1 = min(xa, xb), max(xa, xb), min(ya, yb), max(ya, yb)
        xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
        if kind == "rect":
            return (xx >= x0 - 0.5) & (xx <= x1 + 0.5) & (yy >= y0 - 0.5) & (yy <= y1 + 0.5)
        a, b = max((x1 - x0) / 2 + 0.5, 0.5), max((y1 - y0) / 2 + 0.5, 0.5)
        return ((xx - (x0 + x1) / 2) / a) ** 2 + ((yy - (y0 + y1) / 2) / b) ** 2 <= 1.0
    if kind in ("polygon", "freehand"):
        if len(points) < 3:
            raise OpError("A polygon needs at least three points.")
        pts = np.asarray(points, float)
        x0, x1 = max(int(np.floor(pts[:, 0].min())), 0), min(int(np.ceil(pts[:, 0].max())) + 1, nx)
        y0, y1 = max(int(np.floor(pts[:, 1].min())), 0), min(int(np.ceil(pts[:, 1].max())) + 1, ny)
        out = np.zeros(shape, bool)
        if x1 > x0 and y1 > y0:
            xx, yy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            inside = _Path(pts).contains_points(np.c_[xx.ravel(), yy.ravel()], radius=1e-9)
            out[y0:y1, x0:x1] = inside.reshape(xx.shape)
        return out
    raise OpError(f"Unknown shape '{kind}'.")


@_op("mask.shape", "Mask shape", kind="rect", points=(), combine="add")
def mask_shape(state, resolve, kind, points, combine):
    state.mask = _combine(state.mask, rasterize(kind, [tuple(p) for p in points], state.data.shape), combine)


@_op("mask.floodfill", "Mask region", seed=(0, 0), tolerance=0.0, combine="add")
def mask_floodfill(state, resolve, seed, tolerance, combine):
    """ The connected region around the seed pixel whose values differ from the seed value by at most tolerance. """
    ny, nx = state.data.shape
    x, y = int(round(seed[0])), int(round(seed[1]))
    if not (0 <= x < nx and 0 <= y < ny) or not np.isfinite(state.data[y, x]):
        raise OpError("The seed pixel is outside the image or invalid.")
    if tolerance < 0:
        raise OpError("The tolerance must not be negative.")
    with np.errstate(invalid="ignore"):
        similar = np.abs(state.data - state.data[y, x]) <= tolerance
    labels, _ = ndimage.label(similar)
    state.mask = _combine(state.mask, labels == labels[y, x], combine)


@_op("mask.grow", "Grow mask", pixels=1)
def mask_grow(state, resolve, pixels):
    if pixels < 0:
        raise OpError("The distance must not be negative.")
    if state.mask.any():
        state.mask = ndimage.distance_transform_edt(~state.mask) <= pixels


@_op("mask.shrink", "Shrink mask", pixels=1)
def mask_shrink(state, resolve, pixels):
    if pixels < 0:
        raise OpError("The distance must not be negative.")
    if state.mask.any():
        padded = np.pad(state.mask, 1, constant_values=False)         # the image border counts as unmasked
        state.mask = (ndimage.distance_transform_edt(padded) > pixels)[1:-1, 1:-1] & state.mask


@_op("mask.remove_small", "Remove small mask areas", min_pixels=10)
def mask_remove_small(state, resolve, min_pixels):
    labels, count = ndimage.label(state.mask)
    if count:
        sizes = ndimage.sum(state.mask, labels, np.arange(1, count + 1))
        state.mask = np.isin(labels, 1 + np.flatnonzero(sizes >= min_pixels))


@_op("mask.fill_holes", "Fill mask holes")
def mask_fill_holes(state, resolve):
    state.mask = ndimage.binary_fill_holes(state.mask)
