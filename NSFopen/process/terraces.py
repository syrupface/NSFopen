# -*- coding: utf-8 -*-
""" Terrace analysis of stepped surfaces (vicinal crystals, atomic terraces): the image is divided into terraces at the
steps, a smooth background (tilt, bow) is fitted together with one height per terrace, and the step height is found as
the common step of which the terrace heights are integer multiples.

    result = terraces(state, TerraceSettings())
    result.step, result.step_error        # m
    result.terraces                       # [Terrace(number, height, level, pixels, area, rms), ...]
    result.leveled(state)                 # the image without the background: flat terraces, steps kept

How it works:
  1. Background from slopes: a polynomial whose slopes match the local slopes of the image, robustly (the steps are a
     small part of the pixels and are left out), so a staircase does not tilt the fit.
  2. Steps: where the local slope differs from the background (smoothed, threshold in units of the noise of the slope,
     with hysteresis: weak parts of a step line count if they touch strong ones), broadened by a few pixels. The mask
     can be used as the steps instead or as well (draw or threshold the steps where they are too faint).
  3. Terraces: the connected areas between the steps, at least min_area pixels.
  4. Fit: z = background(x, y) + height of the terrace, all terraces at once (least squares; the terrace heights are
     eliminated exactly, so it is fast for any number of terraces).
  5. Step height: the largest step of which the terrace heights are nearly integer multiples (weighted by area), or a
     given one; then the background, the step and an offset are fitted again with the integer levels.

Only numpy and scipy are needed. """

import dataclasses
import math
from typing import Optional

import numpy as np
from scipy import ndimage

from . import ops

EDGES = ("auto", "mask", "both")


@dataclasses.dataclass
class TerraceSettings:
    smoothing: float = 2.0          # pixels: Gaussian smoothing before the slopes are measured
    low: float = 1.0                # step threshold (weak), in noise units of the slope deviation
    high: float = 3.0               # step threshold (strong): a step line needs at least one pixel above it
    broaden: int = 2                # pixels added on both sides of a step (left out of the fit)
    min_area: int = 80              # pixels: smaller areas between steps are not terraces
    degree: int = 3                 # polynomial background: 1 tilt, 2 bow, 3 or more for more complex shapes
    step: Optional[float] = None    # known step height (m); None: found from the terrace heights
    edges: str = "auto"             # 'auto' (from the slopes), 'mask' (the mask marks the steps), 'both'


@dataclasses.dataclass
class Terrace:
    number: int                     # 1, 2, ... from the lowest terrace up
    height: float                   # m, relative to the lowest terrace, background removed
    level: Optional[int]            # integer level (multiples of the step) or None without a common step
    pixels: int
    area: float                     # m²
    rms: float                      # m: roughness of the terrace around its fitted height


@dataclasses.dataclass
class TerraceResult:
    labels: np.ndarray              # terrace number of every pixel, 0 on the steps (left out of the fit)
    terraces: list
    step: float                     # m, nan if there is no common step height
    step_error: float               # m, standard error from the scatter of the terrace heights around their levels
    periodicity: float              # 0..1: how well the terrace heights fit integer multiples of the step (1: exactly)
    background: np.ndarray          # the fitted background (polynomial), same shape as the image
    rms: float                      # m: rms of the pixels around their terrace heights
    discrepancy: float              # m: rms of the terrace heights around their integer levels
    settings: TerraceSettings
    message: str = ""

    def leveled(self, state: ops.State) -> np.ndarray:
        """ The image without the background: flat terraces, the steps kept. """
        return state.data - self.background

    def summary(self) -> str:
        def nm(value):
            return f"{value * 1e9:.4g} nm" if abs(value) < 1e-6 else f"{value * 1e6:.4g} µm"
        lines = [f"{len(self.terraces)} terraces, fit rms {nm(self.rms)} (polynomial background of degree "
                 f"{self.settings.degree})"]
        if math.isfinite(self.step):
            levels = sorted({t.level for t in self.terraces})
            error = f" ± {nm(self.step_error)}" if math.isfinite(self.step_error) else ""
            given = " (given)" if self.settings.step else ""
            lines.append(f"Step height {nm(self.step)}{error}{given}; {len(levels)} levels from 0 to {max(levels)}, "
                         f"terrace heights within {nm(self.discrepancy)} rms of the levels "
                         f"(periodicity {self.periodicity:.2f})")
            if not self.settings.step and self.step < 1.5 * self.rms:
                lines.append(f"Check the terraces: the step is not much larger than their roughness ({nm(self.rms)}), "
                             "so scan line offsets or noise may have been taken for steps (align the rows first).")
        else:
            lines.append("No common step height. " + (self.message or "The terrace heights are not multiples of one step."))
        return "\n".join(lines)

    def rows(self) -> list:
        """ Table rows: number, height (m), level, pixels, area (m²), rms (m). """
        return [(t.number, t.height, t.level, t.pixels, t.area, t.rms) for t in self.terraces]


# ---- the steps ----

def _coordinates(shape: tuple) -> tuple:
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    return xx / max(nx - 1, 1) - 0.5, yy / max(ny - 1, 1) - 0.5


def _exponents(degree: int) -> list:
    return [(i, d - i) for d in range(1, degree + 1) for i in range(d + 1)]


def _robust_noise(values: np.ndarray) -> tuple:
    median = float(np.median(values))
    return median, 1.4826 * float(np.median(np.abs(values - median))) or float(np.std(values)) or 1.0


def slope_background(data: np.ndarray, degree: int, smoothing: float, rounds: int = 3) -> tuple:
    """ (background without its constant, slope deviation of every pixel): a polynomial whose slopes match the local
        slopes of the image, the pixels whose slope deviates most (the steps) left out in later rounds. """
    x, y = _coordinates(data.shape)
    ny, nx = data.shape
    smooth = ndimage.gaussian_filter(data, smoothing) if smoothing > 0 else data
    gy, gx = np.gradient(smooth)
    gx, gy = gx * max(nx - 1, 1), gy * max(ny - 1, 1)          # per unit of the normalized coordinates
    exponents = _exponents(degree)
    ax = np.stack([i * x ** max(i - 1, 0) * y ** j for i, j in exponents], -1)
    ay = np.stack([j * x ** i * y ** max(j - 1, 0) for i, j in exponents], -1)
    use = np.isfinite(gx) & np.isfinite(gy)
    coefficients = np.zeros(len(exponents))
    deviation = np.zeros(data.shape)
    for _ in range(rounds):
        a = np.concatenate([ax[use], ay[use]])
        b = np.concatenate([gx[use], gy[use]])
        coefficients, *_ = np.linalg.lstsq(a, b, rcond=None)
        deviation = np.hypot(gx - ax @ coefficients, gy - ay @ coefficients)
        median, noise = _robust_noise(deviation[np.isfinite(deviation)])
        use = np.isfinite(deviation) & (deviation < median + 3 * noise)
    background = np.stack([x ** i * y ** j for i, j in exponents], -1) @ coefficients
    return background, np.nan_to_num(deviation, nan=0.0)


def find_steps(deviation: np.ndarray, low: float, high: float) -> np.ndarray:
    """ Step pixels: slope deviation above low (in noise units) and connected to a pixel above high (hysteresis). """
    median, noise = _robust_noise(deviation)
    weak = deviation > median + low * noise
    strong = deviation > median + high * noise
    labels, count = ndimage.label(weak, structure=np.ones((3, 3)))
    if count == 0:
        return weak
    keep = np.zeros(count + 1, bool)
    keep[np.unique(labels[strong])] = True
    keep[0] = False
    return keep[labels]


def segment(steps: np.ndarray, broaden: int, min_area: int) -> tuple:
    """ (labels of every pixel, pixels used in the fit): the areas between the (broadened) steps that are at least
        min_area large; the step pixels get the label of the nearest terrace (for display), the fit leaves them out. """
    wide = ndimage.binary_dilation(steps, iterations=broaden) if broaden > 0 and steps.any() else steps
    labels, count = ndimage.label(~wide)
    if count:
        sizes = np.bincount(labels.ravel())
        small = sizes < min_area
        small[0] = False
        labels[small[labels]] = 0
        _, labels = np.unique(labels, return_inverse=True)
        labels = labels.reshape(steps.shape)
    used = labels > 0
    if used.any() and not used.all():
        _, (rows, cols) = ndimage.distance_transform_edt(~used, return_indices=True)
        labels = labels[rows, cols]
    return labels, used


# ---- the fits ----

def _fit_terraces(data: np.ndarray, labels: np.ndarray, used: np.ndarray, degree: int) -> tuple:
    """ (terrace ids, heights, pixels, background): z = polynomial (no constant) + one height per terrace, least squares
        with the heights eliminated (each column minus its terrace mean). """
    x, y = _coordinates(data.shape)
    exponents = _exponents(degree)
    m = used & np.isfinite(data)
    ids, index = np.unique(labels[m], return_inverse=True)
    counts = np.bincount(index)
    z = data[m]
    terms = np.stack([x[m] ** i * y[m] ** j for i, j in exponents], -1)

    def demeaned(values):
        return values - (np.bincount(index, values) / counts)[index]

    coefficients, *_ = np.linalg.lstsq(np.stack([demeaned(terms[:, k]) for k in range(terms.shape[1])], 1),
                                       demeaned(z), rcond=None)
    heights = np.bincount(index, z - terms @ coefficients) / counts
    background = np.stack([x ** i * y ** j for i, j in exponents], -1) @ coefficients
    return ids, heights, counts, background


def periodicity(heights: np.ndarray, weights: np.ndarray, step: float) -> tuple:
    """ (0..1, offset): how well the heights are integer multiples of the step plus an offset (1: exactly). """
    phase = np.sum(weights * np.exp(2j * np.pi * heights / step)) / np.sum(weights)
    return float(abs(phase)), float(np.angle(phase) / (2 * np.pi) * step)


def find_step(heights: np.ndarray, weights: np.ndarray, smallest: Optional[float] = None) -> tuple:
    """ (step, periodicity): the largest step of which the heights are nearly integer multiples. Every fraction of the
        true step fits as well, so the largest one that fits nearly as well as the best is taken. """
    span = float(np.ptp(heights)) if heights.size else 0.0
    if heights.size < 2 or span <= 0:
        return math.nan, 0.0
    smallest = smallest or span / 200
    candidates = np.geomspace(smallest, span * 1.05, 3000)
    scores = np.array([periodicity(heights, weights, s)[0] for s in candidates])
    best = scores.max()
    good = np.nonzero(scores >= 0.92 * best)[0]
    # the largest candidate in a local maximum near the best score
    index = int(good[-1])
    while index + 1 < len(scores) and scores[index + 1] > scores[index]:
        index += 1
    return float(candidates[index]), float(scores[index])


def terraces(state: ops.State, settings: Optional[TerraceSettings] = None) -> TerraceResult:
    """ Terrace analysis of a height image (see the module description). """
    settings = settings or TerraceSettings()
    if settings.edges not in EDGES:
        raise ops.OpError(f"edges must be one of {', '.join(EDGES)}.")
    if settings.degree < 0 or settings.degree > 8:
        raise ops.OpError("The degree of the background must be 0 to 8.")
    data = ops._filled(state.data.astype(float))
    ny, nx = data.shape
    if min(ny, nx) < 16:
        raise ops.OpError("The image is too small for a terrace analysis.")
    degree = max(int(settings.degree), 1)
    _, deviation = slope_background(data, degree, settings.smoothing)
    steps = np.zeros(data.shape, bool)
    if settings.edges in ("auto", "both"):
        steps |= find_steps(deviation, settings.low, settings.high)
    if settings.edges in ("mask", "both"):
        if not state.mask.any():
            raise ops.OpError("The mask is empty: draw or threshold the steps first, or let them be found automatically.")
        steps |= state.mask
    labels, used = segment(steps, settings.broaden, settings.min_area)
    used &= np.isfinite(state.data)
    if not used.any():
        raise ops.OpError("No terraces found: lower the step thresholds, the broadening or the minimum area.")
    ids, heights, counts, background = _fit_terraces(data, labels, used, settings.degree if settings.degree > 0 else 1)
    if settings.degree == 0:                                # only one height per terrace, no background
        background = np.zeros(data.shape)
        ids, index = np.unique(labels[used], return_inverse=True)
        counts = np.bincount(index)
        heights = np.bincount(index, data[used]) / counts

    step, score, levels, discrepancy, step_error, message = math.nan, 0.0, None, math.nan, math.nan, ""
    if settings.step:
        step = float(settings.step)
        score, _ = periodicity(heights, counts, step)
    elif len(ids) >= 2:
        step, score = find_step(heights, counts)
        if score < 0.6:
            message = (f"The best common step ({step * 1e9:.3g} nm) fits poorly (periodicity {score:.2f}); the terraces "
                       "may be merged at gaps in the steps: lower the thresholds or draw the steps into the mask.")
            step = math.nan
    elif len(ids) < 2:
        message = "Only one terrace was found."

    if math.isfinite(step):
        _, offset = periodicity(heights, counts, step)
        levels_of = np.rint((heights - offset) / step).astype(int)
        levels_of -= levels_of.min()
        # refit: background + step x level + offset, with the step free unless it was given
        x, y = _coordinates(data.shape)
        exponents = _exponents(max(settings.degree, 0)) if settings.degree > 0 else []
        level_map = np.zeros(labels.max() + 1, int)
        level_map[ids] = levels_of
        level_image = level_map[labels]
        m = used & np.isfinite(state.data)
        columns = [x[m] ** i * y[m] ** j for i, j in exponents] + [np.ones(m.sum())]
        z = data[m]
        if settings.step:
            z = z - step * level_image[m]
        else:
            columns.append(level_image[m].astype(float))
        design = np.stack(columns, 1)
        solution, *_ = np.linalg.lstsq(design, z, rcond=None)
        fit_error = math.nan
        if not settings.step:
            step = float(solution[-1])
            # standard error of the step from the fit: it includes how much the background can imitate the steps
            residual_variance = np.sum((z - design @ solution) ** 2) / max(len(z) - design.shape[1], 1)
            try:
                fit_error = float(np.sqrt(residual_variance * np.linalg.inv(design.T @ design)[-1, -1]))
            except np.linalg.LinAlgError:
                fit_error = math.nan
        background = (np.stack([x ** i * y ** j for i, j in exponents], -1) @ solution[:len(exponents)]
                      if exponents else np.zeros(data.shape))
        ids, heights, counts, _ = _refit_heights(data, labels, used, background)
        residuals = heights - (heights - step * levels_of).mean() - step * levels_of
        discrepancy = float(np.sqrt(np.average(residuals ** 2, weights=counts)))
        spread = levels_of - levels_of.mean()
        if len(ids) > 2 and np.sum(spread ** 2) > 0:
            step_error = float(np.sqrt(np.sum(residuals ** 2) / (len(ids) - 2) / np.sum(spread ** 2)))
        if math.isfinite(fit_error):
            step_error = fit_error if not math.isfinite(step_error) else max(step_error, fit_error)
        levels = levels_of
        score, _ = periodicity(heights, counts, step)
        if not settings.step and len(set(levels_of.tolist())) < 3:
            difference = float(np.ptp(heights))
            message = (f"Only two height levels ({difference * 1e9:.3g} nm apart): a common step height needs three or "
                       "more. The terraces may be merged at gaps in the steps: lower the thresholds or draw the steps "
                       "into the mask, or enter the step height.")
            step, step_error, levels, discrepancy = math.nan, math.nan, None, math.nan

    # order the terraces from the lowest up and renumber the labels the same way
    order = np.argsort(heights, kind="stable")
    base = heights[order[0]] if len(order) else 0.0
    renumber = np.zeros(labels.max() + 1, int)
    found = []
    residual = data - background
    for number, k in enumerate(order, start=1):
        renumber[ids[k]] = number
        pixels = used & (labels == ids[k])
        rms = float(np.std(residual[pixels])) if pixels.any() else math.nan
        found.append(Terrace(number, float(heights[k] - base), None if levels is None else int(levels[k]),
                             int(counts[k]), float(counts[k] * state.dx * state.dy), rms))
    new_labels = np.where(used, renumber[labels], 0)
    level_of_label = renumber[labels]
    fitted = np.array([0.0] + [t.height for t in found])[level_of_label] + base
    rms = float(np.sqrt(np.mean((residual[used] - fitted[used]) ** 2))) if used.any() else math.nan
    return TerraceResult(new_labels, found, float(step), step_error, float(score), background, rms,
                         discrepancy, settings, message)


def _refit_heights(data: np.ndarray, labels: np.ndarray, used: np.ndarray, background: np.ndarray) -> tuple:
    m = used & np.isfinite(data)
    ids, index = np.unique(labels[m], return_inverse=True)
    counts = np.bincount(index)
    heights = np.bincount(index, (data - background)[m]) / counts
    return ids, heights, counts, background
