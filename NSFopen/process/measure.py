# -*- coding: utf-8 -*-
# Ported from the Nanosurf Surface Analyzer (NanosurfImg), where it is covered by tests.
""" Measurements on an image: statistics, line profiles, power spectra, bearing curve. No GUI code. """

import dataclasses
import math
import warnings
from typing import Optional

import numpy as np
from scipy import fft as sfft
from scipy import ndimage

from . import ops

STATS_MODES = ("all", "masked", "unmasked")


def region_pixels(data: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    """ Valid pixels of the whole image ('all'), inside the mask ('masked') or outside of it ('unmasked'). """
    if mode not in STATS_MODES:
        raise ops.OpError(f"The region must be one of {', '.join(STATS_MODES)}.")
    use = np.isfinite(data)
    if mode == "masked":
        use &= mask
    elif mode == "unmasked":
        use &= ~mask
    return use


STAT_HELP = {
    "Pixels": "Number of pixels used (all, inside or outside the mask; invalid pixels are left out).",
    "Area": "Area covered by these pixels.",
    "Mean": "Average value. After leveling it is zero by construction.",
    "Median": "Middle value: half of the pixels lie above, half below. Less affected by particles and spikes than the "
              "mean.",
    "Min": "Lowest value.",
    "Max": "Highest value.",
    "Sa": "Arithmetic mean height: the average distance of the surface from its mean plane. The most quoted roughness "
          "number (areal version of Ra).",
    "Sq": "Root mean square height: like Sa but weights large deviations more. For a random surface Sq is about "
          "1.25 x Sa (areal version of Rq / RMS roughness).",
    "Sp": "Highest peak above the mean plane.",
    "Sv": "Deepest valley below the mean plane.",
    "Sz": "Maximum height: from the deepest valley to the highest peak (Sp + Sv). Sensitive to single spikes and "
          "particles.",
    "Ssk": "Skewness: 0 for a symmetric surface; negative for a plateau with pits or scratches; positive for a flat "
           "surface with peaks or particles.",
    "Sku": "Kurtosis: 3 for a random (Gaussian) surface; larger for sharp peaks or pits; smaller for a bumpy, "
           "rounded surface.",
    "Sdq": "Root mean square slope of the surface (0 for a flat surface; tan of the typical slope angle).",
    "Sdr": "Developed interfacial area ratio: how much larger the real surface is than the flat projected area, in %. "
           "Important for adhesion, coatings and catalysis.",
    "Sal": "Autocorrelation length: the shortest distance over which the surface pattern becomes unrelated "
           "(ACF falls to 0.2). Small: fine texture; large: long-range features.",
    "Str": "Texture aspect ratio, 0 to 1: near 1 the surface looks the same in all directions (isotropic); near 0 it "
           "has a strong direction (scratches, grooves, terraces).",
    "Std": "Texture direction: the dominant direction of the lay (grooves, scratches), in degrees from the x axis. "
           "Meaningful only when Str is small.",
    "Correlation length": "Distance along x over which the height correlation falls to 1/e: the typical lateral size of "
                          "the features.",
    "Step height (mask - rest)": "Height difference between the masked area and the rest, e.g. a terrace, a film edge "
                                 "or a lithography line (mask the upper level first).",
    "Step height error": "Uncertainty of the step height from the scatter of both levels.",
}


def statistics(state: ops.State, mode: str = "all", texture: bool = True) -> dict:
    """ {name: (value, unit)} of the pixels selected by mode. Roughness parameters follow ISO 25178 (S parameters) for
        the mean-referenced height: Sa mean absolute deviation, Sq RMS, Ssk skewness, Sku kurtosis (3 for a Gaussian
        distribution), Sp / Sv highest peak / deepest valley above / below the mean, Sz = Sp + Sv, Sdq RMS slope and
        Sdr developed area ratio (only for height data in m, from the pixel differences inside the region); with
        texture: Sal autocorrelation length, Str texture aspect ratio, Std texture direction (see texture_parameters)
        and the correlation length along x (ACF = 1/e, see line_functions). """
    use = region_pixels(state.data, state.mask, mode)
    z = state.data[use]
    unit = state.unit
    nan = float("nan")
    result = {"Pixels": (float(z.size), ""), "Area": (z.size * state.dx * state.dy, "m²")}
    if z.size == 0:
        for name in ("Mean", "Median", "Min", "Max", "Sa", "Sq", "Sp", "Sv", "Sz"):
            result[name] = (nan, unit)
        result.update({"Ssk": (nan, ""), "Sku": (nan, ""), "Sdq": (nan, ""), "Sdr": (nan, "%")})
        result.update({"Sal": (nan, "m"), "Str": (nan, ""), "Std": (nan, "°"), "Correlation length": (nan, "m")})
        return result
    mean = float(z.mean())
    d = z - mean
    sq = float(np.sqrt(np.mean(d ** 2)))
    result.update({
        "Mean": (mean, unit), "Median": (float(np.median(z)), unit),
        "Min": (float(z.min()), unit), "Max": (float(z.max()), unit),
        "Sa": (float(np.mean(np.abs(d))), unit), "Sq": (sq, unit),
        "Ssk": (float(np.mean(d ** 3) / sq ** 3) if sq > 0 else nan, ""),
        "Sku": (float(np.mean(d ** 4) / sq ** 4) if sq > 0 else nan, ""),
        "Sp": (float(d.max()), unit), "Sv": (float(-d.min()), unit), "Sz": (float(z.max() - z.min()), unit),
    })
    sdq, sdr = nan, nan
    if unit == "m":
        gy, gx = np.gradient(np.where(use, state.data, np.nan), state.dy, state.dx)      # NaN outside the region
        slope2 = gx ** 2 + gy ** 2
        slope2 = slope2[np.isfinite(slope2)]
        if slope2.size:
            sdq = float(np.sqrt(slope2.mean()))
            sdr = float(100.0 * (np.mean(np.sqrt(1.0 + slope2)) - 1.0))
    result["Sdq"] = (sdq, "")
    result["Sdr"] = (sdr, "%")
    if state.mask.any() and mode == "all":
        step, step_error = masked_step_height(state)
        result["Step height (mask - rest)"] = (step, unit)
        result["Step height error"] = (step_error, unit)
    if texture:
        try:
            parameters = texture_parameters(state, mode)
            length = line_functions(state, "x", mode)["correlation_length"]
        except ops.OpError:
            parameters, length = {"Sal": nan, "Str": nan, "Std": nan}, nan
        result.update({"Sal": (parameters["Sal"], "m"), "Str": (parameters["Str"], ""), "Std": (parameters["Std"], "°"),
                       "Correlation length": (length, "m")})
    return result


def bearing_curve(state: ops.State, mode: str = "all") -> tuple:
    """ Abbott-Firestone curve: (percent of the pixels above a height, height), highest first. """
    z = np.sort(state.data[region_pixels(state.data, state.mask, mode)])[::-1]
    if z.size == 0:
        return np.array([]), np.array([])
    return 100.0 * (np.arange(z.size) + 0.5) / z.size, z


@dataclasses.dataclass
class Profile:
    distance: np.ndarray            # m along the line
    mean: np.ndarray                # average across the width
    std: np.ndarray                 # standard deviation across the width (0 for width 1)
    minimum: np.ndarray
    maximum: np.ndarray
    length: float                   # m
    unit: str
    angle: float = 0.0              # direction of the line in degrees, counterclockwise from +x (y up)
    start: Optional[tuple] = None   # end points (x, y) in m as on the axes of image(), for drawing the line on it
    end: Optional[tuple] = None

    def roughness(self) -> dict:
        """ Ra, Rq and the height range of the mean profile after removing its linear trend. """
        ok = np.isfinite(self.mean)
        if ok.sum() < 3:
            return {"Ra": math.nan, "Rq": math.nan, "Rz": math.nan}
        x, y = self.distance[ok], self.mean[ok]
        residual = y - np.polyval(np.polyfit(x, y, 1), x)
        return {"Ra": float(np.mean(np.abs(residual))), "Rq": float(np.sqrt(np.mean(residual ** 2))),
                "Rz": float(residual.max() - residual.min())}


def line_profile(state: ops.State, start: tuple, end: tuple, width: int = 1) -> Profile:
    """ Profile from start to end (pixel coordinates x, y), averaged over `width` pixels across the line. The width
        is measured perpendicular to the line in physical units, so images with different x and y pixel size work. """
    width = max(1, int(width))
    dx, dy = state.dx, state.dy
    p0 = np.array([start[0] * dx, start[1] * dy])
    p1 = np.array([end[0] * dx, end[1] * dy])
    length = float(np.hypot(*(p1 - p0)))
    if length < min(dx, dy):
        raise ops.OpError("The line is shorter than one pixel.")
    direction = (p1 - p0) / length
    normal = np.array([-direction[1], direction[0]])
    step = min(dx, dy)
    n = int(math.ceil(length / step)) + 1
    s = np.linspace(0.0, length, n)
    offsets = (np.arange(width) - (width - 1) / 2.0) * step
    px = (p0[0] + s[None, :] * direction[0] + offsets[:, None] * normal[0]) / dx
    py = (p0[1] + s[None, :] * direction[1] + offsets[:, None] * normal[1]) / dy
    data = np.where(np.isfinite(state.data), state.data, np.nan)
    inside = (px >= -0.5) & (px <= data.shape[1] - 0.5) & (py >= -0.5) & (py <= data.shape[0] - 0.5)
    # bilinear interpolation of the valid neighbors; NaN pixels spread into the neighbor only
    values = ndimage.map_coordinates(data, [py.ravel(), px.ravel()], order=1, mode="nearest", cval=np.nan,
                                     prefilter=False).reshape(px.shape)
    values = np.where(inside, values, np.nan)
    with warnings.catch_warnings():                       # all-NaN columns (outside the image) are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        return Profile(s, np.nanmean(values, axis=0), np.nanstd(values, axis=0) if width > 1 else np.zeros(n),
                       np.nanmin(values, axis=0), np.nanmax(values, axis=0), length, state.unit,
                       math.degrees(math.atan2(-(p1 - p0)[1], (p1 - p0)[0])))


def _windowed(data: np.ndarray) -> tuple:
    """ Mean-removed, Hann-windowed copy and the mean of the squared window. """
    filled = np.where(np.isfinite(data), data, np.nanmean(data))
    window = np.outer(np.hanning(data.shape[0]), np.hanning(data.shape[1]))
    return (filled - filled.mean()) * window, float(np.mean(window ** 2))


def spectrum_2d(state: ops.State) -> tuple:
    """ (log10 amplitude of the centered 2D spectrum, kx axis, ky axis in 1/m) for displaying and selecting spots. """
    windowed, _ = _windowed(state.data)
    amplitude = np.abs(sfft.fftshift(sfft.fft2(windowed)))
    kx = sfft.fftshift(sfft.fftfreq(windowed.shape[1], state.dx))
    ky = sfft.fftshift(sfft.fftfreq(windowed.shape[0], state.dy))
    return np.log10(amplitude + np.finfo(float).tiny), kx, ky


def psd_radial(state: ops.State, bins: int = 60, counts: bool = False) -> tuple:
    """ Radially averaged power spectral density: (k in 1/m, PSD in unit² m²), with counts=True also the number of
        Fourier coefficients in each bin. Normalized so that the integral over the whole spectrum equals the variance
        of the image. """
    windowed, window_power = _windowed(state.data)
    ny, nx = windowed.shape
    power = np.abs(sfft.fftshift(sfft.fft2(windowed))) ** 2 * state.dx * state.dy / (nx * ny * window_power)
    kx = sfft.fftshift(sfft.fftfreq(nx, state.dx))
    ky = sfft.fftshift(sfft.fftfreq(ny, state.dy))
    k = np.hypot(*np.meshgrid(kx, ky))
    k_min = 1.0 / max(state.size_x, state.size_y)
    k_max = 0.5 / max(state.dx, state.dy)
    edges = np.logspace(np.log10(k_min), np.log10(k_max), bins + 1)
    index = np.digitize(k.ravel(), edges) - 1
    ok = (index >= 0) & (index < bins)
    total = np.bincount(index[ok], weights=power.ravel()[ok], minlength=bins)
    count = np.bincount(index[ok], minlength=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    keep = count > 0
    if counts:
        return centers[keep], total[keep] / count[keep], count[keep]
    return centers[keep], total[keep] / count[keep]


def rms_by_scale(state: ops.State, sizes: list) -> list:
    """ [(window size in m, mean Sq of the non-overlapping square windows)] for window sizes given in pixels. """
    out = []
    ny, nx = state.data.shape
    for n in sizes:
        n = int(n)
        if n < 4 or n > min(ny, nx):
            continue
        values = []
        for y in range(0, ny - n + 1, n):
            for x in range(0, nx - n + 1, n):
                window = state.data[y:y + n, x:x + n]
                use = np.isfinite(window)
                if use.sum() == window.size:
                    # a plane is removed from each window so that tilt does not count as roughness at large scales
                    yy, xx = np.mgrid[:n, :n]
                    design = np.c_[np.ones(n * n), xx.ravel(), yy.ravel()]
                    coeff = np.linalg.lstsq(design, window.ravel(), rcond=None)[0]
                    values.append(np.std(window.ravel() - design @ coeff))
        if values:
            out.append((n * max(state.dx, state.dy), float(np.mean(values))))
    return out


# ---- correlation functions and texture parameters (ISO 25178) ----

def _region_values(state: ops.State, mode: str) -> tuple:
    """ (data with the mean of the region removed and 0 outside the region, region as float). """
    use = region_pixels(state.data, state.mask, mode)
    if use.sum() < 16:
        raise ops.OpError("Too few pixels in the region.")
    z = np.where(use, state.data - state.data[use].mean(), 0.0)
    return z, use.astype(float)


def acf_2d(state: ops.State, mode: str = "all") -> tuple:
    """ Areal autocorrelation function, normalized to 1 at zero shift: (acf, tau_x, tau_y in m), centered. Every
        shift is averaged over the pixel pairs that both lie in the region (unbiased, works with masks). Shifts with
        fewer than a quarter of the pixels in common are NaN. """
    z, m = _region_values(state, mode)
    ny, nx = z.shape
    size = (sfft.next_fast_len(2 * ny), sfft.next_fast_len(2 * nx))

    def corr(a, b):
        return sfft.irfft2(sfft.rfft2(a, size) * np.conj(sfft.rfft2(b, size)), size)

    products, pairs = corr(z, z), np.round(corr(m, m))
    with np.errstate(all="ignore"):
        acf = products / pairs
    acf[pairs < 0.25 * m.sum()] = np.nan
    acf = np.roll(acf, (ny - 1, nx - 1), axis=(0, 1))[:2 * ny - 1, :2 * nx - 1]
    acf = acf / acf[ny - 1, nx - 1]
    tau_x = (np.arange(2 * nx - 1) - (nx - 1)) * state.dx
    tau_y = (np.arange(2 * ny - 1) - (ny - 1)) * state.dy
    return acf, tau_x, tau_y


def _decay_lengths(acf: np.ndarray, tau_x: np.ndarray, tau_y: np.ndarray, level: float, angles: np.ndarray) -> np.ndarray:
    """ Distance along every direction (angles in degrees from +x, counterclockwise with y up) at which the ACF
        first falls to level; inf if it does not within the computed range. """
    cy, cx = (acf.shape[0] - 1) // 2, (acf.shape[1] - 1) // 2
    dx, dy = tau_x[1] - tau_x[0], tau_y[1] - tau_y[0]
    step = min(dx, dy) / 2
    radius = np.arange(0.0, min(tau_x[-1], tau_y[-1]), step)
    out = np.full(angles.size, np.inf)
    filled = np.nan_to_num(acf, nan=-2.0)
    for i, angle in enumerate(np.radians(angles)):
        px = cx + radius * np.cos(angle) / dx
        py = cy - radius * np.sin(angle) / dy                      # rows go down, y goes up
        values = ndimage.map_coordinates(filled, [py, px], order=1, mode="nearest")
        below = np.flatnonzero(values <= level)
        if below.size:
            k = below[0]
            if k == 0:
                out[i] = 0.0
            else:
                a, b = values[k - 1], values[k]
                out[i] = radius[k - 1] + (a - level) / (a - b) * step if a != b else radius[k]
    return out


def texture_direction(state: ops.State, mode: str = "all") -> float:
    """ Std (ISO 25178): direction of the lay in degrees (0 to 180, counterclockwise from +x with y up), from the
        angular power spectrum: the lay is perpendicular to the direction that holds the most power. """
    z, m = _region_values(state, mode)
    window = np.outer(np.hanning(z.shape[0]), np.hanning(z.shape[1]))
    power = np.abs(sfft.fftshift(sfft.fft2(z * window))) ** 2
    kx = sfft.fftshift(sfft.fftfreq(z.shape[1], state.dx))
    ky = -sfft.fftshift(sfft.fftfreq(z.shape[0], state.dy))        # y up
    KX, KY = np.meshgrid(kx, ky)
    k = np.hypot(KX, KY)
    # only frequencies at least 10 steps from the center: there the grid resolves the angle to a few degrees, and the
    # leakage of the window (one step around strong low-frequency peaks, e.g. scan line offsets) cannot fake a direction
    k_min = 10.0 / min(state.size_x, state.size_y)
    use = (k > k_min) & (k < 0.5 / max(state.dx, state.dy))
    angle = np.degrees(np.arctan2(KY[use], KX[use])) % 180.0
    histogram = np.bincount(np.minimum((angle / 1.0).astype(int), 179), weights=power[use], minlength=180)
    smooth = ndimage.uniform_filter1d(histogram, 5, mode="wrap")
    peak = float(np.argmax(smooth)) + 0.5
    return (peak + 90.0) % 180.0


def texture_parameters(state: ops.State, mode: str = "all", level: float = 0.2) -> dict:
    """ {'Sal': m, 'Str': ratio, 'Std': degrees} (ISO 25178): Sal, the shortest distance in which the ACF falls to
        level (0.2); Str, the ratio of the shortest to the longest such distance (1: isotropic, near 0: strongly
        directional; NaN if the ACF does not fall to the level in some direction within half the image); Std,
        the texture direction. """
    acf, tau_x, tau_y = acf_2d(state, mode)
    angles = np.arange(0.0, 180.0, 1.0)
    lengths = _decay_lengths(acf, tau_x, tau_y, level, angles)
    sal = float(np.min(lengths))
    longest = float(np.max(lengths))
    return {"Sal": sal if math.isfinite(sal) else math.nan,
            "Str": sal / longest if math.isfinite(longest) and longest > 0 else math.nan,
            "Std": texture_direction(state, mode)}


def line_functions(state: ops.State, direction: str = "x", mode: str = "all") -> dict:
    """ One-dimensional functions along the rows ('x', the fast scan axis) or the columns ('y'), averaged over all
        lines, every line with its own mean removed (line leveling): {'tau' m, 'acf' normalized to 1, 'hhcf' unit²,
        'correlation_length' m (ACF = 1/e), 'correlation_length_02' m (ACF = 0.2)}. """
    if direction not in ("x", "y"):
        raise ops.OpError("The direction must be x or y.")
    use = region_pixels(state.data, state.mask, mode)
    data = state.data if direction == "x" else state.data.T
    use = use if direction == "x" else use.T
    pitch = state.dx if direction == "x" else state.dy
    n = data.shape[1]
    with np.errstate(all="ignore"):
        means = np.nansum(np.where(use, data, 0.0), axis=1) / use.sum(axis=1)
    z = np.where(use, data - np.nan_to_num(means)[:, None], 0.0)
    m = use.astype(float)
    size = sfft.next_fast_len(2 * n)

    def corr(a, b):
        return sfft.irfft(sfft.rfft(a, size, axis=1) * np.conj(sfft.rfft(b, size, axis=1)), size, axis=1)[:, :n]

    pairs = np.round(corr(m, m).sum(axis=0))
    zz = corr(z, z).sum(axis=0)
    z2m = corr(z * z, m).sum(axis=0) + corr(m, z * z).sum(axis=0)
    keep = pairs >= max(16, 0.1 * pairs[0])
    with np.errstate(all="ignore"):
        acf = zz / pairs
        hhcf = (z2m - 2 * zz) / pairs
    acf, hhcf = acf[keep] / acf[0], hhcf[keep]
    tau = np.arange(n)[keep] * pitch

    def crossing(level):
        below = np.flatnonzero(acf <= level)
        if not below.size:
            return math.nan
        k = below[0]
        return float(tau[k - 1] + (acf[k - 1] - level) / (acf[k - 1] - acf[k]) * pitch) if k else 0.0

    return {"tau": tau, "acf": acf, "hhcf": hhcf, "correlation_length": crossing(1 / math.e),
            "correlation_length_02": crossing(0.2)}


def slope_distributions(state: ops.State, mode: str = "all", bins: int = 90) -> dict:
    """ Local slopes from central differences (height data): {'inclination': (centers in degrees, fraction of
        pixels), 'azimuth': (centers in degrees 0 to 360, counterclockwise from +x with y up, slope-weighted share)}.
        The azimuth is the direction in which the surface rises. """
    if state.unit != "m":
        raise ops.OpError("Slope distributions need height data in m.")
    gy, gx = np.gradient(np.where(np.isfinite(state.data), state.data, np.nan), state.dy, state.dx)
    use = region_pixels(state.data, state.mask, mode) & np.isfinite(gx) & np.isfinite(gy)
    gx, gy = gx[use], -gy[use]                                       # y up
    slope = np.hypot(gx, gy)
    inclination = np.degrees(np.arctan(slope))
    top = max(float(np.percentile(inclination, 99.5)), 1e-6) if inclination.size else 1.0
    counts, edges = np.histogram(inclination, bins=bins, range=(0.0, top))
    azimuth = np.degrees(np.arctan2(gy, gx)) % 360.0
    weights, a_edges = np.histogram(azimuth, bins=72, range=(0.0, 360.0), weights=slope)
    total = weights.sum()
    return {"inclination": (0.5 * (edges[:-1] + edges[1:]), counts / max(counts.sum(), 1)),
            "azimuth": (0.5 * (a_edges[:-1] + a_edges[1:]), weights / total if total > 0 else weights)}


# ---- critical dimension of a profile ----

@dataclasses.dataclass
class CDFeature:
    center: float                   # m along the profile (middle of the 50 % crossings)
    widths: dict                    # level (fraction of the height, from the bottom) -> width in m
    edges: dict                     # level -> (left, right) positions in m
    angle_left: float               # sidewall angles in degrees from the horizontal (90 = vertical), NaN if unknown
    angle_right: float
    height: float                   # local height: median top of the feature minus median of the floor around it


@dataclasses.dataclass
class CDResult:
    kind: str                       # 'lines' (raised) or 'trenches'
    levels: tuple
    bottom: float                   # global levels of the profile (median of the lower / upper class)
    top: float
    height: float                   # top - bottom (for trenches: the depth)
    features: list
    pitch: float                    # mean center distance of neighboring features (NaN with fewer than two)
    pitch_std: float
    skipped: int                    # features cut by the ends of the profile


def _crossing(x: np.ndarray, z: np.ndarray, start: int, stop: int, level: float) -> float:
    """ Position where z first reaches level walking from index start towards stop (linear interpolation). """
    step = 1 if stop >= start else -1
    for i in range(start, stop + step, step):
        if z[i] >= level:
            if i == start:
                return float(x[i])
            j = i - step
            return float(x[j] + (level - z[j]) / (z[i] - z[j]) * (x[i] - x[j]))
    return math.nan


def critical_dimension(profile: Profile, kind: str = "lines", levels=(0.1, 0.5, 0.9)) -> CDResult:
    """ Critical dimension of a line cut across lines (raised features) or trenches: the profile is split into a
        lower and an upper class (Otsu), their medians are the bottom and top levels; every feature that lies
        completely in the profile gets its width at the given levels (fractions of the height measured from the
        bottom, the outer crossings of the flanks), its sidewall angles (line fit of each flank between 20 and 80 %
        of the height) and its local height; the pitch is the mean distance of the feature centers. """
    if kind not in ("lines", "trenches"):
        raise ops.OpError("The features must be 'lines' or 'trenches'.")
    levels = tuple(sorted(float(f) for f in levels))
    if not levels or not all(0.0 < f < 1.0 for f in levels):
        raise ops.OpError("The levels must be between 0 and 100 % of the height.")
    ok = np.isfinite(profile.mean)
    if ok.sum() < 10:
        raise ops.OpError("The profile has too few valid points.")
    x = profile.distance[ok]
    z = profile.mean[ok] if kind == "lines" else -profile.mean[ok]        # trenches: measured upside down
    split = ops.otsu_threshold(z)
    low, high = z[z <= split], z[z > split]
    if low.size < 3 or high.size < 3:
        raise ops.OpError("No two height levels found: is there a feature on the line?")
    bottom, top = float(np.median(low)), float(np.median(high))
    height = top - bottom
    if height <= 0:
        raise ops.OpError("No height difference between features and floor.")
    # trenches: level f from the bottom of the trench is level 1 - f of the inverted profile
    inverted = {f: (1.0 - f if kind == "trenches" else f) for f in levels}
    above = z >= bottom + 0.5 * height
    runs, count = ndimage.label(above)
    features, skipped = [], 0
    for number in range(1, count + 1):
        idx = np.flatnonzero(runs == number)
        i0, i1 = int(idx[0]), int(idx[-1])
        if i0 == 0 or i1 == len(z) - 1:
            skipped += 1
            continue
        # the flanks reach to the lowest points between this feature and its neighbors (or the profile ends)
        left_stop = int(np.flatnonzero(runs[:i0] > 0)[-1]) + 1 if (runs[:i0] > 0).any() else 0
        right_stop = i1 + 1 + int(np.flatnonzero(runs[i1 + 1:] > 0)[0]) - 1 if (runs[i1 + 1:] > 0).any() else len(z) - 1
        valley_left = left_stop + int(np.argmin(z[left_stop:i0 + 1]))
        valley_right = i1 + int(np.argmin(z[i1:right_stop + 1]))
        widths, edges = {}, {}

        def flank(level_fraction, side):
            """ Crossing of a level on one flank, walked from the 50 % crossing of that flank (i0 on the left, i1 on
                the right): outward (into the valley) for levels below 50 %, inward (onto the top) above; a particle
                on the floor or a dip on the top further away does not matter. """
            level = bottom + level_fraction * height
            edge = i0 if side == "left" else i1
            if level_fraction <= 0.5:
                stop = valley_left if side == "left" else valley_right
                step = -1 if side == "left" else 1
                for i in range(edge, stop + step, step):
                    if z[i] < level:
                        j = i - step
                        return float(x[i] + (level - z[i]) / (z[j] - z[i]) * (x[j] - x[i]))
                return math.nan
            step = 1 if side == "left" else -1
            stop = i1 if side == "left" else i0
            for i in range(edge, stop + step, step):
                if z[i] >= level:
                    j = i - step
                    if not 0 <= j < len(z) or z[j] >= level:
                        return float(x[i])
                    return float(x[j] + (level - z[j]) / (z[i] - z[j]) * (x[i] - x[j]))
            return math.nan

        for f in levels:
            a, b = flank(inverted[f], "left"), flank(inverted[f], "right")
            if math.isfinite(a) and math.isfinite(b) and b > a:
                widths[f], edges[f] = b - a, (a, b)
            else:
                widths[f], edges[f] = math.nan, (math.nan, math.nan)
        angles = []
        for side, sign in (("left", 1.0), ("right", -1.0)):
            p20, p80 = flank(0.2, side), flank(0.8, side)
            angle = math.nan
            if math.isfinite(p20) and math.isfinite(p80) and profile.unit == "m":
                sel = (x >= min(p20, p80)) & (x <= max(p20, p80))
                if sel.sum() >= 2:
                    slope = np.polyfit(x[sel], z[sel], 1)[0]
                    angle = math.degrees(math.atan(sign * slope))
                elif p80 != p20:
                    angle = math.degrees(math.atan(0.6 * height / abs(p80 - p20)))
                else:
                    angle = 90.0
            angles.append(angle)
        floor = np.concatenate([z[left_stop:valley_left + 1], z[valley_right:right_stop + 1]])
        floor = floor[floor < bottom + 0.5 * height]
        local_height = float(np.median(z[i0:i1 + 1]) - (np.median(floor) if floor.size else bottom))
        mid = edges.get(0.5, edges[levels[len(levels) // 2]])
        center = 0.5 * (mid[0] + mid[1]) if math.isfinite(mid[0]) else float(x[(i0 + i1) // 2])
        features.append(CDFeature(center, widths, edges, angles[0], angles[1], local_height))
    centers = np.array([f.center for f in features])
    gaps = np.diff(centers)
    if kind == "trenches":
        bottom, top = -top, -bottom
    return CDResult(kind, levels, bottom, top, height, features, float(gaps.mean()) if gaps.size else math.nan,
                    float(gaps.std()) if gaps.size > 1 else math.nan, skipped)


# ---- step height ----

@dataclasses.dataclass
class Step:
    position: float                 # m along the profile (the edge)
    height: float                   # upper minus lower side: positive for a step up (left to right)
    error: float                    # standard error of the height from the joint fit
    left: tuple                     # (start, end) in m of the part of the left plateau used for the fit
    right: tuple


@dataclasses.dataclass
class StepResult:
    steps: list
    plateaus: list                  # (start, end) in m of every plateau found
    slope: float                    # common slope of the fit (tilt), unit per m
    lines: list                     # (x0, x1, z0, z1) of the fitted plateau lines, for drawing


def _step_edges(x: np.ndarray, z: np.ndarray, window: int, threshold: float, min_height: float = 0.0) -> list:
    """ Indices of height steps. The detector is the mean of the next window minus the mean of the previous window
        (its median, the contribution of a tilt, removed). Its noise is the larger of the pixel noise and its own robust
        spread (the roughness at the scale of the window; steps are rare, so they hardly enter the median). Every
        connected region where it exceeds threshold times the noise, and min_height, gives one step at its strongest
        point. """
    n = len(z)
    total = np.concatenate([[0.0], np.cumsum(z)])
    centers = np.arange(window, n - window + 1)
    jump = (total[centers + window] - total[centers]) / window - (total[centers] - total[centers - window]) / window
    jump = jump - np.median(jump)
    noise = 1.4826 * np.median(np.abs(np.diff(z) - np.median(np.diff(z)))) / math.sqrt(2.0) * math.sqrt(2.0 / window)
    strength = np.abs(jump)
    quiet = np.ones(strength.size, bool)            # the spread of the detector away from steps (many steps on a
    spread = 1.4826 * float(np.median(strength))    # line would inflate it): iterate, leaving out found steps
    for _ in range(4):
        loud = strength > threshold * max(noise, spread, 1e-300)
        quiet = ~ndimage.binary_dilation(loud, iterations=window) if loud.any() else np.ones(strength.size, bool)
        if quiet.sum() < max(10, strength.size // 10):
            break
        spread = 1.4826 * float(np.median(strength[quiet]))
    noise = max(noise, spread)
    above = (strength > threshold * max(noise, 1e-300)) & (strength > min_height)
    labels, count = ndimage.label(above)
    edges = []
    for number in range(1, count + 1):
        idx = np.flatnonzero(labels == number)
        edges.append(int(centers[idx[np.argmax(strength[idx])]]))
    merged = []                                          # regions closer than a window are one step
    for edge in edges:
        if merged and edge - merged[-1] < window:
            continue
        merged.append(edge)
    return merged


def step_heights(profile: Profile, min_plateau: int = 20, exclusion: float = 0.25, threshold: float = 8.0,
                 min_height: float = 0.0) -> StepResult:
    """ Step heights of a profile across one or more steps (ISO 5436-1 style): steps are found where the mean height
        over min_plateau / 2 points jumps by more than threshold noise levels; of every plateau between two steps the
        outer `exclusion` fraction at each end is left out (edge rounding, tip shape); all plateaus are fitted together
        with one common slope and one offset each, and every step height is the difference of neighboring offsets,
        with its standard error. A tilt of the profile therefore does not enter the heights. Steps smaller than
        min_height (in the unit of the profile) are ignored. """
    if not 0 <= exclusion < 0.5:
        raise ops.OpError("The edge exclusion must be between 0 and 50 %.")
    ok = np.isfinite(profile.mean)
    x, z = profile.distance[ok], profile.mean[ok]
    window = max(3, int(min_plateau) // 2)
    if len(z) < 4 * window:
        raise ops.OpError("The line is too short for this minimum plateau length.")
    edges = _step_edges(x, z, window, threshold, min_height)
    if not edges:
        raise ops.OpError("No step found on the line (lower the minimum plateau or the threshold).")
    bounds = [0] + edges + [len(z)]
    plateaus, used = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        cut = int(round((b - a) * exclusion))
        lo, hi = a + cut, b - cut
        if hi - lo < 3:
            raise ops.OpError("A plateau is too short for the edge exclusion: shorten the exclusion or the steps.")
        plateaus.append((float(x[a]), float(x[b - 1])))
        used.append((lo, hi))
    # joint fit: z = slope * x + offset_k on the used part of plateau k
    rows = sum(hi - lo for lo, hi in used)
    design = np.zeros((rows, 1 + len(used)))
    values = np.zeros(rows)
    r = 0
    for k, (lo, hi) in enumerate(used):
        m = hi - lo
        design[r:r + m, 0] = x[lo:hi] - x.mean()
        design[r:r + m, 1 + k] = 1.0
        values[r:r + m] = z[lo:hi]
        r += m
    coeff, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coeff
    dof = max(rows - design.shape[1], 1)
    covariance = np.linalg.pinv(design.T @ design) * (residual @ residual) / dof
    slope, offsets = float(coeff[0]), coeff[1:]
    steps, lines = [], []
    for k, (lo, hi) in enumerate(used):
        x0, x1 = float(x[lo]), float(x[hi - 1])
        lines.append((x0, x1, offsets[k] + slope * (x0 - x.mean()), offsets[k] + slope * (x1 - x.mean())))
    for k, edge in enumerate(edges):
        variance = covariance[1 + k, 1 + k] + covariance[2 + k, 2 + k] - 2 * covariance[1 + k, 2 + k]
        steps.append(Step(float(0.5 * (x[edge - 1] + x[edge])), float(offsets[k + 1] - offsets[k]),
                          float(math.sqrt(max(variance, 0.0))), (float(x[used[k][0]]), float(x[used[k][1] - 1])),
                          (float(x[used[k + 1][0]]), float(x[used[k + 1][1] - 1]))))
    return StepResult(steps, plateaus, slope, lines)


def masked_step_height(state: ops.State, border: int = 3) -> tuple:
    """ (height, standard error) of the masked area above the rest (ISO 5436-1 areal style): one plane for the whole
        image plus an offset for the masked pixels, fitted to the pixels further than `border` pixels from the mask
        edge. NaN if one of the regions is too small. """
    mask = state.mask
    if not mask.any() or mask.all():
        return math.nan, math.nan
    inner = ndimage.binary_erosion(mask, iterations=border, border_value=1)
    outer = ndimage.binary_erosion(~mask, iterations=border, border_value=1)
    use = (inner | outer) & np.isfinite(state.data)
    if (inner & use).sum() < 10 or (outer & use).sum() < 10:
        return math.nan, math.nan
    ny, nx = state.data.shape
    yy, xx = np.nonzero(use)
    design = np.c_[np.ones(xx.size), (xx - nx / 2) * state.dx, (yy - ny / 2) * state.dy, inner[yy, xx].astype(float)]
    values = state.data[yy, xx]
    coeff, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coeff
    variance = (residual @ residual) / max(values.size - 4, 1)
    error = math.sqrt(max(np.linalg.pinv(design.T @ design)[3, 3] * variance, 0.0))
    return float(coeff[3]), error


# ---- lattice ----

@dataclasses.dataclass
class Lattice:
    """ Two primitive lattice vectors in m (x to the right, y up, as the image axes are shown). """
    a: tuple
    b: tuple
    peaks: list                     # ACF maxima used (tau_x, tau_y in m, y down as in the ACF array), for display
    region: str = "all"

    @property
    def length_a(self) -> float:
        return math.hypot(*self.a)

    @property
    def length_b(self) -> float:
        return math.hypot(*self.b)

    @property
    def angle(self) -> float:
        """ Angle between a and b in degrees. """
        a, b = np.array(self.a), np.array(self.b)
        return math.degrees(math.acos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))

    @property
    def orientation(self) -> float:
        """ Direction of a in degrees counterclockwise from the x axis. """
        return math.degrees(math.atan2(self.a[1], self.a[0]))

    @property
    def cell_area(self) -> float:
        return abs(self.a[0] * self.b[1] - self.a[1] * self.b[0])

    def summary(self) -> str:
        def length(value):
            return f"{value * 1e9:.4g} nm" if value < 1e-6 else f"{value * 1e6:.4g} µm"
        ratio = self.length_b / self.length_a
        kind = ""
        if abs(ratio - 1) < 0.03 and abs(self.angle - 60) < 3:
            kind = " (hexagonal)"
        elif abs(ratio - 1) < 0.03 and abs(self.angle - 90) < 3:
            kind = " (square)"
        elif abs(self.angle - 90) < 3:
            kind = " (rectangular)"
        area = self.cell_area
        area_text = f"{area * 1e18:.4g} nm²" if area < 1e-12 else f"{area * 1e12:.4g} µm²"
        return (f"|a| = {length(self.length_a)}, |b| = {length(self.length_b)}, angle a-b = {self.angle:.2f}°{kind}\n"
                f"a points {self.orientation:.2f}° from the x axis; a = ({length(self.a[0])}, {length(self.a[1])}), "
                f"b = ({length(self.b[0])}, {length(self.b[1])})\n"
                f"Unit cell area {area_text}, {1 / area * 1e-12:.4g} cells per µm²; refined with "
                f"{len(self.peaks)} ACF maxima")


def _subpixel_peak(acf: np.ndarray, iy: int, ix: int) -> tuple:
    """ Peak position (row, column) refined by parabolas through the 3 x 3 neighbourhood. """
    ny, nx = acf.shape

    def offset(m, c, p):
        denominator = m - 2 * c + p
        return 0.5 * (m - p) / denominator if denominator < 0 and np.isfinite(denominator) else 0.0

    dy = offset(acf[iy - 1, ix], acf[iy, ix], acf[iy + 1, ix]) if 0 < iy < ny - 1 else 0.0
    dx = offset(acf[iy, ix - 1], acf[iy, ix], acf[iy, ix + 1]) if 0 < ix < nx - 1 else 0.0
    return iy + float(np.clip(dy, -0.5, 0.5)), ix + float(np.clip(dx, -0.5, 0.5))


def acf_maxima(acf: np.ndarray, min_value: float = 0.05) -> list:
    """ Local maxima of an ACF (rows, columns, value), the central peak excluded, sub-pixel refined, sorted by value. """
    filled = np.where(np.isfinite(acf), acf, -np.inf)
    local = (filled == ndimage.maximum_filter(filled, size=5, mode="constant", cval=-np.inf)) & (filled > min_value)
    cy, cx = (acf.shape[0] - 1) // 2, (acf.shape[1] - 1) // 2
    local[max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2] = False
    rows, cols = np.nonzero(local)
    ny, nx = acf.shape
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    found = []
    for r, c in zip(rows, cols):
        # a lattice maximum falls off in every direction; on the ridge of a line grating one direction stays flat
        step = max(2, int(round(0.12 * math.hypot(r - cy, c - cx))))
        drops = []
        for dr, dc in directions:
            rr, cc = r + dr * step, c + dc * step
            drops.append(filled[r, c] - filled[rr, cc] if 0 <= rr < ny and 0 <= cc < nx else np.nan)
        drops = np.array(drops)
        if np.isfinite(drops).sum() < 6:
            continue
        if np.nanmin(drops) < 0.15 * np.nanmax(drops) or np.nanmax(drops) <= 0:
            continue
        found.append((*_subpixel_peak(acf, r, c), float(acf[r, c])))
    return sorted(found, key=lambda p: -p[2])


def nearest_acf_peak(acf: np.ndarray, row: float, col: float, radius: int = 6) -> tuple:
    """ The ACF maximum within `radius` pixels of (row, col), sub-pixel refined: (row, col). """
    ny, nx = acf.shape
    r0, c0 = int(round(row)), int(round(col))
    y0, y1, x0, x1 = max(r0 - radius, 1), min(r0 + radius + 1, ny - 1), max(c0 - radius, 1), min(c0 + radius + 1, nx - 1)
    if y1 <= y0 or x1 <= x0:
        raise ops.OpError("The point lies outside the autocorrelation map.")
    window = np.where(np.isfinite(acf[y0:y1, x0:x1]), acf[y0:y1, x0:x1], -np.inf)
    iy, ix = np.unravel_index(np.argmax(window), window.shape)
    return _subpixel_peak(acf, y0 + iy, x0 + ix)


def _reduce(a: np.ndarray, b: np.ndarray) -> tuple:
    """ Lagrange-Gauss reduction: the shortest basis of the same lattice, angle between 60 and 90 degrees. """
    for _ in range(50):
        if np.linalg.norm(b) < np.linalg.norm(a):
            a, b = b, a
        m = round(float(a @ b) / float(a @ a))
        if m == 0:
            break
        b = b - m * a
    if a @ b < 0:
        b = -b
    return a, b


def lattice(state: ops.State, mode: str = "all", a: Optional[tuple] = None, b: Optional[tuple] = None,
            refine: bool = True) -> Lattice:
    """ The lattice of a periodic image from its autocorrelation: the two shortest independent vectors between strong
        ACF maxima (or start vectors a, b in m, ACF orientation: y down), refined by a least squares fit of all ACF
        maxima that lie on the lattice. Raises ops.OpError when no lattice is found. """
    acf, tau_x, tau_y = acf_2d(state, mode)
    cy, cx = (acf.shape[0] - 1) // 2, (acf.shape[1] - 1) // 2
    dx, dy = state.dx, state.dy
    maxima = acf_maxima(acf)
    vectors = [(np.array([(c - cx) * dx, (r - cy) * dy]), v) for r, c, v in maxima]
    if a is None or b is None:
        if len(vectors) < 2:
            raise ops.OpError("No 2D lattice found: the autocorrelation has no clear side maxima. (A line grating is "
                              "periodic in one direction only: measure its pitch with a line cut.)")
        strong = [(vec, v) for vec, v in vectors if v >= 0.3 * vectors[0][1]] or vectors
        strong.sort(key=lambda item: np.linalg.norm(item[0]))
        first = strong[0][0]
        second = None
        for vec, _ in strong[1:]:
            cosine = abs(first @ vec) / (np.linalg.norm(first) * np.linalg.norm(vec))
            if cosine < math.cos(math.radians(25)):
                second = vec
                break
        if second is None:
            raise ops.OpError("Only one direction of periodicity found (a line grating?): click the second vector "
                              "by hand or measure the pitch with a line cut.")
        a_vec, b_vec = _reduce(first, second)
        # a real lattice explains most strong maxima nearby; noise gives maxima without order
        basis = np.column_stack([a_vec, b_vec])
        reach = 3.5 * max(np.linalg.norm(a_vec), np.linalg.norm(b_vec))
        tolerance = 0.2 * min(np.linalg.norm(a_vec), np.linalg.norm(b_vec))
        near = [vec for vec, _ in strong if np.linalg.norm(vec) <= reach]
        on = sum(np.linalg.norm(basis @ np.round(np.linalg.solve(basis, vec)) - vec) <= tolerance for vec in near)
        if len(near) < 4 or on < 0.6 * len(near):
            raise ops.OpError("No regular 2D lattice found: the autocorrelation maxima do not form a lattice. (A line "
                              "grating is periodic in one direction only: measure its pitch with a line cut.)")
    else:
        a_vec, b_vec = np.array(a, float), np.array(b, float)
        if abs(a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0]) < 1e-6 * np.linalg.norm(a_vec) * np.linalg.norm(b_vec):
            raise ops.OpError("The vectors a and b are parallel.")
    used = []
    if refine:
        basis = np.column_stack([a_vec, b_vec])
        limit = 0.25 * min(np.linalg.norm(a_vec), np.linalg.norm(b_vec))
        rows, targets, weights = [], [], []
        for vec, value in vectors:
            ij = np.round(np.linalg.solve(basis, vec))
            if not ij.any() or np.linalg.norm(basis @ ij - vec) > limit or np.abs(ij).max() > 12:
                continue
            rows.append(ij)
            targets.append(vec)
            weights.append(value)
            used.append(vec)
        if len(rows) >= 2 and np.linalg.matrix_rank(np.array(rows)) == 2:
            design = np.array(rows) * np.sqrt(np.array(weights))[:, None]
            solution, *_ = np.linalg.lstsq(design, np.array(targets) * np.sqrt(np.array(weights))[:, None], rcond=None)
            a_vec, b_vec = solution[0], solution[1]
    if not used:
        used = [a_vec, b_vec]
    flip = np.array([1.0, -1.0])                     # ACF rows go down, the shown y axis goes up
    a_up, b_up = a_vec * flip, b_vec * flip
    tiny = 1e-6 * np.linalg.norm(a_up)
    a_up[np.abs(a_up) < tiny] = 0.0
    b_up[np.abs(b_up) < 1e-6 * np.linalg.norm(b_up)] = 0.0
    if a_up[1] < 0 or (a_up[1] == 0 and a_up[0] < 0):     # a in the upper half plane, b at <= 90 degrees from it
        a_up = -a_up
    if a_up @ b_up < 0:
        b_up = -b_up
    return Lattice(tuple(map(float, a_up)), tuple(map(float, b_up)), [tuple(map(float, v)) for v in used], mode)


# ---- results of the line cut measurements as text (Line cut panel, report) ----


# ---- curvature ----

@dataclasses.dataclass
class Curvature:
    """ Quadratic surface z = a + b x + c y + d x² + e x y + f y² fitted to a height image, and its principal
    curvatures at the center of the paraboloid (where its slope is zero, so the values are exact there). Positive
    curvature: the surface curves up (a bowl or valley), negative: down (a dome or ridge). Positions are image
    coordinates as shown: x to the right from the left edge, y down from the top edge; angles are counterclockwise from
    the x axis as seen on the screen. """
    curvature_1: float              # 1/m, the larger by magnitude
    curvature_2: float
    angle_1: float                  # degrees, direction of curvature_1 (0 to 180)
    angle_2: float
    center: Optional[tuple]         # (x, y) in m of the stationary point; None for a cylinder (no single center)
    center_z: float                 # m, height of the fit at the center
    center_inside: bool
    bow: float                      # m, peak to valley of the fitted surface over the region
    rms: float                      # m, residual of the fit
    coefficients: tuple             # a, b, c, d, e, f in m and m^-1 (x, y in m from the image center)
    region: str = "all"
    manual_center: bool = False     # True: the center was given and the fit made with its vertex there

    @property
    def radius_1(self) -> float:
        return 1.0 / self.curvature_1 if self.curvature_1 else math.inf

    @property
    def radius_2(self) -> float:
        return 1.0 / self.curvature_2 if self.curvature_2 else math.inf

    @property
    def mean_curvature(self) -> float:
        return 0.5 * (self.curvature_1 + self.curvature_2)

    @property
    def gaussian_curvature(self) -> float:
        return self.curvature_1 * self.curvature_2

    def summary(self) -> str:
        def length(value):
            if not math.isfinite(value):
                return "infinite"
            magnitude = abs(value)
            for limit, scale, unit in ((1e-6, 1e9, "nm"), (1e-3, 1e6, "µm"), (1.0, 1e3, "mm")):
                if magnitude < limit:
                    return f"{value * scale:.4g} {unit}"
            return f"{value:.4g} m"

        def shape(k1, k2):
            if k1 * k2 > 0:
                return "a bowl (curves up in every direction)" if k1 > 0 else "a dome (curves down in every direction)"
            if k1 * k2 < 0:
                return "a saddle"
            return "a cylinder" if k1 or k2 else "flat"

        center = ("none (the surface is a cylinder or flat)" if self.center is None else
                  f"x = {length(self.center[0])}, y = {length(self.center[1])}"
                  + (" (set by hand)" if self.manual_center else "")
                  + ("" if self.center_inside else " (outside the image)"))
        return (f"Radius of curvature R1 = {length(self.radius_1)} along {self.angle_1:.1f}°, "
                f"R2 = {length(self.radius_2)} along {self.angle_2:.1f}° ({shape(self.curvature_1, self.curvature_2)})\n"
                f"Center: {center}; bow (peak to valley of the fit) {length(self.bow)}, fit rms {length(self.rms)}")

    def rows(self) -> list:
        """ (name, value, unit) for a table. """
        rows = [("R1", self.radius_1, "m"), ("R2", self.radius_2, "m"), ("Direction of R1", self.angle_1, "°"),
                ("Direction of R2", self.angle_2, "°"), ("Curvature 1", self.curvature_1, "1/m"),
                ("Curvature 2", self.curvature_2, "1/m"), ("Mean curvature", self.mean_curvature, "1/m"),
                ("Gaussian curvature", self.gaussian_curvature, "1/m²"), ("Bow (peak to valley)", self.bow, "m"),
                ("Fit rms", self.rms, "m")]
        if self.center is not None:
            rows += [("Center x", self.center[0], "m"), ("Center y", self.center[1], "m"),
                     ("Center height", self.center_z, "m")]
        return rows

    def profile(self, which: int, width: float, height: float) -> tuple:
        """ ((x0, y0), (x1, y1)): the line along principal direction 1 or 2 through the center (through the image
            center when the center is outside or there is none), clipped to an image of width x height (m). """
        cx, cy = self.center if self.center is not None and self.center_inside else (width / 2, height / 2)
        angle = math.radians(self.angle_1 if which == 1 else self.angle_2)
        ux, uy = math.cos(angle), -math.sin(angle)              # screen y points down
        start, stop = -math.inf, math.inf
        for c, direction, top in ((cx, ux, width), (cy, uy, height)):
            if abs(direction) > 1e-12:
                low, high = sorted(((0 - c) / direction, (top - c) / direction))
                start, stop = max(start, low), min(stop, high)
        return (cx + ux * start, cy + uy * start), (cx + ux * stop, cy + uy * stop)


def extreme_point(state: ops.State, mode: str = "all", which: str = "max", smooth: float = 2.0) -> tuple:
    """ The highest (which='max') or lowest ('min') point of the region, (x, y) in m in image coordinates (x from the
        left edge, y down from the top edge, pixel centers), of the heights smoothed with a Gaussian of `smooth` pixels (0:
        none), so a single noisy pixel does not decide it. A starting point for the center of the curvature. The image
        is taken as it is: a tilt moves the point downhill-to-uphill (by tilt x radius on a sphere), so level the plane
        first. (Removing a fitted plane here would not do: on a dome that is off the center of the image it pulls the
        point to the center.) """
    if which not in ("max", "min"):
        raise ops.OpError("which must be 'max' or 'min'.")
    use = region_pixels(state.data, state.mask, mode)
    if use.sum() < 3:
        raise ops.OpError("Too few pixels in the region.")
    leveled = np.where(use, state.data, 0.0).astype(float)
    if smooth > 0:
        weight = ndimage.gaussian_filter(use.astype(float), smooth)
        with np.errstate(invalid="ignore", divide="ignore"):
            leveled = ndimage.gaussian_filter(leveled, smooth) / weight
    values = np.where(use, leveled if which == "max" else -leveled, -np.inf)
    row, col = np.unravel_index(int(np.argmax(values)), values.shape)
    return float((col + 0.5) * state.dx), float((row + 0.5) * state.dy)


def curvature(state: ops.State, mode: str = "all", center: Optional[tuple] = None) -> Curvature:
    """ Curvature of a height image from a quadratic fit over the region ('all', 'masked', 'unmasked'): the principal
        radii of curvature, their directions, the center and the bow. For wafers, lenses, spheres, tips and bent
        samples; the heights must be in m. center: (x, y) in m (image coordinates: x from the left edge, y down from
        the top edge) to fix the vertex of the fitted surface there instead of finding it, e.g. the apex of a lens
        that a tilt of the sample would move. """
    if state.unit != "m":
        raise ops.OpError("The curvature needs a height channel (heights in m).")
    use = region_pixels(state.data, state.mask, mode)
    if use.sum() < 12:
        raise ops.OpError("Too few pixels in the region for a quadratic fit.")
    ny, nx = state.data.shape
    rows, cols = np.nonzero(use)
    x = (cols + 0.5) * state.dx                     # image coordinates as shown, m
    y = (rows + 0.5) * state.dy
    x0, y0 = 0.5 * nx * state.dx, 0.5 * ny * state.dy
    scale = max(nx * state.dx, ny * state.dy)
    u, v = (x - x0) / scale, (y - y0) / scale       # well conditioned
    z = state.data[use]
    if center is not None:
        # vertex fixed at the center: no linear terms around it; written in the same coefficients as the free fit
        pu, pv = (float(center[0]) - x0) / scale, (float(center[1]) - y0) / scale
        du, dv = u - pu, v - pv
        q, *_ = np.linalg.lstsq(np.stack([np.ones_like(du), du * du, du * dv, dv * dv], 1), z, rcond=None)
        a0, dd, ee, ff = q
        p = np.array([a0 + dd * pu * pu + ee * pu * pv + ff * pv * pv, -2 * dd * pu - ee * pv, -ee * pu - 2 * ff * pv,
                      dd, ee, ff])
    else:
        p, *_ = np.linalg.lstsq(np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], 1), z, rcond=None)
    design = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], 1)
    fitted = design @ p
    rms = float(np.sqrt(np.mean((z - fitted) ** 2)))
    a, b, c, d, e, f = p[0], p[1] / scale, p[2] / scale, p[3] / scale ** 2, p[4] / scale ** 2, p[5] / scale ** 2
    hessian = np.array([[2 * d, e], [e, 2 * f]])
    values, vectors = np.linalg.eigh(hessian)
    order = np.argsort(-np.abs(values))
    values, vectors = values[order], vectors[:, order]
    values[np.abs(values) < 1e-6 * np.abs(values[0])] = 0.0        # a cylinder: no curvature along its axis
    angles = [math.degrees(math.atan2(-vectors[1, k], vectors[0, k])) % 180.0 for k in range(2)]   # screen y up
    center_given = center
    center, center_z, inside = None, math.nan, False
    if center_given is not None:
        center = (float(center_given[0]), float(center_given[1]))
        pu, pv = center[0] - x0, center[1] - y0
        center_z = float(a + b * pu + c * pv + d * pu ** 2 + e * pu * pv + f * pv ** 2)
        inside = 0 <= center[0] <= nx * state.dx and 0 <= center[1] <= ny * state.dy
    elif abs(values[1]) > 1e-9 * abs(values[0]) and abs(values[0]) > 0:
        offset = np.linalg.solve(hessian, -np.array([b, c]))
        center = (float(x0 + offset[0]), float(y0 + offset[1]))
        center_z = float(a + b * offset[0] + c * offset[1] + d * offset[0] ** 2 + e * offset[0] * offset[1]
                         + f * offset[1] ** 2)
        inside = 0 <= center[0] <= nx * state.dx and 0 <= center[1] <= ny * state.dy
    return Curvature(float(values[0]), float(values[1]), angles[0], angles[1], center, center_z, inside,
                     float(np.ptp(fitted)), rms, (float(a), float(b), float(c), float(d), float(e), float(f)), mode,
                     center_given is not None)
