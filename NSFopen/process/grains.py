# -*- coding: utf-8 -*-
# Ported from the Nanosurf Surface Analyzer (NanosurfImg), where it is covered by tests.
""" Grain (particle) analysis. The grains are the connected areas of the mask, so every mask tool (threshold, slope,
fill, drawing, grow/shrink, remove small areas) defines them; touching grains can be split by a watershed of the
heights. Every grain gets a row of geometric and height parameters. No GUI code. """

import dataclasses
import math
from typing import Optional

import numpy as np
from scipy import ndimage

from . import ops

SPLIT_MODES = ("none", "height", "distance")


@dataclasses.dataclass
class GrainSettings:
    connectivity: int = 8           # 4: grains touching only at a corner are separate; 8: they are one grain
    min_pixels: int = 5             # smaller grains are ignored
    exclude_edge: bool = False      # ignore grains that touch the image border (they are cut, their size is wrong)
    split: str = "none"             # split touching grains: 'height' (one grain per height maximum), 'distance'
                                    # (per maximum of the distance to the grain border, for round grains), 'none'
    separation: float = 3.0         # minimum distance between the maxima of a split, in pixels
    smoothing: float = 1.0          # Gaussian sigma (pixels) applied to the heights before finding maxima

    def check(self):
        if self.connectivity not in (4, 8):
            raise ops.OpError("The connectivity must be 4 or 8.")
        if self.split not in SPLIT_MODES:
            raise ops.OpError(f"The split mode must be one of {', '.join(SPLIT_MODES)}.")
        if self.min_pixels < 1 or self.separation < 1 or self.smoothing < 0:
            raise ops.OpError("Minimum size and separation must be at least 1 pixel, the smoothing not negative.")


@dataclasses.dataclass
class GrainResult:
    labels: np.ndarray              # 0 = background, 1..n = grain number (numbered top to bottom, left to right)
    rows: list                      # one dict per grain: {column: value} in SI units
    summary: dict                   # {name: (value, unit)}
    units: dict                     # column -> unit


COLUMNS = [("Grain", ""), ("Pixels", ""), ("Area", "m²"), ("Equivalent diameter", "m"), ("Perimeter", "m"),
           ("Circularity", ""), ("Major axis", "m"), ("Minor axis", "m"), ("Orientation", "°"),
           ("Max Feret", "m"), ("Max Feret direction", "°"), ("Min Feret", "m"), ("Min Feret direction", "°"),
           ("Feret ratio", ""), ("Convex hull area", "m²"), ("Solidity", ""), ("Surface area", "m²"),
           ("Center x", "m"), ("Center y", "m"), ("Max height", "z"), ("Mean height", "z"), ("Median height", "z"),
           ("Min height", "z"), ("RMS height", "z"), ("Height above base", "z"), ("Volume above zero", "zm²"),
           ("Volume above base", "zm²"), ("Volume above background", "zm²"), ("Touches edge", "")]

COLUMN_HELP = {
    "Pixels": "Number of mask pixels of the grain.",
    "Area": "Projected (flat) area.",
    "Equivalent diameter": "Diameter of the disc with the same projected area.",
    "Perimeter": "Length of the outline in the plane (pixel edges, corrected for the staircase).",
    "Circularity": "4π·area / perimeter²: 1 for a disc, smaller for elongated or ragged grains.",
    "Major axis": "Length of the equivalent ellipse (same second moments) along its long axis.",
    "Minor axis": "Width of the equivalent ellipse.",
    "Orientation": "Direction of the major axis, degrees counterclockwise from the x axis.",
    "Max Feret": "Maximum caliper size: the largest distance between two points of the outline (the widest gap the "
                 "grain could fill). The usual 'length' of a particle (ISO 13322).",
    "Max Feret direction": "Direction of the maximum caliper size, degrees counterclockwise from x.",
    "Min Feret": "Minimum caliper size: the narrowest gap the grain could pass through (its 'width').",
    "Min Feret direction": "Direction in which the minimum caliper size is measured.",
    "Feret ratio": "Max Feret / Min Feret: 1 for round or square grains, large for rods and fibers.",
    "Convex hull area": "Area of the convex hull (a rubber band around the grain).",
    "Solidity": "Area / convex hull area: near 1 for compact grains, lower for agglomerates, branched or concave "
                "grains.",
    "Surface area": "True 3D area of the grain surface (height channels only).",
    "Center x": "x of the center of mass of the grain area.",
    "Center y": "y of the center of mass of the grain area.",
    "Max height": "Highest value in the grain.",
    "Mean height": "Mean value in the grain.",
    "Median height": "Median value in the grain.",
    "Min height": "Lowest value in the grain.",
    "RMS height": "Standard deviation of the values in the grain: the roughness on top of it.",
    "Height above base": "Max height minus the base (the median of the pixels just around the grain). For "
                         "nanoparticles the most reliable size: the lateral sizes are widened by the tip.",
    "Volume above zero": "Volume between the grain surface and z = 0 (set the zero level first).",
    "Volume above base": "Volume above the base level (the median of the pixels just around the grain).",
    "Volume above background": "Volume above a background interpolated under the grain from its surroundings "
                               "(Laplace): correct on tilted or curved backgrounds.",
    "Touches edge": "The grain touches the image border, so it is cut and its size is wrong.",
}


def _local_maxima(values: np.ndarray, inside: np.ndarray, separation: float) -> np.ndarray:
    """ Marker image (1..n) of the maxima of values inside the grains that are separated by at least `separation`
        pixels (plateaus give one marker). """
    size = max(3, int(2 * round(separation) + 1))
    peak = (values == ndimage.maximum_filter(np.where(inside, values, -np.inf), size=size)) & inside
    markers, _ = ndimage.label(peak, structure=np.ones((3, 3)))
    return markers


def _watershed(values: np.ndarray, markers: np.ndarray, inside: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """ Priority flood from the markers: the unlabeled pixel with the highest value next to a labeled one is labeled
        next (so the grains grow downhill from their maxima and meet in the valleys). Restricted to inside. """
    import heapq
    ny, nx = values.shape
    labels = np.where(inside, markers, 0).astype(np.int32).ravel()
    flat_values = values.ravel()
    allowed = inside.ravel()
    steps = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy or dx) and structure[dy + 1, dx + 1]]
    heap = []
    counter = 0

    def push_neighbors(index):
        nonlocal counter
        y, x = divmod(index, nx)
        for dy, dx in steps:
            yy, xx = y + dy, x + dx
            if 0 <= yy < ny and 0 <= xx < nx:
                other = yy * nx + xx
                if allowed[other] and labels[other] == 0:
                    counter += 1
                    heapq.heappush(heap, (-flat_values[other], counter, other, labels[index]))

    for index in np.flatnonzero(labels):
        push_neighbors(int(index))
    while heap:
        _, _, index, label = heapq.heappop(heap)
        if labels[index]:
            continue
        labels[index] = label
        push_neighbors(index)
    return labels.reshape(ny, nx)


def label_grains(state: ops.State, settings: GrainSettings) -> np.ndarray:
    settings.check()
    structure = np.ones((3, 3), int) if settings.connectivity == 8 else ndimage.generate_binary_structure(2, 1)
    inside = state.mask & np.isfinite(state.data)
    labels, count = ndimage.label(inside, structure=structure)
    if count and settings.split != "none":
        if settings.split == "height":
            values = np.where(inside, state.data, 0.0)
            if settings.smoothing > 0:                       # normalized smoothing inside the grains only
                weight = ndimage.gaussian_filter(inside.astype(float), settings.smoothing)
                with np.errstate(all="ignore"):
                    values = ndimage.gaussian_filter(values, settings.smoothing) / weight
        else:
            values = ndimage.distance_transform_edt(inside, sampling=(state.dy, state.dx))
        values = np.where(inside, values, 0.0)
        markers = _local_maxima(np.where(inside, values, -np.inf), inside, settings.separation)
        split = _watershed(values, markers, inside, structure)
        missing = inside & (split == 0)                      # not reached by the flooding: keep the original grain
        if missing.any():
            split[missing] = split.max() + labels[missing]
        labels = _relabel(split)
        count = int(labels.max())
    keep = np.zeros(count + 1, bool)
    if count:
        sizes = ndimage.sum(np.ones_like(labels), labels, np.arange(1, count + 1))
        keep[1:] = sizes >= settings.min_pixels
        if settings.exclude_edge:
            edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
            keep[edge[edge > 0]] = False
    labels = np.where(keep[labels], labels, 0)
    return _relabel(labels)


def _relabel(labels: np.ndarray) -> np.ndarray:
    """ Consecutive numbers 1..n in reading order (by the first pixel of every grain). """
    flat = labels.ravel()
    ids, first = np.unique(flat, return_index=True)
    order = [i for _, i in sorted(zip(first, ids)) if i > 0]
    lookup = np.zeros(int(ids.max()) + 1 if ids.size else 1, np.int32)
    lookup[order] = np.arange(1, len(order) + 1)
    return lookup[labels]


def _perimeter(region: np.ndarray, dx: float, dy: float) -> float:
    """ Length of the pixel edges between the grain and its surroundings (a staircase estimate, multiplied by pi/4
        so that it is unbiased for random orientations of smooth outlines). """
    padded = np.pad(region, 1)
    vertical = np.count_nonzero(padded[1:-1, 1:] != padded[1:-1, :-1])      # edges between x-neighbors
    horizontal = np.count_nonzero(padded[1:, 1:-1] != padded[:-1, 1:-1])    # edges between y-neighbors
    return (vertical * dy + horizontal * dx) * math.pi / 4


def _feret(region: np.ndarray, x0: float, y0: float, dx: float, dy: float) -> tuple:
    """ (max Feret, its direction, min Feret, its direction, convex hull area) of a grain from the corners of its
        pixels. Directions in degrees counterclockwise from +x with y up, in [0, 180). """
    padded = np.pad(region, 1)
    inner = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    ys, xs = np.nonzero(region & ~inner)                 # only pixels with a free side can reach the hull
    height = region.shape[0] + 1
    codes = np.unique(np.concatenate([(xs + a) * height + height - 1 - (ys + b) for a in (0, 1) for b in (0, 1)]))
    column, row = np.divmod(codes, height)
    corners = np.column_stack([column * dx + x0, -(height - 1 - row) * dy - y0])          # y up
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(corners)
    except Exception:
        return (math.nan,) * 5
    points = corners[hull.vertices]
    differences = points[:, None, :] - points[None, :, :]
    distances = np.hypot(differences[..., 0], differences[..., 1])
    i, j = np.unravel_index(np.argmax(distances), distances.shape)
    vector = points[j] - points[i]
    max_size = float(distances[i, j])
    max_direction = math.degrees(math.atan2(vector[1], vector[0])) % 180.0
    min_size, min_direction = math.inf, math.nan
    for k in range(len(points)):                         # the minimum width is perpendicular to a hull edge
        edge = points[(k + 1) % len(points)] - points[k]
        length = math.hypot(*edge)
        if length == 0:
            continue
        normal = np.array([-edge[1], edge[0]]) / length
        width = float(np.ptp(points @ normal))
        if width < min_size:
            min_size, min_direction = width, math.degrees(math.atan2(normal[1], normal[0])) % 180.0
    return max_size, max_direction, min_size, min_direction, float(hull.volume)


def _background(state: ops.State, labels: np.ndarray) -> np.ndarray:
    """ The data with every grain replaced by a Laplace interpolation of its surroundings (NaN if impossible). """
    work = state.copy()
    try:
        ops._laplace_fill(work, (labels > 0) | ~np.isfinite(state.data))
    except ops.OpError:
        return np.full(state.data.shape, np.nan)
    return work.data


def analyze(state: ops.State, settings: GrainSettings = None) -> GrainResult:
    """ Labels the grains of the mask and measures them. Heights ('z' columns) are in the unit of the channel. """
    settings = settings or GrainSettings()
    labels = label_grains(state, settings)
    count = int(labels.max())
    dx, dy = state.dx, state.dy
    pixel_area = dx * dy
    ny, nx = labels.shape
    border = np.zeros(labels.shape, bool)
    border[0], border[-1], border[:, 0], border[:, -1] = True, True, True, True
    data = state.data
    rows = []
    slices = ndimage.find_objects(labels)
    background = _background(state, labels) if count else data
    if state.unit == "m":                                # local area factor for the true surface area
        gy, gx = np.gradient(np.where(np.isfinite(data), data, np.nanmean(data) if np.isfinite(data).any() else 0),
                             dy, dx)
        area_factor = np.sqrt(1.0 + gx ** 2 + gy ** 2)
    else:
        area_factor = None
    for number, box in enumerate(slices, start=1):
        if box is None:
            continue
        region = labels[box] == number
        z = data[box][region]
        yy, xx = np.nonzero(region)
        xs, ys = (xx + box[1].start + 0.5) * dx, (yy + box[0].start + 0.5) * dy
        area = region.sum() * pixel_area
        perimeter = _perimeter(region, dx, dy)
        cx, cy = xs.mean(), ys.mean()
        cov = np.cov(np.vstack([xs - cx, ys - cy])) if xs.size > 1 else np.zeros((2, 2))
        cov = cov + np.diag([dx * dx, dy * dy]) / 12.0                    # the extent of a pixel itself
        eigen, vectors = np.linalg.eigh(cov)
        major, minor = 4.0 * math.sqrt(max(eigen[1], 0.0)), 4.0 * math.sqrt(max(eigen[0], 0.0))
        angle = math.degrees(math.atan2(-vectors[1, 1], vectors[0, 1])) % 180.0    # from +x, counterclockwise
        # base: the median of the background pixels around the grain, else (surrounded by grains) its minimum
        y0, y1 = max(box[0].start - 1, 0), min(box[0].stop + 1, ny)
        x0, x1 = max(box[1].start - 1, 0), min(box[1].stop + 1, nx)
        local = labels[y0:y1, x0:x1] == number
        ring = ndimage.binary_dilation(local, structure=np.ones((3, 3))) & (labels[y0:y1, x0:x1] == 0)
        around = data[y0:y1, x0:x1]
        outside = around[ring & np.isfinite(around)]
        base = float(np.median(outside)) if outside.size else float(z.min())
        feret_max, feret_max_dir, feret_min, feret_min_dir, hull_area = _feret(
            region, box[1].start * dx, box[0].start * dy, dx, dy)
        under = background[box][region]
        rows.append({
            "Grain": number, "Pixels": int(region.sum()), "Area": area,
            "Equivalent diameter": 2.0 * math.sqrt(area / math.pi), "Perimeter": perimeter,
            "Circularity": min(1.0, 4.0 * math.pi * area / perimeter ** 2) if perimeter > 0 else math.nan,
            "Major axis": major, "Minor axis": minor, "Orientation": angle,
            "Max Feret": feret_max, "Max Feret direction": feret_max_dir,
            "Min Feret": feret_min, "Min Feret direction": feret_min_dir,
            "Feret ratio": feret_max / feret_min if feret_min > 0 else math.nan,
            "Convex hull area": hull_area,
            "Solidity": min(1.0, area / hull_area) if hull_area > 0 else math.nan,
            "Surface area": float(area_factor[box][region].sum()) * pixel_area if area_factor is not None
            else math.nan,
            "Center x": cx, "Center y": cy,
            "Max height": float(z.max()), "Mean height": float(z.mean()), "Median height": float(np.median(z)),
            "Min height": float(z.min()), "RMS height": float(z.std()),
            "Height above base": float(z.max()) - base,
            "Volume above zero": float(z.sum()) * pixel_area,
            "Volume above base": float(np.clip(z - base, 0.0, None).sum()) * pixel_area,
            "Volume above background": float(np.nansum(z - under)) * pixel_area if np.isfinite(under).all()
            else math.nan,
            "Touches edge": bool((region & border[box]).any()),
        })
    units = {name: (unit.replace("z", state.unit) if unit.startswith("z") else unit) for name, unit in COLUMNS}
    units["Volume above zero"] = units["Volume above base"] = units["Volume above background"] = f"{state.unit}·m²"
    total_area = state.size_x * state.size_y
    covered = sum(r["Area"] for r in rows)

    def stat(name):
        values = np.array([r[name] for r in rows], float)
        return (float(values.mean()), float(values.std())) if values.size else (math.nan, math.nan)

    summary = {"Grains": (float(count), ""), "Density": (count / total_area if total_area else math.nan, "1/m²"),
               "Covered area": (100.0 * covered / total_area if total_area else math.nan, "%")}
    for name in ("Area", "Equivalent diameter", "Height above base", "Max height", "Volume above base"):
        mean, std = stat(name)
        summary[f"Mean {name.lower()}"] = (mean, units[name])
        summary[f"Std {name.lower()}"] = (std, units[name])
    return GrainResult(labels, rows, summary, units)


# ---- display units (grain table, report) ----
