""" 3D view of an image as a matplotlib figure, after the 3D view of the Nanosurf Surface Analyzer. There the surface is
drawn by OpenGL shaders; here the same mesh, camera and lighting are drawn by a small rasterizer in numpy, so it runs
in scripts without Qt or a graphics card. Matplotlib only adds the picture, the axes and the color bar.

Scene coordinates: the image lies in the x-y plane, centered on the origin, its longer side 1 long; y points to the top
of the image (row 0 of the array passed in at +y). Heights are normalized so that the full height range is 1 at value
scale 1 and multiplied by the value scale (z_scale) when drawn. """

import dataclasses
import math
import warnings
from typing import Optional

import numpy as np

from . import ops
from . import colormaps

# ---- materials (ambient, diffuse, specular RGB, shininess), after the classic OpenGL material table ----

MATERIALS = {
    "Gray": ((0.20, 0.20, 0.20), (0.75, 0.75, 0.75), (0.35, 0.35, 0.35), 32.0),
    "Matte white": ((0.25, 0.25, 0.25), (0.85, 0.85, 0.85), (0.05, 0.05, 0.05), 8.0),
    "Brass": ((0.329, 0.223, 0.027), (0.780, 0.569, 0.114), (0.992, 0.941, 0.808), 27.9),
    "Bronze": ((0.213, 0.128, 0.054), (0.714, 0.428, 0.181), (0.394, 0.272, 0.167), 25.6),
    "Chrome": ((0.25, 0.25, 0.25), (0.40, 0.40, 0.40), (0.775, 0.775, 0.775), 76.8),
    "Copper": ((0.191, 0.074, 0.023), (0.704, 0.270, 0.083), (0.257, 0.138, 0.086), 12.8),
    "Gold": ((0.247, 0.200, 0.075), (0.752, 0.606, 0.226), (0.628, 0.556, 0.366), 51.2),
    "Silver": ((0.192, 0.192, 0.192), (0.508, 0.508, 0.508), (0.508, 0.508, 0.508), 51.2),
    "Pewter": ((0.106, 0.059, 0.114), (0.427, 0.471, 0.541), (0.333, 0.333, 0.522), 9.8),
    "Pearl": ((0.25, 0.207, 0.207), (1.0, 0.829, 0.829), (0.297, 0.297, 0.297), 11.3),
    "Jade": ((0.135, 0.223, 0.158), (0.54, 0.89, 0.63), (0.316, 0.316, 0.316), 12.8),
    "Ruby": ((0.175, 0.012, 0.012), (0.614, 0.041, 0.041), (0.728, 0.627, 0.627), 76.8),
    "Turquoise": ((0.1, 0.187, 0.175), (0.396, 0.742, 0.691), (0.297, 0.308, 0.307), 12.8),
    "Obsidian": ((0.054, 0.05, 0.066), (0.183, 0.17, 0.225), (0.333, 0.329, 0.346), 38.4),
}

MODES = ("gradient", "overlay", "lighting", "overlay_light")
MASK_COLOR = (1.0, 0.25, 0.25)
MASK_ALPHA = 0.55
OVERSAMPLING = 2                                    # pixels drawn per pixel of the picture, along both axes
CHUNK = 4_000_000                                   # candidate pixels handled at once by the rasterizer


def colormap(cmap):
    """ A matplotlib color map from a matplotlib name, a Nanosurf map of colormaps.py, or a color map itself. """
    return colormaps.colormap(cmap)


def material_of(material) -> tuple:
    """ (ambient, diffuse, specular, shininess) of a name in MATERIALS (any case), or the tuple itself. """
    if not isinstance(material, str):
        return material
    for name, value in MATERIALS.items():
        if name.lower() == material.lower():
            return value
    raise ValueError(f"Unknown material '{material}': use one of {', '.join(MATERIALS)}.")


# ---- mesh ----

@dataclasses.dataclass
class Mesh:
    vertices: np.ndarray            # (n, 7) float32: x, y, z (normalized), gx, gy (dz/dx, dz/dy normalized), color
                                    # value 0..1, mask 0/1
    triangles: np.ndarray           # (m, 3) uint32
    size: tuple                     # (width, height) of the image in scene units (the longer side 1)
    z_range: tuple                  # (min, max) of the heights in the unit of the data
    color_range: tuple              # (min, max) of the color values in their unit
    shape: tuple                    # (rows, columns) of the grid
    step: int                       # pixels per grid point


def _resampled(data: np.ndarray, step: int) -> np.ndarray:
    """ Every step-th pixel along both axes, averaged over step x step blocks (NaN ignored). """
    if step <= 1:
        return data.astype(float)
    ny, nx = data.shape
    ny2, nx2 = ny // step, nx // step
    blocks = data[:ny2 * step, :nx2 * step].reshape(ny2, step, nx2, step)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(blocks, axis=(1, 3))


def _range(values: np.ndarray, mode: str = "full") -> tuple:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.0, 1.0
    low, high = (np.percentile(finite, (1, 99)) if mode == "p1" else (finite.min(), finite.max()))
    return (float(low), float(high)) if high > low else (float(low) - 0.5, float(low) + 0.5)


def build_mesh(state: ops.State, color: Optional[ops.State] = None, max_points: int = 512, color_range="full") -> Mesh:
    """ The triangle mesh of an image: at most max_points grid points per side (blocks of pixels averaged). Color
        values come from `color` (another channel of the same size, for the overlay modes) or from the heights; they
        are normalized to 0..1 over color_range: (low, high), 'full' or 'p1' (1-99 %). Triangles touching NaN pixels
        are left out. """
    data = state.data
    ny, nx = data.shape
    step = max(1, int(math.ceil(max(ny, nx) / max(2, int(max_points)))))
    z = _resampled(data, step)
    gy, gx = z.shape
    if gy < 2 or gx < 2:
        raise ops.OpError("The image is too small for a 3D view.")
    z_low, z_high = _range(z)
    span = z_high - z_low
    width, height = state.size_x, state.size_y
    longest = max(width, height)
    size = (width / longest, height / longest)
    x = (np.arange(gx) + 0.5) / gx * size[0] - size[0] / 2
    y = size[1] / 2 - (np.arange(gy) + 0.5) / gy * size[1]          # row 0 at the top (+y)
    xx, yy = np.meshgrid(x, y)
    zn = (z - z_low) / span - 0.5                                   # -0.5 .. 0.5 at value scale 1
    filled = np.where(np.isfinite(zn), zn, 0.0)
    pitch_x, pitch_y = size[0] / gx, size[1] / gy
    dzdy_rows, dzdx = np.gradient(filled, pitch_y, pitch_x)
    dzdy = -dzdy_rows                                               # rows go down, y goes up
    source = color if color is not None else state
    if source.data.shape != data.shape:
        raise ops.OpError("The color channel has a different size (was it cropped?).")
    values = _resampled(source.data, step)
    c_low, c_high = _range(values, color_range) if isinstance(color_range, str) else color_range
    t = np.clip((values - c_low) / (c_high - c_low if c_high > c_low else 1.0), 0.0, 1.0)
    mask = _resampled(state.mask.astype(float), step) > 0.5
    vertices = np.stack([xx, yy, filled, dzdx, dzdy, np.nan_to_num(t), mask.astype(float)], axis=-1)
    vertices = vertices.reshape(-1, 7).astype(np.float32)
    index = np.arange(gy * gx, dtype=np.uint32).reshape(gy, gx)
    a, b = index[:-1, :-1].ravel(), index[:-1, 1:].ravel()
    c, d = index[1:, :-1].ravel(), index[1:, 1:].ravel()
    triangles = np.stack([np.stack([a, c, b], 1), np.stack([b, c, d], 1)], 1).reshape(-1, 3)
    bad = ~np.isfinite(zn).ravel()
    if bad.any():
        triangles = triangles[~bad[triangles].any(axis=1)]
    return Mesh(vertices, triangles, size, (z_low, z_high), (c_low, c_high), (gy, gx), step)


def default_z_scale(state: ops.State, mesh: Mesh) -> float:
    """ The value scale at which the height range has its true proportion, but stays visible: AFM heights are tiny,
        so the true scale would be flat. """
    low, high = mesh.z_range
    true_scale = (high - low) / max(state.size_x, state.size_y) if state.unit == "m" else 0.2
    return float(min(max(true_scale, 0.15), 0.6))


# ---- camera ----

@dataclasses.dataclass
class Camera:
    phi: float = -30.0              # rotation around the vertical axis, degrees
    theta: float = 55.0             # tilt from the vertical (0: looking straight down), degrees
    z_scale: float = 0.25           # value scale: height of the full data range relative to the longer image side
    perspective: bool = True

    def view(self) -> np.ndarray:
        """ 4 x 4 view matrix (scene to camera; column vectors). """
        rz, rx = np.radians(self.phi), np.radians(-self.theta)
        cz, sz, cx, sx = math.cos(rz), math.sin(rz), math.cos(rx), math.sin(rx)
        spin = np.array([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        tilt = np.array([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]])
        move = np.eye(4)
        move[2, 3] = -2.6
        return move @ tilt @ spin

    def projection(self) -> np.ndarray:
        near, far = 0.1, 20.0
        if self.perspective:
            f = 1.0 / math.tan(math.radians(30.0) / 2)
            return np.array([[f, 0, 0, 0], [0, f, 0, 0],
                             [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)], [0, 0, -1, 0]])
        h = 0.72
        return np.array([[1 / h, 0, 0, 0], [0, 1 / h, 0, 0], [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                         [0, 0, 0, 1]])

    def model(self) -> np.ndarray:
        return np.diag([1.0, 1.0, self.z_scale, 1.0])

    def clip(self, points: np.ndarray) -> np.ndarray:
        """ (x, y, distance from the camera plane) of scene points (n, 3): x, y in -1..1 of the view, y up. """
        homogeneous = np.c_[points, np.ones(len(points))]
        camera = homogeneous @ (self.view() @ self.model()).T
        projected = camera @ self.projection().T
        distance = -camera[:, 2]
        with np.errstate(all="ignore"):
            return np.c_[projected[:, :2] / projected[:, 3:4], distance]


def light_direction(phi: float, theta: float) -> np.ndarray:
    """ Unit vector towards the light in scene coordinates: phi around the vertical, theta above the horizontal. """
    p, t = math.radians(phi), math.radians(theta)
    return np.array([math.cos(t) * math.cos(p), math.cos(t) * math.sin(p), math.sin(t)])


def nice_ticks(low: float, high: float, count: int = 5) -> list:
    """ Round tick values inside [low, high]. """
    if not high > low:
        return [low]
    raw = (high - low) / max(1, count)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    first = math.ceil(low / step - 1e-9) * step
    ticks, value = [], first
    while value <= high + 1e-9 * step:
        ticks.append(0.0 if abs(value) < 1e-12 * step else value)
        value += step
    return ticks


def display_unit(unit: str, values: np.ndarray) -> tuple:
    """ (scale, label) that shows the values of an SI unit with a readable number: m -> nm / µm, V -> mV, ... """
    if unit not in ("m", "V", "A", "N", "s", "Hz"):
        return 1.0, unit
    finite = values[np.isfinite(values)] if values.size else values
    span = float(np.percentile(np.abs(finite), 99)) if finite.size else 0.0
    if span == 0.0:
        return 1.0, unit
    exponent = int(np.clip(np.floor(np.log10(span) / 3.0) * 3, -12, 6 if unit == "Hz" else 0))
    prefix = {-12: "p", -9: "n", -6: "µ", -3: "m", 0: "", 3: "k", 6: "M"}[exponent]
    return 10.0 ** -exponent, prefix + unit


# ---- drawing the surface ----

def rasterize(screen: np.ndarray, depth: np.ndarray, triangles: np.ndarray, width: int, height: int) -> tuple:
    """ Z-buffer drawing of triangles, all at once in chunks of candidate pixels instead of one by one. screen: (n, 2)
        pixel positions of the vertices, y down. Returns the triangle of every pixel (-1: none) and its first two
        barycentric coordinates. They are linear on the screen, which is close enough for triangles of a few pixels. """
    triangle = np.full((height, width), -1, np.int32)
    first, second = np.zeros((height, width), np.float32), np.zeros((height, width), np.float32)
    nearest = np.full((height, width), np.inf)
    xs, ys, zs = screen[triangles, 0], screen[triangles, 1], depth[triangles]
    with np.errstate(invalid="ignore"):
        denominator = (ys[:, 1] - ys[:, 2]) * (xs[:, 0] - xs[:, 2]) + (xs[:, 2] - xs[:, 1]) * (ys[:, 0] - ys[:, 2])
        usable = np.isfinite(denominator) & (denominator != 0) & (zs.min(axis=1) > 0.1)       # not behind the camera
        x0 = np.ceil(np.nan_to_num(xs.min(axis=1)) - 0.5).clip(0, width).astype(np.int64)
        x1 = (np.floor(np.nan_to_num(xs.max(axis=1)) - 0.5) + 1).clip(0, width).astype(np.int64)
        y0 = np.ceil(np.nan_to_num(ys.min(axis=1)) - 0.5).clip(0, height).astype(np.int64)
        y1 = (np.floor(np.nan_to_num(ys.max(axis=1)) - 0.5) + 1).clip(0, height).astype(np.int64)
    usable &= (x1 > x0) & (y1 > y0)
    numbers = np.flatnonzero(usable)
    if not numbers.size:
        return triangle, first, second
    columns, rows = (x1 - x0)[numbers], (y1 - y0)[numbers]
    area = columns * rows
    chunk_of = np.cumsum(area) // CHUNK
    for chunk in np.unique(chunk_of):
        part = numbers[chunk_of == chunk]
        part_columns, part_area = columns[chunk_of == chunk], area[chunk_of == chunk]
        which = np.repeat(np.arange(part.size), part_area)
        local = np.arange(which.size) - np.repeat(np.cumsum(part_area) - part_area, part_area)
        t = part[which]
        px = x0[t] + local % part_columns[which]
        py = y0[t] + local // part_columns[which]
        qx, qy = px + 0.5, py + 0.5
        tx, ty, tz, d = xs[t], ys[t], zs[t], denominator[t]
        a = ((ty[:, 1] - ty[:, 2]) * (qx - tx[:, 2]) + (tx[:, 2] - tx[:, 1]) * (qy - ty[:, 2])) / d
        b = ((ty[:, 2] - ty[:, 0]) * (qx - tx[:, 2]) + (tx[:, 0] - tx[:, 2]) * (qy - ty[:, 2])) / d
        c = 1.0 - a - b
        z = a * tz[:, 0] + b * tz[:, 1] + c * tz[:, 2]
        keep = (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6)
        keep[keep] = z[keep] < nearest[py[keep], px[keep]]
        order = np.flatnonzero(keep)
        order = order[np.argsort(-z[order], kind="stable")]         # the nearest is written last and stays
        where = py[order], px[order]
        nearest[where] = z[order]
        triangle[where] = t[order]
        first[where], second[where] = a[order], b[order]
    return triangle, first, second


def shade(mesh: Mesh, camera: Camera, triangle: np.ndarray, first: np.ndarray, second: np.ndarray, mode: str, cmap,
          material: tuple, light: tuple, show_mask: bool, background: tuple) -> np.ndarray:
    """ RGB picture from what rasterize() found: the lighting of the Analyzer's fragment shader, for every pixel. """
    picture = np.empty(triangle.shape + (3,), np.float32)
    picture[:] = background
    hit = triangle >= 0
    if not hit.any():
        return picture
    corners = mesh.triangles[triangle[hit]]
    a, b = first[hit, None], second[hit, None]
    c = 1.0 - a - b

    def attribute(column):
        v = mesh.vertices[corners, column]
        return (a[:, 0] * v[:, 0] + b[:, 0] * v[:, 1] + c[:, 0] * v[:, 2]).astype(np.float32)

    base = cmap(attribute(5))[:, :3].astype(np.float32)
    if mode == "gradient" or mode == "overlay":
        rgb = base
    else:
        zs = camera.z_scale
        normal = np.stack([-zs * attribute(3), -zs * attribute(4), np.ones(base.shape[0], np.float32)], axis=1)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        position = np.stack([attribute(0), attribute(1), attribute(2) * zs], axis=1)
        eye = np.linalg.inv(camera.view())[:3, 3].astype(np.float32)
        to_eye = eye - position
        to_eye /= np.linalg.norm(to_eye, axis=1, keepdims=True)
        normal *= np.where((normal * to_eye).sum(axis=1, keepdims=True) < 0, -1.0, 1.0)         # seen from below
        towards = light_direction(*light).astype(np.float32)
        diffuse = np.clip(normal @ towards, 0.0, None)[:, None]
        reflected = 2.0 * diffuse * normal - towards
        specular = np.where(diffuse > 0, np.clip((reflected * to_eye).sum(axis=1, keepdims=True), 0.0, None), 0.0)
        ka, kd, ks, shininess = material
        specular = specular ** shininess
        if mode == "lighting":
            rgb = np.array(ka, np.float32) + np.array(kd, np.float32) * diffuse + np.array(ks, np.float32) * specular
        else:
            rgb = base * (0.35 + 0.75 * diffuse) + np.array(ks, np.float32) * specular * 0.6
    if show_mask:
        masked = attribute(6) > 0.5
        rgb[masked] = rgb[masked] * (1 - MASK_ALPHA) + np.array(MASK_COLOR, np.float32) * MASK_ALPHA
    picture[hit] = np.clip(rgb, 0.0, 1.0)
    return picture


def _screen_fit(camera: Camera, mesh: Mesh, width: int, height: Optional[int], margins: tuple, zoom: float) -> tuple:
    """ (function from scene points to pixels with y down, figure height in pixels): the box around the surface fills
        the figure without the margins (left, right, top, bottom). Without a height, the figure is as high as needed. """
    w2, h2 = mesh.size[0] / 2, mesh.size[1] / 2
    box = np.array([[x, y, z] for x in (-w2, w2) for y in (-h2, h2) for z in (-0.5, 0.5)])
    corners = camera.clip(box)[:, :2]
    low, high = corners.min(axis=0), corners.max(axis=0)
    left, right, top, bottom = margins
    span_x, span_y = max(high[0] - low[0], 1e-6), max(high[1] - low[1], 1e-6)
    pixels = (width - left - right) / span_x
    if height is None:
        height = int(round(max(span_y * pixels + top + bottom, 0.4 * width)))
    pixels = min(pixels, (height - top - bottom) / span_y)
    center_x = left + (width - left - right) / 2
    center_y = top + (height - top - bottom) / 2
    middle = (low + high) / 2

    def to_screen(points):
        clip = camera.clip(np.asarray(points, float))
        x = center_x + (clip[:, 0] - middle[0]) * pixels * zoom
        y = center_y - (clip[:, 1] - middle[1]) * pixels * zoom
        return np.c_[x, y], clip[:, 2]

    return to_screen, height


def _draw_axes(ax, to_screen, state: ops.State, mesh: Mesh, color, k: float, fontsize: float, line_width: float,
               labels: bool, dpi: float):
    """ x and y axes along the two bottom edges that meet at the corner nearest to the viewer (lowest on the screen),
        the z axis at the far end of the one of them that is further left on the screen. As in image(), x and y grow
        from the lower left corner of the picture. """
    w2, h2 = mesh.size[0] / 2, mesh.size[1] / 2
    bottom = -0.5
    corners = np.array([[-w2, -h2, bottom], [w2, -h2, bottom], [w2, h2, bottom], [-w2, h2, bottom]])
    screen = to_screen(corners)[0]
    if not np.isfinite(screen).all():
        return
    near = int(np.argmax(screen[:, 1]))
    x_end = {0: 1, 1: 0, 2: 3, 3: 2}[near]                 # the neighbor along x
    y_end = {0: 3, 3: 0, 1: 2, 2: 1}[near]                 # the neighbor along y
    scale_l, label_l = display_unit("m", np.array([max(state.size_x, state.size_y)]))
    center = to_screen([[0, 0, bottom]])[0][0]
    z_index = x_end if screen[x_end, 0] < screen[y_end, 0] else y_end
    avoid = screen[z_index]                                # tick labels next to the z axis are left out

    def line(a, b):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=line_width, solid_capstyle="round", clip_on=False)

    def text(point, string, align="center"):
        ax.text(point[0], point[1], string, color=color, fontsize=fontsize, ha=align, va="center", clip_on=False)

    def outward(point):
        away = point - center
        return away / (np.hypot(*away) or 1.0)

    def axis(start_index, end_index, physical, label, skip_origin=False):
        """ An edge from corner start to corner end; tick values grow from the image origin side. """
        a, b = corners[start_index], corners[end_index]
        sa, sb = screen[start_index], screen[end_index]
        line(sa, sb)
        length_px = float(np.hypot(*(sb - sa)))
        axis_index = 0 if a[1] == b[1] else 1
        origin_is_a = (a[0] < b[0]) if axis_index == 0 else (a[1] < b[1])
        start, end = (a, b) if origin_is_a else (b, a)
        count = int(np.clip(length_px / (70 * k), 2, 6))
        total = physical * scale_l
        middle = (sa + sb) / 2
        title_at = middle + 50 * k * outward(middle)
        for tick in nice_ticks(0.0, total, count):
            p = to_screen([start + (end - start) * (tick / total)])[0][0]
            o = outward(p)
            line(p, p + 5 * k * o)
            at_corner = np.hypot(*(p - screen[near])) < 26 * k        # the corner shared with the other axis
            if labels and not (skip_origin and at_corner) and np.hypot(*(p - avoid)) > 22 * k \
                    and np.hypot(*(p + 16 * k * o - title_at)) > 24 * k:
                text(p + 16 * k * o, f"{tick:g}")
        if labels:
            text(title_at, label)

    axis(near, x_end, state.size_x, f"x [{label_l}]")
    axis(near, y_end, state.size_y, f"y [{label_l}]", skip_origin=True)     # one label at the shared corner
    z_corner = corners[z_index]
    low, high = mesh.z_range
    scale_z, label_z = display_unit(state.unit, np.array([low, high]))
    bottom_z, top_z = to_screen([[z_corner[0], z_corner[1], -0.5], [z_corner[0], z_corner[1], 0.5]])[0]
    if np.hypot(*(top_z - bottom_z)) < 2 * fontsize / 72 * dpi:             # seen from above: no room for a z axis
        return
    line(bottom_z, top_z)
    count = int(np.clip(np.hypot(*(top_z - bottom_z)) / (28 * k), 2, 6))    # at least two ticks
    ticks = nice_ticks(low * scale_z, high * scale_z, count)
    if len(ticks) < 2:
        ticks = nice_ticks(low * scale_z, high * scale_z, count * 2)
    if labels and len(ticks) > 2 and np.hypot(*(top_z - bottom_z)) / (len(ticks) - 1) < 1.5 * fontsize / 72 * dpi:
        ticks = [ticks[0], ticks[-1]]                       # a short axis: their labels would touch
    for tick in ticks:
        fraction = (tick / scale_z - low) / (high - low) - 0.5
        p = to_screen([[z_corner[0], z_corner[1], fraction]])[0][0]
        line(p, (p[0] - 5 * k, p[1]))
        if labels:
            text(p + np.array([-8 * k, 0]), f"{tick:g}", "right")
    if labels:
        text(top_z + np.array([0, -14 * k]), f"z [{label_z}]")


def _draw_colorbar(ax, mesh: Mesh, cmap, unit: str, color, width: int, height: int, k: float, fontsize: float):
    bar_width, top, bottom = 16 * k, 42 * k, height - 42 * k
    x = width - 74 * k
    bar = ax.inset_axes([x / width, 1 - bottom / height, bar_width / width, (bottom - top) / height])
    low, high = mesh.color_range
    scale, label = display_unit(unit, np.array([low, high]))
    bar.imshow(cmap(np.linspace(1.0, 0.0, 256))[:, None, :3], aspect="auto", extent=(0, 1, low * scale, high * scale))
    bar.set_xticks([])
    bar.yaxis.tick_right()
    bar.tick_params(axis="y", labelsize=fontsize, color=color, labelcolor=color)
    bar.set_title(f"[{label}]", fontsize=fontsize, color=color)
    for spine in bar.spines.values():
        spine.set_edgecolor(color)


def figure(state: ops.State, color: Optional[ops.State] = None, mode: Optional[str] = None, cmap="afmhot",
           color_unit: Optional[str] = None, color_range="full", material=None, light=(45.0, 45.0), rotation=-30.0,
           tilt=55.0, z_scale=None, zoom=1.0, perspective=True, axes=True, labels=True, colorbar=True, show_mask=False,
           max_points=512, theme="light", fontsize=10, line_width=1.0, figsize=None, dpi=300, ax=None):
    """ The 3D view of a state as a matplotlib figure, or on the axes ax; see process.plot3d. """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    if mode is None:
        mode = "overlay" if color is not None else "lighting" if material is not None else "gradient"
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}': use one of {', '.join(MODES)}.")
    if mode.startswith("overlay") and color is None:
        raise ValueError(f"Mode '{mode}' takes its colors from another channel: pass color=...")
    if theme not in ("light", "dark"):
        raise ValueError("The theme must be 'light' or 'dark'.")
    surface_material = material_of(material if material is not None else "Gray")
    colors_of = colormap(cmap)
    mesh = build_mesh(state, color, max_points, color_range)
    if isinstance(z_scale, str):
        if z_scale != "true" or state.unit != "m":
            raise ValueError("z_scale='true' needs heights in m; otherwise give a number.")
        z_scale = max((mesh.z_range[1] - mesh.z_range[0]) / max(state.size_x, state.size_y), 1e-6)
    camera = Camera(rotation, tilt, default_z_scale(state, mesh) if z_scale is None else float(z_scale), perspective)

    background, foreground = (to_rgb("w"), "#202020") if theme == "light" else (to_rgb("k"), "#e6e6e6")
    show_bar = colorbar and mode != "lighting"
    if ax is not None:
        if ax.figure.get_layout_engine() is not None:      # e.g. layout='constrained': the place of the axes
            ax.figure.draw_without_rendering()
        box = ax.get_position()
        figure_size = ax.figure.get_size_inches()
        figsize = (box.width * figure_size[0], box.height * figure_size[1])
    width = int(round((figsize[0] if figsize else 6.0) * dpi))
    height = int(round(figsize[1] * dpi)) if figsize else None
    k = dpi / 96 * fontsize / 10                            # 1 pixel of the Analyzer's screen at this font size
    room = (28 if ax is not None else 40) * k if (axes or labels) else 10 * k
    margins = (room * 1.6, room * 1.2 + (100 * k if show_bar else 0), room * 1.1, room * 1.6)
    to_screen, height = _screen_fit(camera, mesh, width, height, margins, zoom)

    screen, depth = to_screen(mesh.vertices[:, :3])
    triangle, first, second = rasterize(screen * OVERSAMPLING, depth, mesh.triangles, width * OVERSAMPLING,
                                        height * OVERSAMPLING)
    large = shade(mesh, camera, triangle, first, second, mode, colors_of, surface_material, light, show_mask, background)
    picture = large.reshape(height, OVERSAMPLING, width, OVERSAMPLING, 3).mean(axis=(1, 3))

    if ax is None:
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.patch.set_facecolor(background)
        ax = fig.add_axes([0, 0, 1, 1])
    fig = ax.figure
    ax.imshow(picture, extent=(0, width, height, 0), interpolation="nearest", aspect="equal")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    if axes:
        _draw_axes(ax, to_screen, state, mesh, foreground, k, fontsize, line_width, labels, dpi)
    if show_bar:
        unit = color_unit if color is not None and color_unit is not None else state.unit
        _draw_colorbar(ax, mesh, colors_of, unit, foreground, width, height, k, fontsize)
    return fig
