""" Terrace analysis (process.terraces) and curvature (measure.curvature) on synthetic surfaces with known answers. """

import math

import numpy as np
import pytest

from NSFopen.process import measure, ops, terraces


def _state(z, size=2e-6, mask=None):
    return ops.State(z, np.zeros(z.shape, bool) if mask is None else mask, size, size * z.shape[0] / z.shape[1], "m")


def staircase(step=0.31e-9, n=256, noise=0.03e-9, bow=2e-9, seed=1):
    """ Curved steps of one height on a tilted, bowed background, with noise; also the true level of every pixel. """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / n
    level = np.floor(6.5 * xx + 0.8 * np.sin(2 * np.pi * yy * 1.3) + 0.5 * yy)
    z = (level * step + 1.5e-9 * xx - 0.7e-9 * yy + bow * ((xx - 0.4) ** 2 + (yy - 0.6) ** 2)
         + rng.normal(0, noise, (n, n)))
    return z, level - level.min()


def test_step_height_and_levels_of_a_staircase():
    z, level = staircase()
    result = terraces.terraces(_state(z))
    assert result.step == pytest.approx(0.31e-9, rel=0.01)
    assert result.step_error < 0.005e-9
    assert {t.level for t in result.terraces} == set(range(int(level.max()) + 1))
    assert result.rms == pytest.approx(0.03e-9, rel=0.15)                 # only the noise is left
    assert result.periodicity > 0.95
    heights = [t.height for t in result.terraces]
    assert heights == sorted(heights) and heights[0] == 0
    # leveling: every terrace flat, so the leveled image within a terrace has only the noise
    leveled = result.leveled(_state(z))
    biggest = max(result.terraces, key=lambda t: t.pixels).number
    assert np.std(leveled[result.labels == biggest]) < 0.05e-9
    assert "Step height 0.31" in result.summary()


def test_given_step_and_the_operation():
    z, _ = staircase(step=0.5e-9)
    result = terraces.terraces(_state(z), terraces.TerraceSettings(step=0.5e-9))
    assert result.step == 0.5e-9 and "(given)" in result.summary()
    leveled = ops.apply_op(_state(z), "level.terraces", {})
    assert np.allclose(leveled.data, terraces.terraces(_state(z)).leveled(_state(z)))


def test_steps_from_the_mask():
    """ Faint steps: the automatic search finds nothing, the steps drawn into the mask divide the terraces. """
    n = 200
    yy, xx = np.mgrid[0:n, 0:n]
    level = (xx // 50).astype(float)
    rng = np.random.default_rng(3)
    z = level * 0.2e-9 + rng.normal(0, 0.15e-9, (n, n))                # steps hidden in the noise
    mask = np.zeros((n, n), bool)
    for column in (50, 100, 150):
        mask[:, column - 1:column + 1] = True
    result = terraces.terraces(_state(z, mask=mask), terraces.TerraceSettings(edges="mask", degree=1))
    assert len(result.terraces) == 4
    assert result.step == pytest.approx(0.2e-9, rel=0.05)
    with pytest.raises(ops.OpError, match="mask is empty"):
        terraces.terraces(_state(z), terraces.TerraceSettings(edges="mask"))


def test_no_steps_and_two_levels():
    flat = np.random.default_rng(0).normal(0, 0.05e-9, (128, 128))
    result = terraces.terraces(_state(flat))
    assert math.isnan(result.step) and "No common step height" in result.summary()
    z, _ = staircase(n=128)
    z = np.where(np.arange(128) < 64, 0.0, 0.3e-9)[None, :] + np.zeros((128, 128))
    z = z + np.random.default_rng(1).normal(0, 0.01e-9, z.shape)
    result = terraces.terraces(_state(z))
    assert len(result.terraces) == 2 and math.isnan(result.step) and "three or more" in result.message


def test_curvature_of_known_surfaces():
    n, size = 200, 10e-6
    yy, xx = np.mgrid[0:n, 0:n]
    x, y = (xx + 0.5) * size / n, (yy + 0.5) * size / n
    # elliptic dome, radii 20 and 80 µm, the strongly curved direction 30° from x (counterclockwise on the screen)
    a = math.radians(30)
    u = (x - 4e-6) * math.cos(a) - (y - 6e-6) * math.sin(a)
    v = (x - 4e-6) * math.sin(a) + (y - 6e-6) * math.cos(a)
    z = -(u * u / (2 * 20e-6) + v * v / (2 * 80e-6)) + 1e-3 * x
    c = measure.curvature(_state(z, size))
    assert c.radius_1 == pytest.approx(-20e-6, rel=1e-6) and c.radius_2 == pytest.approx(-80e-6, rel=1e-6)
    assert c.angle_1 == pytest.approx(30, abs=0.01) and c.angle_2 == pytest.approx(120, abs=0.01)
    assert c.center_inside and "dome" in c.summary()
    start, end = c.profile(1, size, size)                 # the line along direction 1 runs through the center
    cross = (end[0] - start[0]) * (c.center[1] - start[1]) - (end[1] - start[1]) * (c.center[0] - start[0])
    assert abs(cross) < 1e-15
    # bowl (sphere of 50 µm: the quadratic fit gives the radius within 1 % over a field of a fifth of it)
    r2 = (x - 3e-6) ** 2 + (y - 6e-6) ** 2
    c = measure.curvature(_state(50e-6 - np.sqrt(50e-6 ** 2 - r2), size))
    assert c.radius_1 == pytest.approx(50e-6, rel=0.01) and c.radius_2 == pytest.approx(50e-6, rel=0.01)
    assert c.center == pytest.approx((3e-6, 6e-6), abs=5e-9) and "bowl" in c.summary()
    # cylinder: one direction flat, no single center
    c = measure.curvature(_state((y - 5e-6) ** 2 / (2 * 30e-6), size))
    assert c.radius_1 == pytest.approx(30e-6) and math.isinf(c.radius_2) and c.center is None
    assert "cylinder" in c.summary()
    with pytest.raises(ops.OpError, match="height channel"):
        measure.curvature(ops.State(z, np.zeros(z.shape, bool), size, size, "V"))


def test_curvature_with_the_center_given():
    """ The vertex fixed at a given center: same radii as the free fit when it is the true center; with a tilt, which
    moves the free center, the given apex still gives the radii, and the rms shows the tilt it cannot follow. """
    n, size = 200, 10e-6
    yy, xx = np.mgrid[0:n, 0:n]
    x, y = (xx + 0.5) * size / n, (yy + 0.5) * size / n
    z = ((x - 4e-6) ** 2 + (y - 5e-6) ** 2) / (2 * 40e-6)
    c = measure.curvature(_state(z, size), center=(4e-6, 5e-6))
    assert c.manual_center and "(set by hand)" in c.summary()
    assert c.radius_1 == pytest.approx(40e-6, rel=1e-9) and c.radius_2 == pytest.approx(40e-6, rel=1e-9)
    assert c.center == (4e-6, 5e-6) and c.center_z == pytest.approx(0, abs=1e-15)
    tilted = z + 2e-3 * x
    free = measure.curvature(_state(tilted, size))
    assert free.center[0] == pytest.approx(4e-6 - 2e-3 * 40e-6, abs=1e-9)       # the tilt moves the free center
    fixed = measure.curvature(_state(tilted, size), center=(4e-6, 5e-6))
    assert fixed.radius_2 == pytest.approx(40e-6, rel=1e-6)                    # along y the tilt does nothing
    assert fixed.rms > 1e-9 > free.rms
    # the same coefficients describe the surface: the free-form solution of a fixed fit gives back the center
    a, b, c_, d, e, f = fixed.coefficients
    hessian = np.array([[2 * d, e], [e, 2 * f]])
    offset = np.linalg.solve(hessian, -np.array([b, c_]))
    assert offset + [size / 2, size / 2] == pytest.approx([4e-6, 5e-6], abs=1e-12)


def test_extreme_point():
    """ The highest / lowest point of an off-center dome, through noise; and of the masked part only. """
    n, dx = 100, 1e-7
    y, x = (np.mgrid[0:n, 0:n] + 0.5) * dx
    dome = -((x - 3e-6) ** 2 + (y - 7e-6) ** 2) / 1e-4
    data = dome + 3e-10 * np.random.default_rng(2).standard_normal(dome.shape)
    state = ops.State(data, np.zeros(data.shape, bool), n * dx, n * dx, "m")
    cx, cy = measure.extreme_point(state)
    assert abs(cx - 3e-6) < 3 * dx and abs(cy - 7e-6) < 3 * dx
    cx, cy = measure.extreme_point(ops.State(-data, state.mask, n * dx, n * dx, "m"), which="min")
    assert abs(cx - 3e-6) < 3 * dx and abs(cy - 7e-6) < 3 * dx
    mask = np.zeros(data.shape, bool)
    mask[:, 60:] = True                                  # the top of the dome is outside: the edge of the mask
    cx, _ = measure.extreme_point(ops.State(data, mask, n * dx, n * dx, "m"), "masked")
    assert cx == pytest.approx(60.5 * dx)
