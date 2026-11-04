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
