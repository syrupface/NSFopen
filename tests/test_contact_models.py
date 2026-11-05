""" Bottom-effect correction and the Maugis-Dugdale model (NSFopen.process.contact). """

import math

import numpy as np
import pytest

from NSFopen.process import contact


def test_dimitriadis_series():
    """ The published series for nu = 0.5: not bonded 0.884, 0.781, 0.386, 0.0048; bonded 1.133, 1.283, 0.769 and
        -0.0975 (the general formula gives the minus sign of the 4th term). """
    assert contact._dimitriadis(False, 0.5) == pytest.approx((0.884, 0.781, 0.386, 0.0048), abs=6e-4)
    assert contact._dimitriadis(True, 0.5) == pytest.approx((1.133, 1.283, 0.769, -0.0975), abs=6e-4)


def test_bottom_factor_shapes():
    thick = contact.Tip("sphere", 1e-6)
    assert contact.bottom_factor([1e-7], thick)[0] == 1.0
    sphere = contact.Tip("sphere", 1e-6, thickness=1e-6)
    chi = math.sqrt(1e-6 * 1e-7) / 1e-6
    c = contact._dimitriadis(True, 0.5)
    assert contact.bottom_factor([1e-7], sphere)[0] == pytest.approx(1 + sum(c[i] * chi ** (i + 1) for i in range(4)))
    free = contact.Tip("sphere", 1e-6, thickness=1e-6, bonded=False)
    assert 1 < contact.bottom_factor([1e-7], free)[0] < contact.bottom_factor([1e-7], sphere)[0]
    cone = contact.Tip("cone", half_angle=20.0, thickness=1e-6)
    x = 1e-7 * math.tan(math.radians(20)) / 1e-6
    assert contact.bottom_factor([1e-7], cone)[0] == pytest.approx(1 + 0.721 * x + 0.650 * x ** 2 + 0.491 * x ** 3
                                                                   + 0.225 * x ** 4)
    punch = contact.Tip("punch", 1e-7, thickness=1e-6)
    assert contact.bottom_factor([0.0, 1e-7], punch) == pytest.approx([1 + 0.1133 + 0.01283 + 0.000598 - 0.0000291] * 2)
    with pytest.raises(contact.FitError):
        contact.bottom_factor([1e-7], contact.Tip("cone", thickness=1e-6, bonded=False))
    with pytest.raises(contact.FitError):
        contact.bottom_factor([1e-7], contact.Tip("hyperboloid", thickness=1e-6))


def test_bottom_effect_recovers_modulus():
    """ A thin bonded layer indented with a bead: the corrected Hertz fit gets the modulus back, the plain one reads it
        too high. """
    radius, e, thickness = 2.5e-6, 5e3, 2e-6
    thin = contact.Tip("sphere", radius, thickness=thickness)
    delta = np.linspace(-200e-9, 600e-9, 200)
    force = e / 0.75 * contact.unit_force(delta - 50e-9, thin)
    corrected = contact.fit_elastic(delta, force, thin, "hertz")
    plain = contact.fit_elastic(delta, force, contact.Tip("sphere", radius), "hertz")
    assert corrected.modulus == pytest.approx(e, rel=1e-3)
    assert corrected.contact_point == pytest.approx(50e-9, abs=1e-9)
    assert plain.modulus > 1.3 * e
    values = corrected.values()
    assert values["sample thickness"][0] == thickness and values["bottom factor"][0] > 1.3
    with pytest.raises(contact.FitError):
        contact.fit_jkr(delta, force, thin)


def test_maugis_limits():
    """ Small lambda: DMT (P = A³ - 2, D = A²); large lambda: the JKR curve; the pull-off goes from 2 pi W R to
        3/2 pi W R. """
    area = np.array([1.0, 2.0, 4.0])
    p, d = contact.maugis_reduced(0.001, area)       # (D approaches A² slowly, about as lambda^(1/3))
    assert p == pytest.approx(area ** 3 - 2.0, abs=0.06) and d == pytest.approx(area ** 2, abs=0.06)
    w = 0.01
    d, f = contact._md_curve(w, 50.0, 0.05)
    dj, fj = contact._jkr_curve(w, 1.0, 0.05)
    xs = np.linspace(max(d.min(), dj.min()) + 0.003, 0.05, 7)
    assert np.interp(xs, d, f) == pytest.approx(np.interp(xs, dj, fj), rel=0.01, abs=2e-5)
    assert -f.min() == pytest.approx(1.5 * math.pi * w, rel=1e-3)
    assert -contact._md_curve(w, 0.005, 0.05)[1].min() == pytest.approx(2.0 * math.pi * w, rel=0.01)


@pytest.mark.parametrize("lam", [0.15, 0.8, 4.0])
def test_maugis_fit(lam):
    radius, e, nu, work = 20e-9, 1e9, 0.5, 0.05
    e_star = e / (1 - nu * nu)
    tip = contact.Tip("sphere", radius, poisson=nu)
    d, _ = contact._md_curve(work / (e_star * radius), lam, 0.6)
    delta = np.linspace(d.min() * radius, 0.5 * radius, 300)[::-1]
    force = contact.maugis_force(delta, e_star, work, lam, radius)
    low = int(np.argmin(force))
    delta, force = delta[:low + 30] + 3e-9, force[:low + 30]
    force = force + np.random.default_rng(0).normal(0, 1e-11, force.size)
    result = contact.fit_maugis(delta, force, tip)
    assert result.modulus == pytest.approx(e, rel=0.01)
    assert result.work_of_adhesion == pytest.approx(work, rel=0.01)
    assert result.contact_point == pytest.approx(3e-9, abs=0.2e-9)
    if lam < 2:
        assert result.extra["Maugis λ"] == pytest.approx(lam, rel=0.05)
        assert result.values()["Tabor μ"][0] == pytest.approx(1.157 * lam, rel=0.05)
    else:                          # the JKR side: the curve hardly changes with lambda any more
        assert result.extra["Maugis λ"] > 2.5
