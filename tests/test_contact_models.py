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


def _example_spec():
    import pathlib
    import NSFopen
    from NSFopen.process import spectroscopy as sp
    return sp.SpecFile(pathlib.Path(NSFopen.__file__).parent / "example" / "Force_Spectroscopy" / "spectroscopy.nid")


def test_vectorized_grid_gives_the_same_fit():
    """ The contact point grid of Hertz, Sneddon and DMT computed at once gives the same results as point by point. """
    from NSFopen.process import spectroscopy as sp
    spec = _example_spec()
    original = contact._search
    try:
        for model, shape in (("hertz", "sphere"), ("sneddon", "hyperboloid"), ("dmt", "cone")):
            analysis = sp.ForceAnalysis(spec, sp.ForceSettings(model=model, shape=shape))
            fast = analysis.run(range(12), workers=1).per_curve("E")[:12]
            contact._search = lambda cost, low, high, grid=60, costs_of=None: original(cost, low, high, grid)
            slow = analysis.run(range(12), workers=1).per_curve("E")[:12]
            contact._search = original
            assert np.array_equal(fast, slow, equal_nan=True)
    finally:
        contact._search = original


def test_several_processes_give_the_same_results():
    from NSFopen.process import spectroscopy as sp
    spec = _example_spec()
    analysis = sp.ForceAnalysis(spec, sp.ForceSettings(model="jkr"))
    curves = list(range(24))
    one = analysis.run(curves, workers=1)
    seen = []
    two = analysis.run(curves, workers=2, progress=lambda done, total: seen.append((done, total)) is None)
    assert [row["curve"] for row in two.rows] == [n + 1 for n in curves]
    assert np.array_equal(one.per_curve("E"), two.per_curve("E"), equal_nan=True)
    assert seen and seen[-1] == (24, 24)
    stopped = analysis.run(curves, workers=2, progress=lambda done, total: done < 6)
    assert 6 <= len(stopped.rows) < 24
    assert sp.auto_workers(0.5) == 1


def test_subset_carries_only_its_curves():
    """ The processes get the curves of their chunk (SpecFile.subset), not the file: the same fits, a small copy. """
    import pickle
    from NSFopen.process import spectroscopy as sp
    spec = _example_spec()
    analysis = sp.ForceAnalysis(spec, sp.ForceSettings(model="hertz"))
    chunk = [10, 11, 12]
    sub = spec.subset(chunk, analysis.segment_names(), analysis.channels())
    sub = pickle.loads(pickle.dumps(sub))
    assert sub.n_curves == spec.n_curves and sub.grid == spec.grid and sub._reader is None
    part = sp.ForceAnalysis(sub, sp.ForceSettings(model="hertz"))
    part._sign = analysis.curve(0, fit=False) and analysis._sign
    for n in chunk:
        assert part.curve(n).values["E"] == analysis.curve(n).values["E"]
    assert len(pickle.dumps(sub)) < 0.2 * sum(spec.data(s, c, n).nbytes for s in analysis.segment_names()
                                              for c in analysis.channels() for n in range(spec.n_curves))


def test_jkr_closed_form_and_fit():
    """ The closed-form JKR force equals the parametric curve, its derivatives are right, and a fit of a synthetic
        JKR retract curve gets E, W and the contact point back. """
    w = 0.02
    d, f = contact._jkr_curve(w, 1.0, 2.0)
    assert contact._jkr_unit(d, w) == pytest.approx(f, rel=1e-9, abs=1e-12)
    h = np.linspace(-0.3, 2.0, 40)
    force, slope, by_w = contact._jkr_unit(h, w, True)
    step = 1e-7
    assert slope[force != 0] == pytest.approx(((contact._jkr_unit(h + step, w) - contact._jkr_unit(h - step, w))
                                              / (2 * step))[force != 0], rel=1e-5)
    assert by_w[force != 0] == pytest.approx(((contact._jkr_unit(h, w * 10 ** step)
                                              - contact._jkr_unit(h, w * 10 ** -step)) / (2 * step))[force != 0],
                                             rel=1e-5)
    radius, e, work = 20e-9, 1e9, 0.05
    e_star = e / 0.75
    delta = np.linspace(-6e-9, 10e-9, 250)[::-1]
    model = contact.jkr_force(delta - 2e-9, e_star, work, radius)
    low = int(np.argmin(model))
    result = contact.fit_jkr(delta[:low + 20], model[:low + 20], contact.Tip("sphere", radius))
    assert result.modulus == pytest.approx(e, rel=1e-3)
    assert result.work_of_adhesion == pytest.approx(work, rel=1e-3)
    assert result.contact_point == pytest.approx(2e-9, abs=1e-11)
