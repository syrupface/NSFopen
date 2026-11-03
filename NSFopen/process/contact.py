# -*- coding: utf-8 -*-
# Part of NSFopen.process; the Nanosurf Surface Analyzer (Spectroscopy page) uses this module.
""" Contact mechanics of force curves: the elastic force of a tip shape pressed into a flat sample, and the fits of the
models Hertz, Sneddon, DMT, JKR and Oliver-Pharr. No GUI code; all values in SI units.

Coordinates. A force curve is given as the sample position delta = z - F / k (the piezo displacement towards the sample
minus the bending of the cantilever, i.e. where the tip apex is relative to the undeformed sample), and the force F
(repulsive > 0). The indentation is h = delta - delta0, with the contact point delta0 found by the fit.

Elastic force. F = E* g(h) with the reduced modulus E* = E / (1 - nu^2) for a rigid tip; g depends only on the shape:

    sphere (Hertz)        g = 4/3 sqrt(R) h^1.5                       paraboloid approximation of a sphere, h << R
    sphere (Sneddon)      h = a/2 ln((R+a)/(R-a)), g = 1/2 [(a^2+R^2) ln((R+a)/(R-a)) - 2 a R]     exact sphere
    cone                  g = 2/pi tan(alpha) h^2                      alpha = half opening angle (Sneddon 1965)
    pyramid (4 sides)     g = tan(alpha) / sqrt(2) h^2                 alpha = angle between axis and face (Bilodeau
                                                                       1992 in the form used by Rico et al. 2005)
    cylinder punch        g = 2 R h                                    flat end of radius R (Sneddon 1965)
    hyperboloid           apex radius R, asymptotic half angle alpha (Akhremitchev and Walker 1999)

Models.
    hertz          sphere, elastic, no adhesion (paraboloid formula)
    sneddon        any shape, elastic, no adhesion (exact sphere for the sphere)
    dmt            any shape: F = E* g(h) - F_ad with F_ad the measured pull-off force (minimum of the curve),
                   fitted on the retract curve from the maximum down to the pull-off point; for a sphere the work of
                   adhesion is W = F_ad / (2 pi R) (Derjaguin, Muller, Toporov 1975)
    jkr            sphere: delta = a^2/R - sqrt(2 pi W a / E*), F = 4 E* a^3 / (3 R) - sqrt(8 pi W E* a^3)
                   (Johnson, Kendall, Roberts 1971), fitted on the retract curve; E* and W
    oliver_pharr   any shape: power law F = alpha (h - h_f)^m fitted to the upper part of the unloading (retract)
                   curve, stiffness S = dF/dh at h_max, contact depth h_c = h_max - eps F_max / S, projected area
                   A(h_c) of the shape, E_r = sqrt(pi) S / (2 sqrt(A)) (Oliver and Pharr 1992, 2004)

For hertz, sneddon, dmt and jkr the modulus enters linearly: for a given contact point (and W for JKR) it is the least
squares solution in closed form, and only the contact point is searched (a grid, then a bounded refinement). """

import dataclasses
import functools
import math
from typing import Optional

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

SHAPES = [("sphere", "Sphere"), ("cone", "Cone"), ("pyramid", "Pyramid (4 sides)"), ("punch", "Cylinder punch"),
          ("hyperboloid", "Hyperboloid")]
MODELS = [("hertz", "Hertz"), ("sneddon", "Sneddon"), ("dmt", "DMT"), ("jkr", "JKR"), ("oliver_pharr", "Oliver-Pharr")]
MODEL_SHAPES = {"hertz": ("sphere",), "sneddon": tuple(s for s, _ in SHAPES), "dmt": tuple(s for s, _ in SHAPES),
                "jkr": ("sphere",), "oliver_pharr": tuple(s for s, _ in SHAPES)}
MODEL_SEGMENT = {"hertz": "approach", "sneddon": "approach", "dmt": "retract", "jkr": "retract",
                 "oliver_pharr": "retract"}                  # the segment each model is fitted on by default
EPSILON = {"sphere": 0.75, "cone": 0.72, "pyramid": 0.75, "punch": 1.0, "hyperboloid": 0.75}   # Oliver-Pharr
MODEL_HELP = {
    "hertz": "Elastic sphere (paraboloid approximation, good while the indentation is much smaller than R). "
             "No adhesion: fit the approach curve.",
    "sneddon": "Elastic contact of any axisymmetric shape (Sneddon 1965); the exact sphere formula for the sphere. "
               "No adhesion: fit the approach curve.",
    "dmt": "Elastic contact plus a constant adhesion force: stiff samples with small adhesion. Fitted on the retract "
           "curve from the maximum force down to the pull-off point.",
    "jkr": "Elastic sphere with adhesion inside the contact: soft samples with strong adhesion. Fitted on the retract "
           "curve; gives the modulus and the work of adhesion.",
    "oliver_pharr": "Unloading stiffness at the maximum force and the projected contact area: elastic-plastic "
                    "samples (permanent indents). Fitted on the upper part of the retract curve.",
}
_TABLE_POINTS = 1500


class FitError(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class Tip:
    shape: str = "sphere"
    radius: float = 10e-9            # m: sphere and hyperboloid apex radius, punch radius
    half_angle: float = 20.0         # degrees: cone, pyramid (axis to face), hyperboloid asymptote
    poisson: float = 0.5             # of the sample

    @property
    def alpha(self) -> float:
        return math.radians(self.half_angle)


# ---- elastic force of a shape per unit reduced modulus ----

def _sphere_exact_table(radius: float) -> tuple:
    """ (h, g) of the exact sphere over a in (0, R), for interpolation. """
    t = np.linspace(1e-4, 1.0 - 1e-6, _TABLE_POINTS) ** 1.0
    a = radius * t
    log = np.log((radius + a) / (radius - a))
    return 0.5 * a * log, 0.5 * ((a * a + radius * radius) * log - 2.0 * a * radius)


def _hyperboloid(h: np.ndarray, radius: float, alpha: float) -> np.ndarray:
    """ Hyperboloid of apex radius R and asymptotic half angle alpha (Akhremitchev and Walker 1999): the contact radius
        a is the root of h = a^2/(2R) zeta (pi/2 + atan(1/(2 zeta) - zeta/2)), zeta = R cot(alpha) / a (bisection in
        log space for all points at once), then g = a^3 / R (zeta^2 + zeta/2 (1 - zeta^2) (pi/2 + atan(...))). """
    cot_a = 1.0 / math.tan(alpha)

    def arc(zeta):
        x = 1.0 / (2.0 * zeta) - zeta / 2.0
        with np.errstate(divide="ignore"):
            return np.where(x < 0, -np.arctan(1.0 / x), np.pi / 2 + np.arctan(x))

    def depth(a):
        zeta = radius * cot_a / a
        return (a * a / (2.0 * radius)) * zeta * arc(zeta)

    low = np.full(h.shape, radius * 1e-8)
    high = np.full(h.shape, 1.0)
    for _ in range(90):
        middle = np.sqrt(low * high)
        below = depth(middle) < h
        low = np.where(below, middle, low)
        high = np.where(below, high, middle)
    a = np.sqrt(low * high)
    zeta = radius * cot_a / a
    return (zeta ** 2 + (zeta / 2.0) * (1.0 - zeta ** 2) * arc(zeta)) * a ** 3 / radius


def unit_force(h, tip: Tip, exact_sphere: bool = False) -> np.ndarray:
    """ Elastic force per unit reduced modulus, g(h) in N/Pa, of the shape at the indentations h (m); 0 for h <= 0. """
    h = np.asarray(h, dtype=float)
    depth = np.where(h > 0, h, 0.0)
    shape, radius, alpha = tip.shape, tip.radius, tip.alpha
    if shape == "sphere" and not exact_sphere:
        g = (4.0 / 3.0) * math.sqrt(radius) * depth ** 1.5
    elif shape == "sphere":
        table_h, table_g = _sphere_exact_table(radius)
        g = np.interp(depth, table_h, table_g, right=np.nan)
        far = depth > table_h[-1]                               # deeper than the table: the sphere is fully in
        g = np.where(far, table_g[-1] + 2.0 * radius * (depth - table_h[-1]), g)
    elif shape == "cone":
        g = (2.0 / math.pi) * math.tan(alpha) * depth ** 2
    elif shape == "pyramid":
        g = (math.tan(alpha) / math.sqrt(2.0)) * depth ** 2
    elif shape == "punch":
        g = 2.0 * radius * depth
    elif shape == "hyperboloid":
        g = np.zeros_like(depth)
        inside = depth > 0
        if np.any(inside):
            g[inside] = _hyperboloid(depth[inside], radius, alpha)
    else:
        raise FitError(f"Unknown tip shape '{shape}'.")
    return np.where(h > 0, g, 0.0)


def projected_area(hc: float, tip: Tip) -> float:
    """ Projected contact area (m²) of the shape at the contact depth hc (Oliver-Pharr). """
    hc = max(float(hc), 0.0)
    shape, radius, tan_a = tip.shape, tip.radius, math.tan(tip.alpha)
    if shape == "sphere":
        return math.pi * (2.0 * radius * hc - hc * hc) if hc < radius else math.pi * radius * radius
    if shape == "cone":
        return math.pi * (hc * tan_a) ** 2
    if shape == "pyramid":
        return 4.0 * (hc * tan_a) ** 2
    if shape == "punch":
        return math.pi * radius * radius
    if shape == "hyperboloid":
        c = radius / tan_a ** 2                                # z(r) = sqrt(c² + r²/tan²α) - c, apex radius R
        return math.pi * tan_a ** 2 * (hc * hc + 2.0 * hc * c)
    raise FitError(f"Unknown tip shape '{shape}'.")


# ---- results ----

@dataclasses.dataclass
class FitResult:
    model: str
    shape: str
    modulus: float                       # Young's modulus E of the sample, Pa (rigid tip)
    reduced_modulus: float               # E* = E / (1 - nu²), Pa
    contact_point: float                 # delta0, m (sample position coordinates)
    adhesion_force: float = 0.0          # N (DMT: fitted offset; JKR: pull-off force 3/2 pi W R)
    work_of_adhesion: float = math.nan   # J/m² (DMT sphere, JKR)
    max_indentation: float = math.nan    # m
    rms: float = math.nan                # residual rms, N
    r2: float = math.nan
    points: int = 0
    success: bool = True
    extra: dict = dataclasses.field(default_factory=dict)   # Oliver-Pharr: S, hc, A, hf, m, eps
    fit_delta: Optional[np.ndarray] = None                  # the fitted points and the model, for the plot
    fit_force: Optional[np.ndarray] = None

    def values(self) -> dict:
        """ name -> (value, unit) for tables and statistics. """
        out = {"E": (self.modulus, "Pa"), "E*": (self.reduced_modulus, "Pa"),
               "contact point": (self.contact_point, "m"), "max indentation": (self.max_indentation, "m")}
        if self.model in ("dmt", "jkr"):
            out["adhesion (fit)"] = (self.adhesion_force, "N")
        if math.isfinite(self.work_of_adhesion):
            out["work of adhesion"] = (self.work_of_adhesion, "J/m²")
        for key, unit in (("S", "N/m"), ("hc", "m"), ("A", "m²"), ("hf", "m"), ("m", "")):
            if key in self.extra:
                out[key] = (self.extra[key], unit)
        out["fit rms"] = (self.rms, "N")
        out["R²"] = (self.r2, "")
        return out


def _r2(force, model) -> float:
    total = float(np.sum((force - force.mean()) ** 2))
    return 1.0 - float(np.sum((force - model) ** 2)) / total if total > 0 else math.nan


def _check(delta, force, minimum=6):
    delta = np.asarray(delta, dtype=float)
    force = np.asarray(force, dtype=float)
    good = np.isfinite(delta) & np.isfinite(force)
    if good.sum() < minimum:
        raise FitError(f"Too few points to fit ({int(good.sum())}).")
    return delta[good], force[good]


def _search(cost, low: float, high: float, grid: int = 60) -> float:
    """ Minimum of a 1D cost over [low, high]: a grid, then a bounded refinement around the best grid point. """
    xs = np.linspace(low, high, grid)
    costs = np.array([cost(x) for x in xs])
    best = int(np.nanargmin(costs))
    lo, hi = xs[max(best - 1, 0)], xs[min(best + 1, grid - 1)]
    if hi <= lo:
        return float(xs[best])
    result = minimize_scalar(cost, bounds=(lo, hi), method="bounded", options={"xatol": (high - low) * 1e-6})
    return float(result.x) if result.fun <= costs[best] else float(xs[best])


# ---- elastic models: hertz, sneddon (approach) and dmt (retract) ----

def fit_elastic(delta, force, tip: Tip, model: str = "hertz") -> FitResult:
    """ Fits F = E* g(delta - delta0) (hertz, sneddon) or F = E* g(delta - delta0) - F_ad (dmt) to the points given.
        For hertz and sneddon give the approach curve from before the contact to the maximum; the model force is 0
        out of contact. For dmt give the retract curve from the maximum force to the pull-off point (minimum). """
    if model not in ("hertz", "sneddon", "dmt"):
        raise FitError(f"'{model}' is not an elastic model.")
    if tip.shape not in MODEL_SHAPES[model]:
        raise FitError(f"The {model} model is not defined for the shape '{tip.shape}'.")
    delta, force = _check(delta, force)
    exact = model == "sneddon"
    adhesion = model == "dmt"
    span = float(delta.max() - delta.min())
    if span <= 0:
        raise FitError("The curve has no extent.")
    low, high = float(delta.min()) - (0.2 * span if adhesion else 0.0), float(delta.max()) - span * 1e-3

    f_ad = max(-float(force.min()), 0.0) if adhesion else 0.0    # DMT: the measured pull-off force

    def solve(delta0):
        g = unit_force(delta - delta0, tip, exact)
        target = force + f_ad
        norm = float(g @ g)
        e_star = max(float(g @ target) / norm, 0.0) if norm > 0 else 0.0
        return e_star, f_ad, e_star * g - f_ad

    def cost(delta0):
        return float(np.sum((force - solve(delta0)[2]) ** 2))

    delta0 = _search(cost, low, high)
    e_star, f_ad, model_force = solve(delta0)
    if not (e_star > 0 and math.isfinite(e_star)):
        raise FitError("No contact found: the force does not rise with the indentation.")
    nu = tip.poisson
    work = f_ad / (2.0 * math.pi * tip.radius) if adhesion and tip.shape == "sphere" else math.nan
    order = np.argsort(delta)
    return FitResult(model, tip.shape, e_star * (1.0 - nu * nu), e_star, delta0, f_ad, work,
                     float(delta.max() - delta0), float(np.sqrt(np.mean((force - model_force) ** 2))),
                     _r2(force, model_force), int(delta.size), True,
                     fit_delta=delta[order], fit_force=model_force[order])


# ---- JKR (sphere, retract) ----

def _jkr_curve(w: float, radius: float, delta_max: float) -> tuple:
    """ Stable branch of JKR in units of the radius (lengths / R): (delta, force / (E* R²)) for w = W / (E* R) from the
        turning point of delta up to delta_max. a = a_min + t² resolves the square root at the turning point. """
    w = max(w, 1e-14)
    a_min = (math.pi * w / 8.0) ** (1.0 / 3.0)
    a_top = max(math.sqrt(max(delta_max, 0.0)) * 1.2, a_min) + (2.0 * math.pi * w) ** (1.0 / 3.0) + 1e-6
    t = np.linspace(0.0, math.sqrt(max(a_top - a_min, 1e-12)), 600)
    a = a_min + t * t
    d = a * a - np.sqrt(2.0 * math.pi * w * a)
    f = 4.0 * a ** 3 / 3.0 - np.sqrt(8.0 * math.pi * w * a ** 3)
    return d, f


def jkr_force(h, reduced_modulus: float, work: float, radius: float) -> np.ndarray:
    """ JKR force (N) at the indentations h (m) relative to the point of zero indentation; 0 where detached. """
    h = np.asarray(h, dtype=float) / radius
    d, f = _jkr_curve(work / (reduced_modulus * radius), 1.0, float(np.nanmax(h, initial=0.0)))
    return np.interp(h, d, f, left=0.0) * reduced_modulus * radius * radius


def fit_jkr(delta, force, tip: Tip) -> FitResult:
    """ JKR fit of the retract curve of a sphere (points from the maximum force through the pull-off). E* is the
        least squares solution for every (W / E* R, delta0); those two are searched on a grid and refined. """
    if tip.shape != "sphere":
        raise FitError("JKR is defined for a sphere only.")
    delta, force = _check(delta, force)
    radius = tip.radius
    span = float(delta.max() - delta.min())
    pull_off = max(-float(force.min()), 0.0)
    order = np.argsort(delta)
    x = (delta[order] - float(delta.max())) / radius          # <= 0, in units of R
    y = force[order]

    def projected(params):
        log_w, x0 = params
        d, f = _jkr_curve(10.0 ** log_w, 1.0, -x0)
        shape = np.interp(x - x0, d, f, left=0.0) * radius * radius
        norm = float(shape @ shape)
        e_star = max(float(shape @ y) / norm, 0.0) if norm > 0 else 0.0
        return shape, e_star

    def residual(params):
        shape, e_star = projected(params)
        return (y - e_star * shape) / max(float(np.abs(y).max()), 1e-30)

    x_low, x_high = float(x.min()) - 0.2 * span / radius, -1e-6
    starts = [(lw, x0) for lw in np.linspace(-8.0, 1.0, 10) for x0 in np.linspace(x_low, x_high, 12)]
    costs = [float(np.sum(residual(s) ** 2)) for s in starts]
    best = starts[int(np.argmin(costs))]
    result = least_squares(residual, best, bounds=([-10.0, x_low], [3.0, x_high]), diff_step=1e-6, max_nfev=300)
    shape, e_star = projected(result.x)
    if not (e_star > 0 and math.isfinite(e_star)):
        raise FitError("The JKR fit did not converge.")
    w_ratio, x0 = 10.0 ** result.x[0], result.x[1]
    work = w_ratio * e_star * radius
    model_force = e_star * shape
    nu = tip.poisson
    delta0 = float(delta.max()) + x0 * radius
    return FitResult("jkr", "sphere", e_star * (1.0 - nu * nu), e_star, delta0, 1.5 * math.pi * work * radius,
                     work, float(delta.max() - delta0), float(np.sqrt(np.mean((y - model_force) ** 2))),
                     _r2(y, model_force), int(y.size), bool(result.success),
                     extra={"measured pull-off": pull_off},
                     fit_delta=delta[order], fit_force=model_force)


# ---- Oliver-Pharr (unloading curve) ----

def fit_oliver_pharr(delta, force, tip: Tip, contact_point: float, upper: float = 0.95, lower: float = 0.2,
                     beta: float = 1.0) -> FitResult:
    """ Oliver-Pharr analysis of the retract (unloading) curve: h = delta - contact_point (from the approach curve),
        F = alpha (h - hf)^m fitted to the points with lower F_max <= F <= upper F_max. """
    delta, force = _check(delta, force)
    h = delta - contact_point
    f_max = float(force.max())
    if not f_max > 0:
        raise FitError("The retract curve has no positive force.")
    h_max = float(h[int(np.argmax(force))])
    use = (force >= lower * f_max) & (force <= upper * f_max) & (h <= h_max)
    if use.sum() < 5:
        raise FitError("Too few unloading points between the force limits.")
    hu, fu = h[use], force[use]
    slope0 = np.polyfit(hu, fu, 1)[0]
    hf0 = h_max - f_max / slope0 if slope0 > 0 else float(hu.min()) - 1e-9

    def residual(p):
        alpha, hf, m = p
        return (alpha * np.clip(hu - hf, 1e-30, None) ** m - fu) / f_max

    m0 = 1.5
    alpha0 = f_max / max(h_max - hf0, 1e-12) ** m0
    try:
        result = least_squares(residual, [alpha0, min(hf0, float(hu.min()) - 1e-12), m0],
                               bounds=([0.0, -np.inf, 1.0], [np.inf, float(hu.min()) - 1e-15, 3.0]),
                               x_scale=[alpha0, max(h_max - hf0, 1e-9), 1.0], max_nfev=500)
    except ValueError as e:
        raise FitError(f"The unloading fit failed: {e}")
    alpha, hf, m = result.x
    stiffness = m * alpha * (h_max - hf) ** (m - 1.0)
    eps = EPSILON[tip.shape]
    hc = h_max - eps * f_max / stiffness
    area = projected_area(hc, tip)
    if not (area > 0 and stiffness > 0):
        raise FitError("The contact depth is not positive: the unloading curve is too steep for this shape.")
    e_r = math.sqrt(math.pi) * stiffness / (2.0 * beta * math.sqrt(area))
    nu = tip.poisson
    model_force = alpha * np.clip(hu - hf, 0.0, None) ** m
    order = np.argsort(hu)
    return FitResult("oliver_pharr", tip.shape, e_r * (1.0 - nu * nu), e_r, contact_point, 0.0, math.nan, h_max,
                     float(np.sqrt(np.mean((fu - model_force) ** 2))), _r2(fu, model_force), int(hu.size),
                     bool(result.success), extra={"S": stiffness, "hc": hc, "A": area, "hf": hf, "m": m, "eps": eps},
                     fit_delta=hu[order] + contact_point, fit_force=model_force[order])


@functools.lru_cache(maxsize=None)
def allowed_shapes(model: str) -> tuple:
    return MODEL_SHAPES[model]
