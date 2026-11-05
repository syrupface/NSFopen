# -*- coding: utf-8 -*-
# Part of NSFopen.process; the Nanosurf Surface Analyzer (Spectroscopy page) uses this module.
""" Contact mechanics of force curves: the elastic force of a tip shape pressed into a flat sample, and the fits of the
models Hertz, Sneddon, DMT, JKR, Maugis-Dugdale and Oliver-Pharr, and the bottom-effect correction of thin samples. No GUI
code; all values in SI units.

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
    maugis         sphere: Maugis-Dugdale (Maugis 1992), the adhesion zone around the contact of width set by the
                   Maugis parameter lambda; DMT for lambda -> 0, JKR for lambda -> infinity. In the units of Johnson and
                   Greenwood (1997), K = 4 E*/3, A = a / (pi W R^2 / K)^(1/3), P = F / (pi W R),
                   D = delta / (pi^2 W^2 R / K^2)^(1/3), m = c / a (c: outer radius of the adhesion zone):
                     lambda A^2/2 [sqrt(m^2-1) + (m^2-2) atan sqrt(m^2-1)]
                       + 4 lambda^2 A/3 [sqrt(m^2-1) atan sqrt(m^2-1) - m + 1] = 1
                     P = A^3 - lambda A^2 [sqrt(m^2-1) + m^2 atan sqrt(m^2-1)]
                     D = A^2 - 4/3 lambda A sqrt(m^2-1)
                   Fitted on the retract curve: E*, W and lambda; the Tabor parameter is mu = 1.157 lambda
                   (Johnson 1997), the adhesive stress sigma0 = lambda/2 (pi W K^2 / R)^(1/3).
    oliver_pharr   any shape: power law F = alpha (h - h_f)^m fitted to the upper part of the unloading (retract)
                   curve, stiffness S = dF/dh at h_max, contact depth h_c = h_max - eps F_max / S, projected area
                   A(h_c) of the shape, E_r = sqrt(pi) S / (2 sqrt(A)) (Oliver and Pharr 1992, 2004)

Bottom effect (thin samples on a hard substrate, of thickness T at the point): the elastic force is multiplied by
f = 1 + c1 chi + c2 chi^2 + c3 chi^3 + c4 chi^4 (hertz, sneddon, dmt):
    sphere     chi = sqrt(R h) / T; bonded or not bonded to the substrate, any nu (Dimitriadis et al. 2002: the series in
               alpha0, beta0 of their Eq. 11-12; bonded nu = 0.5: 1.133, 1.283, 0.769, -0.0975)
    cone       chi = h tan(alpha) / T, bonded, nu = 0.5: 0.721, 0.650, 0.491, 0.225 (Garcia and Garcia 2018)
    pyramid    as the cone of the same force (tan(alpha_cone) = pi tan(alpha) / (2 sqrt 2))
    punch      chi = R / T, bonded, nu = 0.5: 1.133, 1.283, 0.598, -0.291 (Garcia and Garcia 2018)
The series hold for chi up to about 1; beyond that the result is an estimate.

For hertz, sneddon, dmt, jkr and maugis the modulus enters linearly: for a given contact point (and W for JKR) it is the least
squares solution in closed form, and only the contact point is searched (a grid, then a bounded refinement). """

import dataclasses
import functools
import math
from typing import Optional

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

SHAPES = [("sphere", "Sphere"), ("cone", "Cone"), ("pyramid", "Pyramid (4 sides)"), ("punch", "Cylinder punch"),
          ("hyperboloid", "Hyperboloid")]
MODELS = [("hertz", "Hertz"), ("sneddon", "Sneddon"), ("dmt", "DMT"), ("jkr", "JKR"), ("maugis", "Maugis-Dugdale"),
          ("oliver_pharr", "Oliver-Pharr")]
MODEL_SHAPES = {"hertz": ("sphere",), "sneddon": tuple(s for s, _ in SHAPES), "dmt": tuple(s for s, _ in SHAPES),
                "jkr": ("sphere",), "maugis": ("sphere",), "oliver_pharr": tuple(s for s, _ in SHAPES)}
MODEL_SEGMENT = {"hertz": "approach", "sneddon": "approach", "dmt": "retract", "jkr": "retract", "maugis": "retract",
                 "oliver_pharr": "retract"}
BOTTOM_MODELS = ("hertz", "sneddon", "dmt")                  # the models the bottom-effect correction is for
BOTTOM_SHAPES = ("sphere", "cone", "pyramid", "punch")                  # the segment each model is fitted on by default
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
    "maugis": "Elastic sphere with adhesion acting over a zone around the contact (Maugis-Dugdale): between DMT and JKR, "
              "for any sample. Fitted on the retract curve; gives the modulus, the work of adhesion and the Maugis "
              "parameter (small: DMT applies, large: JKR applies).",
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
    thickness: float = math.inf      # m: of the sample at the point (bottom-effect correction); inf: thick
    bonded: bool = True              # the sample is bonded to the substrate (bottom effect)

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


def _dimitriadis(bonded: bool, nu: float) -> tuple:
    """ c1..c4 of the sphere (Dimitriadis et al. 2002, Eq. 11 and 12 written as a series in chi). """
    if bonded:
        a0 = -(1.2876 - 1.4678 * nu + 1.3442 * nu * nu) / (1.0 - nu)
        b0 = (0.6387 - 1.0277 * nu + 1.5164 * nu * nu) / (1.0 - nu)
    else:
        a0 = -0.347 * (3.0 - 2.0 * nu) / (1.0 - nu)
        b0 = 0.056 * (5.0 - 2.0 * nu) / (1.0 - nu)
    pi = math.pi
    return (-2.0 * a0 / pi, 4.0 * a0 * a0 / pi ** 2, -8.0 / pi ** 3 * (a0 ** 3 + 4.0 * pi * pi / 15.0 * b0),
            16.0 * a0 / pi ** 4 * (a0 ** 3 + 3.0 * pi * pi / 5.0 * b0))


GARCIA_CONE = (0.721, 0.650, 0.491, 0.225)
GARCIA_PUNCH = (1.133, 1.283, 0.598, -0.291)


def bottom_factor(h, tip: Tip) -> np.ndarray:
    """ Bottom-effect factor f(h) >= 1 of a sample of thickness tip.thickness (1 for a thick sample), see the module
        doc. h: indentations (m). """
    h = np.clip(np.asarray(h, dtype=float), 0.0, None)
    thickness = tip.thickness
    if not (math.isfinite(thickness) and thickness > 0):
        return np.ones_like(h)
    shape = tip.shape
    if shape not in BOTTOM_SHAPES:
        raise FitError(f"The bottom-effect correction is not known for the shape '{shape}'.")
    if not tip.bonded and shape != "sphere":
        raise FitError("A sample not bonded to the substrate: the correction is known for the sphere only.")
    if shape == "sphere":
        c, chi = _dimitriadis(tip.bonded, tip.poisson), np.sqrt(tip.radius * h) / thickness
    elif shape == "punch":
        c, chi = GARCIA_PUNCH, np.full_like(h, tip.radius / thickness)
    else:
        tan_a = math.tan(tip.alpha)
        if shape == "pyramid":
            tan_a = math.pi * tan_a / (2.0 * math.sqrt(2.0))
        c, chi = GARCIA_CONE, h * tan_a / thickness
    return 1.0 + c[0] * chi + c[1] * chi ** 2 + c[2] * chi ** 3 + c[3] * chi ** 4


def bottom_chi(h_max: float, tip: Tip) -> float:
    """ chi of the bottom effect at the indentation h_max (NaN for a thick sample). """
    thickness = tip.thickness
    if not (math.isfinite(thickness) and thickness > 0):
        return math.nan
    if tip.shape == "sphere":
        return math.sqrt(tip.radius * max(h_max, 0.0)) / thickness
    if tip.shape == "punch":
        return tip.radius / thickness
    tan_a = math.tan(tip.alpha) * (math.pi / (2.0 * math.sqrt(2.0)) if tip.shape == "pyramid" else 1.0)
    return max(h_max, 0.0) * tan_a / thickness


def unit_force(h, tip: Tip, exact_sphere: bool = False) -> np.ndarray:
    """ Elastic force per unit reduced modulus, g(h) in N/Pa, of the shape at the indentations h (m); 0 for h <= 0.
        With a finite tip.thickness the bottom-effect factor is included. """
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
    if math.isfinite(tip.thickness):
        g = g * bottom_factor(depth, tip)
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
    extra: dict = dataclasses.field(default_factory=dict)   # Oliver-Pharr: S, hc, A, hf, m, eps; Maugis: lambda, ...
    fit_delta: Optional[np.ndarray] = None                  # the fitted points and the model, for the plot
    fit_force: Optional[np.ndarray] = None

    def values(self) -> dict:
        """ name -> (value, unit) for tables and statistics. """
        out = {"E": (self.modulus, "Pa"), "E*": (self.reduced_modulus, "Pa"),
               "contact point": (self.contact_point, "m"), "max indentation": (self.max_indentation, "m")}
        if self.model in ("dmt", "jkr", "maugis"):
            out["adhesion (fit)"] = (self.adhesion_force, "N")
        if math.isfinite(self.work_of_adhesion):
            out["work of adhesion"] = (self.work_of_adhesion, "J/m²")
        for key, unit in (("S", "N/m"), ("hc", "m"), ("A", "m²"), ("hf", "m"), ("m", ""), ("Maugis λ", ""),
                          ("Tabor μ", ""), ("adhesive stress", "Pa"), ("sample thickness", "m"),
                          ("bottom χ", ""), ("bottom factor", "")):
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


def _search(cost, low: float, high: float, grid: int = 60, costs_of=None) -> float:
    """ Minimum of a 1D cost over [low, high]: a grid, then a bounded refinement around the best grid point.
        costs_of(xs): the costs of all grid points at once (vectorized), if given. """
    xs = np.linspace(low, high, grid)
    costs = costs_of(xs) if costs_of is not None else np.array([cost(x) for x in xs])
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

    def costs_of(xs):                   # the grid at once: one row of g per contact point
        g = unit_force(delta[None, :] - xs[:, None], tip, exact)
        target = force + f_ad
        norm = np.einsum("ij,ij->i", g, g)
        with np.errstate(invalid="ignore", divide="ignore"):
            e_star = np.where(norm > 0, np.maximum(g @ target / norm, 0.0), 0.0)
        residual = force[None, :] - (e_star[:, None] * g - f_ad)
        return np.einsum("ij,ij->i", residual, residual)

    delta0 = _search(cost, low, high, costs_of=costs_of if delta.size * 60 <= 4_000_000 else None)
    e_star, f_ad, model_force = solve(delta0)
    if not (e_star > 0 and math.isfinite(e_star)):
        raise FitError("No contact found: the force does not rise with the indentation.")
    nu = tip.poisson
    work = f_ad / (2.0 * math.pi * tip.radius) if adhesion and tip.shape == "sphere" else math.nan
    order = np.argsort(delta)
    h_max = float(delta.max() - delta0)
    extra = {}
    if math.isfinite(tip.thickness):
        extra = {"sample thickness": tip.thickness, "bottom χ": bottom_chi(h_max, tip),
                 "bottom factor": float(bottom_factor(h_max, tip))}
    return FitResult(model, tip.shape, e_star * (1.0 - nu * nu), e_star, delta0, f_ad, work,
                     h_max, float(np.sqrt(np.mean((force - model_force) ** 2))),
                     _r2(force, model_force), int(delta.size), True, extra=extra,
                     fit_delta=delta[order], fit_force=model_force[order])


def _no_bottom(tip: Tip, name: str):
    if math.isfinite(tip.thickness):
        raise FitError(f"The bottom-effect correction is for Hertz, Sneddon and DMT, not {name}: set the sample to "
                       f"thick.")


# ---- JKR (sphere, retract) ----

_JKR_Q_MIN = -(27.0 / 256.0) ** (1.0 / 3.0)     # reduced delta at the turning point: no contact below


def _jkr_unit(h, w: float, derivatives: bool = False):
    """ JKR force / (E* R²) at the indentations h / R (stable branch, 0 where detached) for w = W / (E* R), exact.
        With a = c u², c = (2 pi w)^(1/3): delta = c² (u⁴ - u), F = c³ (4/3 u⁶ - 2 u³). u⁴ - u = q = delta / c² is
        solved in closed form (Ferrari): u = (s + sqrt(2/s - s²)) / 2, s = sqrt(2 m), m the real root of
        m³ + q m - 1/8 = 0 (hyperbolic form, stable for every q >= q_min). """
    h = np.asarray(h, dtype=float)
    c = (2.0 * math.pi * max(w, 1e-14)) ** (1.0 / 3.0)
    q = h / (c * c)
    inside = q >= _JKR_Q_MIN
    qq = np.where(inside, q, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        positive = qq > 1e-12
        root_p = np.sqrt(np.abs(qq) / 3.0)
        m_pos = 2.0 * root_p * np.sinh(np.arcsinh(3.0 / (16.0 * np.where(positive, qq, 1.0)) /
                                                   np.where(positive, root_p, 1.0)) / 3.0)
        negative = qq < -1e-12
        arg = 3.0 / (16.0 * np.abs(np.where(negative, qq, 1.0))) / np.where(negative, root_p, 1.0)
        m_neg = 2.0 * root_p * np.cosh(np.arccosh(np.maximum(arg, 1.0)) / 3.0)
        m = np.where(positive, m_pos, np.where(negative, m_neg, 0.5))
        s = np.sqrt(2.0 * m)
        u = 0.5 * (s + np.sqrt(np.maximum(2.0 / s - s * s, 0.0)))
    u3 = u ** 3
    force = np.where(inside, c ** 3 * (4.0 / 3.0 * u3 * u3 - 2.0 * u3), 0.0)
    if not derivatives:
        return force
    # dF/dh = c G'(u) / q'(u) and dF/dlog10(w) = ln10 c/3 (3 c² G - 2 h G'/q'), G = 4/3 u⁶ - 2 u³, q = u⁴ - u
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(inside, c * (8.0 * u3 * u * u - 6.0 * u * u) / np.maximum(4.0 * u3 - 1.0, 1e-9), 0.0)
    g = 4.0 / 3.0 * u3 * u3 - 2.0 * u3
    by_w = np.where(inside, math.log(10.0) * c / 3.0 * (3.0 * c * c * g - 2.0 * h * slope / c), 0.0)
    return force, slope, by_w


def _jkr_curve(w: float, radius: float, delta_max: float) -> tuple:
    """ Stable branch of JKR in units of the radius (lengths / R): (delta, force / (E* R²)) for w = W / (E* R) from the
        turning point of delta up to delta_max (for plots; the fit uses _jkr_unit). """
    c = (2.0 * math.pi * max(w, 1e-14)) ** (1.0 / 3.0)
    u0 = 4.0 ** (-1.0 / 3.0)
    top = max((max(delta_max, 0.0) / (c * c) + 2.0) ** 0.25 * 1.2, u0 + 1e-6)
    u = u0 + np.linspace(0.0, math.sqrt(top - u0), 800) ** 2
    return c * c * (u ** 4 - u) * radius, c ** 3 * (4.0 / 3.0 * u ** 6 - 2.0 * u ** 3)


def jkr_force(h, reduced_modulus: float, work: float, radius: float) -> np.ndarray:
    """ JKR force (N) at the indentations h (m) relative to the point of zero indentation; 0 where detached. """
    h = np.asarray(h, dtype=float) / radius
    return _jkr_unit(h, work / (reduced_modulus * radius)) * reduced_modulus * radius * radius


def _jkr_starts(x: np.ndarray, y: np.ndarray, radius: float, log_ws: np.ndarray, x0s: np.ndarray,
                count: int = 2) -> list:
    """ The best (log w, x0) of the start grid of the JKR fit, the best x0 of each w, the best count of them: one curve
        per w (a shift of x0 only moves it), all x0 at once. """
    found = []
    shifted = (x[None, :] - x0s[:, None]).ravel()
    for log_w in log_ws:
        shapes = _jkr_unit(shifted, 10.0 ** log_w).reshape(x0s.size, x.size) * radius * radius
        norms = np.einsum("ij,ij->i", shapes, shapes)
        with np.errstate(invalid="ignore", divide="ignore"):
            e_star = np.where(norms > 0, np.maximum(shapes @ y / norms, 0.0), 0.0)
        residual = y[None, :] - e_star[:, None] * shapes
        costs = np.einsum("ij,ij->i", residual, residual)
        k = int(np.argmin(costs))
        found.append((float(costs[k]), float(log_w), float(x0s[k])))
    found.sort()
    return [(log_w, x0) for _, log_w, x0 in found[:count]]


def fit_jkr(delta, force, tip: Tip) -> FitResult:
    """ JKR fit of the retract curve of a sphere (points from the maximum force through the pull-off). E* is the
        least squares solution for every (W / E* R, delta0); those two are searched on a grid and refined. """
    if tip.shape != "sphere":
        raise FitError("JKR is defined for a sphere only.")
    _no_bottom(tip, "JKR")
    delta, force = _check(delta, force)
    radius = tip.radius
    span = float(delta.max() - delta.min())
    pull_off = max(-float(force.min()), 0.0)
    order = np.argsort(delta)
    x = (delta[order] - float(delta.max())) / radius          # <= 0, in units of R
    y = force[order]
    x_low, x_high = float(x.min()) - 0.2 * span / radius, -1e-6

    def projected(params):
        log_w = min(max(float(params[0]), -10.0), 3.0)
        x0 = min(max(float(params[1]), x_low), x_high)
        shape = _jkr_unit(x - x0, 10.0 ** log_w) * radius * radius
        norm = float(shape @ shape)
        e_star = max(float(shape @ y) / norm, 0.0) if norm > 0 else 0.0
        return shape, e_star

    scale = max(float(np.abs(y).max()), 1e-30)

    def residual(params):
        shape, e_star = projected(params)
        return (y - e_star * shape) / scale

    def jacobian(params):
        log_w = min(max(float(params[0]), -10.0), 3.0)
        x0 = min(max(float(params[1]), x_low), x_high)
        shape, slope, by_w = (v * radius * radius for v in _jkr_unit(x - x0, 10.0 ** log_w, True))
        norm = float(shape @ shape)
        if norm <= 0:
            return np.zeros((y.size, 2))
        e_star = float(shape @ y) / norm
        out = np.empty((y.size, 2))
        for k, d_shape in enumerate((by_w, -slope)):          # d/d log w, d/d x0
            d_e = (float(d_shape @ y) - 2.0 * e_star * float(shape @ d_shape)) / norm
            out[:, k] = -(d_e * shape + e_star * d_shape) / scale
        return out

    # W and E* trade off on many curves (a shallow valley): refined from the two best starts, the better one kept;
    result = None
    for start in _jkr_starts(x, y, radius, np.linspace(-8.0, 1.0, 20), np.linspace(x_low, x_high, 24), 1):
        trial = least_squares(residual, start, jac=jacobian, bounds=([-10.0, x_low], [3.0, x_high]), max_nfev=300)
        if result is None or trial.cost < result.cost:
            result = trial
    result.x = np.array([min(max(result.x[0], -10.0), 3.0), min(max(result.x[1], x_low), x_high)])
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


# ---- Maugis-Dugdale (sphere, retract) ----

def _md_m(area: np.ndarray, lam: float) -> np.ndarray:
    """ m = c/a >= 1 of the Maugis-Dugdale equation at the reduced contact radii A (bisection in log(m - 1)). """
    def excess(m):
        r = np.sqrt(m * m - 1.0)
        t = np.arctan(r)
        return lam * area ** 2 / 2.0 * (r + (m * m - 2.0) * t) + 4.0 * lam * lam * area / 3.0 * (r * t - m + 1.0) - 1.0

    low = np.full(area.shape, -14.0)                     # log10(m - 1)
    high = np.full(area.shape, 8.0)
    for _ in range(42):
        middle = 0.5 * (low + high)
        above = excess(1.0 + 10.0 ** middle) > 0
        high = np.where(above, middle, high)
        low = np.where(above, low, middle)
    return 1.0 + 10.0 ** (0.5 * (low + high))


def maugis_reduced(lam: float, area: np.ndarray) -> tuple:
    """ (P, D) of Maugis-Dugdale in the reduced units of the module doc, at the reduced contact radii A. """
    area = np.asarray(area, dtype=float)
    m = _md_m(area, lam)
    r = np.sqrt(m * m - 1.0)
    t = np.arctan(r)
    return area ** 3 - lam * area ** 2 * (r + m * m * t), area ** 2 - 4.0 / 3.0 * lam * area * r


@functools.lru_cache(maxsize=512)
def _md_branch(lam: float) -> tuple:
    """ Stable branch (D, P) of Maugis-Dugdale in reduced units, from the turning point of D up to A = 3000 (where the
        adhesion no longer matters); it depends on lambda only, so it is computed once per lambda. """
    area = np.geomspace(1e-3, 3000.0, 2500)
    p, d = maugis_reduced(lam, area)
    start = int(np.argmin(d))
    return d[start:], p[start:]


def _md_curve(w: float, lam: float, delta_max: float = 0.0) -> tuple:
    """ Stable branch of Maugis-Dugdale in units of the radius (lengths / R): (delta, force / (E* R²)) for
        w = W / (E* R), from the turning point of delta on. """
    w = max(w, 1e-14)
    k = 4.0 / 3.0
    depth = (math.pi ** 2 * w * w / (k * k)) ** (1.0 / 3.0)
    d, p = _md_branch(float(lam))
    if d[-1] * depth < delta_max:                       # deeper than the table: Hertz (P = A³, D = A²) beyond it
        extra = np.geomspace(d[-1] * 1.01, delta_max / depth * 1.1, 40)
        d, p = np.concatenate([d, extra]), np.concatenate([p, extra ** 1.5])
    return d * depth, p * math.pi * w


def maugis_force(h, reduced_modulus: float, work: float, lam: float, radius: float) -> np.ndarray:
    """ Maugis-Dugdale force (N) at the indentations h (m) relative to the point of zero indentation (0 where
        detached). """
    h = np.asarray(h, dtype=float) / radius
    d, f = _md_curve(work / (reduced_modulus * radius), lam, float(np.nanmax(h, initial=0.0)))
    return np.interp(h, d, f, left=0.0) * reduced_modulus * radius * radius


def _md_pull_off(lam: float) -> float:
    """ -min P of Maugis-Dugdale (pull-off force / (pi W R)): 2 for DMT, 3/2 for JKR. """
    return -float(_md_branch(float(lam))[1].min())


def fit_maugis(delta, force, tip: Tip) -> FitResult:
    """ Maugis-Dugdale fit of the retract curve of a sphere (points from the maximum force through the pull-off).
        lambda changes the curve only a little over a decade, so a free fit of E*, W, lambda and delta0 has local
        minima. For a given lambda the work of adhesion follows from the measured pull-off force (F = pi W R P(lambda)),
        leaving E* and delta0: that fit is made for lambda on a grid (each one starting from the last), the best
        lambda refined, and at the end all four are fitted together from there. """
    if tip.shape != "sphere":
        raise FitError("Maugis-Dugdale is defined for a sphere only.")
    _no_bottom(tip, "Maugis-Dugdale")
    delta, force = _check(delta, force)
    radius = tip.radius
    span = float(delta.max() - delta.min())
    pull_off = max(-float(force.min()), 0.0)
    if not pull_off > 0:
        raise FitError("No adhesion on the retract curve: use Hertz or Sneddon.")
    order = np.argsort(delta)
    x = (delta[order] - float(delta.max())) / radius
    y = force[order]
    scale = max(float(np.abs(y).max()), 1e-30)
    x_low, x_high = float(x.min()) - 0.2 * span / radius, -1e-6

    def free(params):                      # (log w, log lambda, x0), E* linear
        log_w, log_lam, x0 = params
        d, f = _md_curve(10.0 ** log_w, 10.0 ** log_lam, -x0)
        shape = np.interp(x - x0, d, f, left=0.0) * radius * radius
        norm = float(shape @ shape)
        e_star = max(float(shape @ y) / norm, 0.0) if norm > 0 else 0.0
        return shape, e_star

    def free_residual(params):
        shape, e_star = free(params)
        return (y - e_star * shape) / scale

    def tied_residual(params, lam, work):   # (log w, x0) with W from the pull-off: E* = W / (w R)
        log_w, x0 = params
        w = 10.0 ** log_w
        d, f = _md_curve(w, lam, -x0)
        e_star = work / (w * radius)
        return (y - e_star * radius * radius * np.interp(x - x0, d, f, left=0.0)) / scale

    def at(log_lam, start):
        lam = 10.0 ** log_lam
        work = pull_off / (math.pi * radius * _md_pull_off(lam))
        result = least_squares(tied_residual, start, args=(lam, work), bounds=([-10.0, x_low], [3.0, x_high]),
                               diff_step=1e-5, max_nfev=40)
        return float(result.cost), result.x

    grid = np.linspace(-1.5, 1.3, 29)                 # log10 lambda in steps of 0.1: the minimum is narrow
    first = [(lw, x0) for lw in np.linspace(-8.0, 1.0, 6) for x0 in np.linspace(x_low, x_high, 6)]
    lam0, work0 = 10.0 ** grid[0], pull_off / (math.pi * radius * _md_pull_off(10.0 ** grid[0]))
    start = min(first, key=lambda s: float(np.sum(tied_residual(s, lam0, work0) ** 2)))
    found = []
    for log_lam in grid:
        cost, start = at(log_lam, start)
        found.append((cost, log_lam, start))
    best = min(range(len(found)), key=lambda k: found[k][0])
    lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    solutions = {}

    def profile(log_lam):
        cost, solution = at(log_lam, found[best][2])
        solutions[log_lam] = solution
        return cost

    refined = minimize_scalar(profile, bounds=(lo, hi), method="bounded", options={"xatol": 0.01})
    log_lam = float(refined.x)
    log_w, x0 = solutions.get(log_lam, found[best][2])
    result = least_squares(free_residual, (log_w, log_lam, x0), bounds=([-10.0, -2.0, x_low], [3.0, 1.5, x_high]),
                           diff_step=1e-5, max_nfev=100)
    shape, e_star = free(result.x)
    if not (e_star > 0 and math.isfinite(e_star)):
        raise FitError("The Maugis-Dugdale fit did not converge.")
    w_ratio, lam, x0 = 10.0 ** result.x[0], 10.0 ** result.x[1], result.x[2]
    work = w_ratio * e_star * radius
    k = 4.0 * e_star / 3.0
    sigma0 = lam / 2.0 * (math.pi * work * k * k / radius) ** (1.0 / 3.0)
    model_force = e_star * shape
    nu = tip.poisson
    delta0 = float(delta.max()) + x0 * radius
    return FitResult("maugis", "sphere", e_star * (1.0 - nu * nu), e_star, delta0,
                     math.pi * work * radius * _md_pull_off(lam), work,
                     float(delta.max() - delta0), float(np.sqrt(np.mean((y - model_force) ** 2))),
                     _r2(y, model_force), int(y.size), bool(result.success),
                     extra={"measured pull-off": pull_off, "Maugis λ": lam, "Tabor μ": 1.157 * lam,
                            "adhesive stress": sigma0},
                     fit_delta=delta[order], fit_force=model_force)


# ---- Oliver-Pharr (unloading curve) ----

def fit_oliver_pharr(delta, force, tip: Tip, contact_point: float, upper: float = 0.95, lower: float = 0.2,
                     beta: float = 1.0) -> FitResult:
    """ Oliver-Pharr analysis of the retract (unloading) curve: h = delta - contact_point (from the approach curve),
        F = alpha (h - hf)^m fitted to the points with lower F_max <= F <= upper F_max. """
    _no_bottom(tip, "Oliver-Pharr")
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
