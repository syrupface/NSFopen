# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:13:47 2020

@author: Edward

Script interface to image processing and analysis. The computations are functions
of plain arrays in ops.py, measure.py, grains.py, psdfit.py, fitting.py,
profile_iso.py, combine.py and surface3d.py; the class process holds one file and applies them
to its channels.
"""

import inspect
import math

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from NSFopen.read import read

from . import combine as _combine
from . import fitting, grains as _grains, measure, ops, profile_iso, psdfit, surface3d

OpError = ops.OpError

CURVE_COLOR, FIT_COLOR = "#1f6f8b", "#d1495b"
HEIGHT_CHANNELS = ("Z-Axis", "Topography")       # what the height is called in different files; the default is the first


class process:
    """An image file with its processing and analysis functions.

    Every processing function works on one channel, by default the forward
    Z-Axis, and changes it in place unless inplace=False, when it returns the
    new array and leaves the channel as it was.

    Besides the functions below there is one method per operation of ops.py,
    e.g. level_facet(), filter_fft(), glitch_scars(), mask_otsu(), geometry_crop():
    see operations() and help(process.level_facet). Coordinates passed to
    them (x, y) are (column, row) of the array.

    A mask, kept per channel in .masks, is made by the mask_* functions or set by
    set_mask(). Levels use it with fit_on="exclude" or "only", filters with
    apply_to="masked" or "unmasked", and statistics and the grain analysis by
    region="masked" or "unmasked".
    """

    def __init__(self, filename, **kwargs):
        afm = read(filename, **kwargs)
        self.afm = afm
        self.data = afm.data
        self.param = afm.param
        self.masks = {}
        self._sizes = {}

    # ---- access to the channels ----

    @property
    def channels(self):
        """The image channels of the file: {direction: [names]}."""
        out = {}
        for key in self.data.index:
            if key[0] == "Image":
                out.setdefault(key[1], []).append(key[2])
        return out

    def _key(self, direction, channel):
        channels = self.channels
        if not channels:
            raise ValueError(f"{self.afm.filename} holds no images.")
        if direction not in channels:
            raise ValueError(f"No direction '{direction}': use one of {', '.join(channels)}.")
        if channel not in channels[direction]:
            height = next((name for name in HEIGHT_CHANNELS if name in channels[direction]), None)
            if channel != HEIGHT_CHANNELS[0] or height is None:
                raise ValueError(f"No channel '{channel}' in {direction}: use one of {', '.join(channels[direction])}.")
            channel = height                         # the default channel in a file that calls the height differently
        return ("Image", direction, channel)

    def _channel(self, direction, channel):
        """(direction, channel) as a key of .masks and of the sizes, with the name the file uses."""
        return self._key(direction, channel)[1:]

    def array(self, direction="Forward", channel="Z-Axis"):
        """The channel as a float array."""
        return np.asarray(self.data[self._key(direction, channel)], dtype=float)

    def _store(self, direction, channel, values):
        self.data.iat[self.data.index.get_loc(self._key(direction, channel))] = values

    def _scan_size(self):
        p = self.param
        if "size_x" in p.index:
            return float(p["size_x"]), float(p["size_y"])
        if getattr(self.afm, "image_size", None):
            return self.afm.image_size
        return float(p["X"]["range"][0]), float(p["Y"]["range"][0])

    def size(self, direction="Forward", channel="Z-Axis"):
        """(x, y) size of the channel in m. Crop and rescale_xy change it."""
        return self._sizes.get(self._channel(direction, channel), self._scan_size())

    def unit(self, direction="Forward", channel="Z-Axis"):
        units = getattr(self.afm, "units", {})
        return units.get(self._key(direction, channel), "m")

    @property
    def _row0_is_bottom(self):
        p = self.param
        if "scan_up" in p.index:
            return bool(p["scan_up"])
        header = p["HeaderDump"]["DataSet-Info"] if "HeaderDump" in p.index.get_level_values(0) else {}
        return str(header.get("Scan direction", "Up")).strip().lower() == "up"

    @property
    def origin(self):
        """Where row 0 of the arrays is in the picture, as the origin of matplotlib's imshow: 'lower' or 'upper'."""
        return "lower" if self._row0_is_bottom else "upper"

    def set_channel(self, values, direction="Forward", channel="Z-Axis", unit=None):
        """Stores your own array as a channel, new or existing, for all the processing and plots.

        Parameters
        ----------

            values : array
                Of the size of the other channels of this direction
            unit : string, default the unit of an existing channel, else none
                Its unit, in which the values are, e.g. 'm' or 'V'
        """
        values = np.asarray(values, dtype=float)
        if channel in self.channels.get(direction, []) or channel == HEIGHT_CHANNELS[0]:
            key = self._key(direction, channel)
            shape = self.array(direction, channel).shape
        else:
            names = self.channels.get(direction, [])
            shape = self.array(direction, names[0]).shape if names else values.shape
            key = ("Image", direction, channel)
            self.data.loc[key] = None
            self.afm.units[key] = ""
        if values.shape != shape:
            raise ValueError(f"The array has the shape {values.shape}, the channels of {direction} {shape}.")
        self._store(direction, channel, values)
        if unit is not None:
            self.afm.units[key] = unit

    def set_mask(self, mask, direction="Forward", channel="Z-Axis"):
        """Sets the mask of a channel; None clears it."""
        key = self._channel(direction, channel)
        if mask is None:
            self.masks.pop(key, None)
            return
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.array(direction, channel).shape:
            raise ValueError("The mask has a different shape than the channel.")
        self.masks[key] = mask

    def _state(self, direction, channel, mask=None):
        data = self.array(direction, channel).copy()
        if mask is None:
            mask = self.masks.get(self._channel(direction, channel))
        mask = np.zeros(data.shape, bool) if mask is None else np.array(mask, dtype=bool)
        size_x, size_y = self.size(direction, channel)
        return ops.State(data, mask, size_x, size_y, self.unit(direction, channel))

    def _view(self, direction, channel, mask=None):
        """State as the analysis functions expect it: row 0 at the top of the picture."""
        state = self._state(direction, channel, mask)
        if self._row0_is_bottom:
            state.data, state.mask = state.data[::-1], state.mask[::-1]
        return state

    def _resolver(self, direction):
        def resolve(key):
            if isinstance(key, (tuple, list)):
                return self.array(key[0], key[1])
            return self.array(direction, key)

        return resolve

    # ---- processing ----

    @staticmethod
    def operations():
        """Names of all operations with their parameters and defaults."""
        return {name: dict(spec.defaults) for name, spec in ops.OPS.items()}

    def apply(self, operation, direction="Forward", channel="Z-Axis", inplace=True, **params):
        """Applies an operation of ops.py by name, e.g. apply("level.facet").

        direction is the scan direction, so the rows or columns choice of some
        operations is the keyword axis here.
        """
        spec = ops.OPS.get(operation)
        if spec is not None and "direction" in spec.defaults and "axis" in params:
            params["direction"] = params.pop("axis")
        if operation == "geometry.rotate" and self._row0_is_bottom:  # rotations are as seen in the picture
            turn = params.get("turn", "cw")
            params["turn"] = {"cw": "ccw", "ccw": "cw"}.get(turn, turn)
        new = ops.apply_op(self._state(direction, channel), operation, params, self._resolver(direction))
        if not inplace:
            return new.mask if operation.startswith("mask.") else new.data
        self._store(direction, channel, new.data)
        self.masks[self._channel(direction, channel)] = new.mask
        self._sizes[self._channel(direction, channel)] = (new.size_x, new.size_y)
        return None

    def _finish(self, direction, channel, new, inplace):
        if not inplace:
            return new.data
        self._store(direction, channel, new.data)
        return None

    def flatten(self, order=1, direction="Forward", channel="Z-Axis", mask=[], inplace=True, fit_on=None, clip=0.0):
        """Flattens an image using a nth-order polynomial per scan line

        Parameters
        ----------

            order : int, default 1
                Order of the polynomial
            direction : {'Forward','Backward'}, default 'Forward'
                Scan direction to use
            channel : string, default 'Z-Axis'
                Name of channel to use
            mask : boolean array
                Mask where True values are ignored in the input array
            inplace : boolean, default True
            fit_on : {'all','exclude','only'}, default 'exclude' if mask is given, else 'all'
                Pixels of the channel's mask (or the given mask) to fit
            clip : scalar, default 0
                Repeat the fit without pixels further than clip standard deviations

        Return
        ------
            array or None
                Flattened image or None if inplace=True

        """
        given = np.any(mask)
        fit_on = fit_on or ("exclude" if given else "all")
        state = self._state(direction, channel, mask if given else None)
        ops.level_lines(state, None, order, "rows", "poly", fit_on, clip)
        return self._finish(direction, channel, state, inplace)

    def plane_level(self, direction="Forward", channel="Z-Axis", inplace=True, mask=None, fit_on=None, clip=0.0):
        """Subtracts a plane fitted to the whole image, see flatten for the parameters"""
        given = mask is not None and np.any(mask)
        fit_on = fit_on or ("exclude" if given else "all")
        state = self._state(direction, channel, mask if given else None)
        ops.level_plane(state, None, fit_on, clip)
        return self._finish(direction, channel, state, inplace)

    def remove_outliers(self, direction="Forward", channel="Z-Axis", q=0.01, inplace=True):
        """Replaces the lowest and highest values by interpolation from the pixels around them

        Parameters
        ----------

            q : scalar or [low, high], default 0.01
                Fraction of the pixels to replace at each end of the distribution
        """
        q0, q1 = q if isinstance(q, (list, tuple)) else (q, q)
        state = self._state(direction, channel)
        low, high = np.nanquantile(state.data, [q0, 1 - q1])
        with np.errstate(invalid="ignore"):
            ops._laplace_fill(state, (state.data < low) | (state.data > high) | ~np.isfinite(state.data))
        return self._finish(direction, channel, state, inplace)

    def denoise(self, direction="Forward", channel="Z-Axis", function="gaussian", size=5, inplace=True, **kwargs):
        """Applies a 2D filter to an image.

        Parameters
        ----------

            direction: {'Forward','Backward'}, default 'Forward'
                Scan direction to use
            channel: string, default 'Z-Axis'
                Name of channel to use
            function: {'gaussian','median'}, default 'gaussian'
                Name of filter to apply
            size: scalar
                Size of the filter
            inplace: boolean, default True
                Whether to modify the input array or create new one
            **kwargs:
                Passed to scipy.ndimage filters

        Return
        ------
            array or None
                Filtered image or None if inplace=True

        """
        from scipy.ndimage import gaussian_filter, median_filter

        data = self.array(direction, channel)
        if function == "median":
            out = median_filter(data, size=size, **kwargs)
        else:
            out = gaussian_filter(data, sigma=size, **kwargs)
        if inplace:
            self._store(direction, channel, out)
            return None
        return out

    # ---- statistics ----

    def stats(self):
        """Sa, Sq, Ssk, Sku, Min, Max of all channels as a Series indexed by (direction, channel, name).

        Sku here is the excess kurtosis (0 for a Gaussian surface); statistics()
        gives the ISO 25178 value, which is 3 for a Gaussian surface.
        """

        def Mu(x, i):
            return np.mean((x - x.mean()) ** i)

        func = {
            "Sa": lambda x: np.mean(np.abs(x - x.mean())),
            "Sq": lambda x: np.sqrt(Mu(x, 2)),
            "Ssk": lambda x: Mu(x, 3) / Mu(x, 2) ** (3.0 / 2.0),
            "Sku": lambda x: Mu(x, 4) / Mu(x, 2) ** 2 - 3,
            "Min": np.min,
            "Max": np.max,
        }
        out = {}
        for (direction, channel), values in self.data["Image"].items():
            x = np.asarray(values, dtype=float)
            x = x[np.isfinite(x)]
            for name, f in func.items():
                out[(direction, channel, name)] = f(x)
        return pd.Series(out).sort_index(level=[0, 1], sort_remaining=False)

    def statistics(self, direction="Forward", channel="Z-Axis", region="all", texture=True):
        """ISO 25178 statistics of a channel, with and without its mask.

        Parameters
        ----------

            region : {'all','masked','unmasked'}, default 'all'
            texture : boolean, default True
                Also Sal, Str, Std and the correlation length (slower)

        Return
        ------
            DataFrame
                Columns value and unit, indexed by parameter: Pixels, Area, Mean,
                Median, Min, Max, Sa, Sq, Ssk, Sku, Sp, Sv, Sz, Sdq, Sdr, ...
        """
        result = measure.statistics(self._view(direction, channel), region, texture)
        return pd.DataFrame({"value": [float(value) for value, _ in result.values()],
                             "unit": [unit for _, unit in result.values()]}, index=list(result))

    def bearing_curve(self, direction="Forward", channel="Z-Axis", region="all"):
        """Abbott-Firestone curve: (percent of the pixels above a height, height)."""
        return measure.bearing_curve(self._view(direction, channel), region)

    def histogram_fit(self, direction="Forward", channel="Z-Axis", model="gauss", bins=100, low=None, high=None,
                      region="all"):
        """Fits a distribution to the histogram of a channel.

        Parameters
        ----------

            model : {'gauss','gauss2','gauss3','lorentz','voigt'}, default 'gauss'
            low, high : scalar
                Range of the fit, in the unit of the channel

        Return
        ------
            HistogramFit
                .peaks with center, width, fwhm, area; .summary(); .curve(x)
        """
        state = self._view(direction, channel)
        values = state.data[measure.region_pixels(state.data, state.mask, region)]
        return fitting.fit_histogram(values, model, bins, low, high)

    # ---- lines ----

    def line_profile(self, start, end, width=1, direction="Forward", channel="Z-Axis", unit="m"):
        """A line cut at any angle, averaged across its width.

        Parameters
        ----------

            start, end : (x, y)
                End points. unit 'm': physical position as on the axes of image()
                with the origin at the lower left; unit 'px': (column, row) of the array
            width : int, default 1
                Pixels across the line to average
            unit : {'m','px'}, default 'm'

        Return
        ------
            Profile
                .distance, .mean, .std, .minimum, .maximum, .length, .angle, .roughness()
        """
        state = self._view(direction, channel)
        ny = state.data.shape[0]

        def pixel(point):
            if unit == "px":
                return (point[0], ny - 1 - point[1] if self._row0_is_bottom else point[1])
            if unit == "m":
                return (point[0] / state.dx - 0.5, (state.size_y - point[1]) / state.dy - 0.5)
            raise ValueError("The unit must be 'm' or 'px'.")

        def position(point):
            return ((point[0] + 0.5) * state.dx, state.size_y - (point[1] + 0.5) * state.dy)

        first, last = pixel(start), pixel(end)
        result = measure.line_profile(state, first, last, width)
        result.start, result.end = position(first), position(last)
        return result

    @staticmethod
    def critical_dimension(profile, kind="lines", levels=(0.1, 0.5, 0.9)):
        """Widths, sidewall angles and pitch of lines or trenches on a line_profile."""
        return measure.critical_dimension(profile, kind, levels)

    @staticmethod
    def step_heights(profile, min_plateau=20, exclusion=0.25, threshold=8.0, min_height=0.0):
        """Step heights on a line_profile (ISO 5436-1 style), with their errors."""
        return measure.step_heights(profile, min_plateau, exclusion, threshold, min_height)

    @staticmethod
    def profile_roughness(profile, cutoff=0.0, short="auto", form="line"):
        """Roughness and waviness of a line_profile (ISO 21920): Ra, Rq, Rz, Wa, ... in .params

        Parameters
        ----------

            cutoff : scalar, default 0
                Cutoff in m; 0 chooses one from the length of the profile
            short : {'auto','none'}
            form : {'line','poly2','none'}
        """
        return profile_iso.analyze(profile.distance, profile.mean, profile.unit,
                                   profile_iso.Settings(cutoff, short, form))

    def masked_step_height(self, direction="Forward", channel="Z-Axis", border=3):
        """(height, standard error) of the masked area above the rest of the image."""
        return measure.masked_step_height(self._view(direction, channel), border)

    # ---- spectra and correlation ----

    def psd(self, direction="Forward", channel="Z-Axis", bins=60, counts=False):
        """Radially averaged power spectral density: (k in 1/m, PSD)."""
        return measure.psd_radial(self._view(direction, channel), bins, counts)

    def psd_fit(self, direction="Forward", channel="Z-Axis", model="abc_noise", low=None, high=None, bins=60):
        """Fits the radial PSD: corner, power law, noise floor, effective resolution.

        Parameters
        ----------

            model : {'abc_noise','abc','broken'}, default 'abc_noise'
            low, high : scalar
                Range of the fit in 1/m

        Return
        ------
            PsdFit
                .params, .results, .summary(), .curve(f)
        """
        state = self._view(direction, channel)
        k, power, counts = measure.psd_radial(state, bins, True)
        return psdfit.fit_psd(k, power, model, low, high, counts, max(state.dx, state.dy))

    def spectrum(self, direction="Forward", channel="Z-Axis"):
        """(log10 amplitude of the centered 2D spectrum, kx, ky in 1/m), for filter_notch."""
        return measure.spectrum_2d(self._state(direction, channel))

    def acf(self, direction="Forward", channel="Z-Axis", region="all"):
        """Areal autocorrelation: (acf, tau_x, tau_y in m), centered, as seen in the picture."""
        return measure.acf_2d(self._view(direction, channel), region)

    def line_functions(self, direction="Forward", channel="Z-Axis", axis="x", region="all"):
        """ACF, HHCF and correlation length along the rows (axis 'x') or the columns ('y')."""
        return measure.line_functions(self._view(direction, channel), axis, region)

    def rms_by_scale(self, sizes, direction="Forward", channel="Z-Axis"):
        """[(window size in m, mean Sq of the windows)] for window sizes in pixels."""
        return measure.rms_by_scale(self._state(direction, channel), sizes)

    def slope_distributions(self, direction="Forward", channel="Z-Axis", region="all", bins=90):
        """Distributions of the local inclination and azimuth."""
        return measure.slope_distributions(self._view(direction, channel), region, bins)

    def lattice(self, direction="Forward", channel="Z-Axis", region="all", a=None, b=None, refine=True):
        """Lattice vectors of a periodic image from its autocorrelation; see .summary()."""
        return measure.lattice(self._view(direction, channel), region, a, b, refine)

    # ---- grains ----

    def grains(self, direction="Forward", channel="Z-Axis", **settings):
        """Grain (particle) analysis of the mask of a channel.

        Parameters
        ----------

            connectivity : {4, 8}, default 8
            min_pixels : int, default 5
            exclude_edge : boolean, default False
            split : {'none','height','distance'}, default 'none'
                Splits touching grains by a watershed
            separation, smoothing : scalar
                Pixels between the maxima of a split, Gaussian sigma before finding them

        Return
        ------
            GrainResult
                .table is a DataFrame with one row per grain (Area, Equivalent diameter,
                Max Feret, Height above base, Volume above base, ...), .units its units,
                .summary the totals, .labels an image of the grain numbers. Center y
                counts from the bottom of the image.
        """
        mask = self.masks.get(self._channel(direction, channel))
        if mask is None or not mask.any():
            raise ValueError("The channel has no mask: use e.g. mask_otsu() or mask_threshold() first.")
        state = self._view(direction, channel)
        result = _grains.analyze(state, _grains.GrainSettings(**settings))
        for row in result.rows:
            row["Center y"] = state.size_y - row["Center y"]
        result.table = pd.DataFrame(result.rows)
        if self._row0_is_bottom:
            result.labels = result.labels[::-1]
        return result

    # ---- two channels ----

    def combine(self, a, b, operation="half_diff", shift_x=0.0, shift_y=0.0, name=None):
        """Arithmetic of two channels, e.g. lateral force from forward and backward.

        Parameters
        ----------

            a, b : (direction, channel)
            operation : {'diff','sum','half_diff','mean','a','b'}, default 'half_diff'
            shift_x, shift_y : scalar
                Sub-pixel shift of b, in pixels; see find_shift
            name : string, default None
                Also stores the result as a new channel of the direction of a

        Return
        ------
            array
        """
        result = _combine.combine(self._state(*a), self._state(*b), operation, shift_x, shift_y).data
        if name:
            self.set_channel(result, a[0], name, self.unit(*a))
        return result

    def find_shift(self, a, b, max_shift=0.25):
        """(shift_x, shift_y, correlation) in pixels that moves channel b onto a; a, b = (direction, channel)."""
        return _combine.find_shift(self.array(*a), self.array(*b), max_shift)

    # ---- plots ----

    @staticmethod
    def _scaled(values, unit, scale):
        """(factor, unit label): the readable unit of the values, or the given factor without a label we can know."""
        if scale is None:
            return surface3d.display_unit(unit, values)
        if unit == "m":
            return scale, {1.0: "m", 1e9: "nm", 1e6: "\u00b5m", 1e12: "pm"}.get(scale, "")
        return scale, unit if scale == 1 else ""

    @staticmethod
    def _plot_axes(ax, figsize, dpi, theme):
        """(figure, axes, foreground color): new ones of the theme, or the axes given by the user."""
        import matplotlib.pyplot as plt

        if theme not in ("light", "dark"):
            raise ValueError("The theme must be 'light' or 'dark'.")
        color, background = ("k", "w") if theme == "light" else ("w", "k")
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            fig.patch.set_facecolor(background)
        ax.set_facecolor(background)
        return ax.figure, ax, color

    @staticmethod
    def _color_axes(ax, color, fontsize):
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
        ax.tick_params(color=color, labelcolor=color, labelsize=fontsize)

    def image(
        self,
        direction="Forward",
        channel="Z-Axis",
        scale=None,
        cmap="afmhot",
        cbar=None,
        fontsize=10,
        scalebar=None,
        theme="light",
        title="auto",
        show_mask=False,
        colorbar=True,
        color_range="full",
        figsize=None,
        dpi=300,
        ax=None,
        profile=None,
        **kwargs,
    ):
        """Plots an AFM image

        Parameters
        ----------

            direction: {'Forward','Backward'}, default 'Forward'
                Scan direction to use
            channel: string, default 'Z-Axis'
                Name of channel to use, see .channels
            scale: scalar, default None
                Factor for the data before plotting; by default a readable unit from the channel's unit (nm, mV, ...)
            cmap: string or colormap, default 'afmhot'
                Name from matplotlib or of a gradient of this package ('Gold'), or a colormap
            cbar: string, default None
                Label of the color bar, by default the channel and its unit
            fontsize: scalar, default 10
                Font size to use throughout
            scalebar: True or dict, {'length':scalar,'color':string}, default None
                Applies scalebar to bottom right of image instead of XY axis scale.
                * 'length' : length of scalebar in um, by default a round length near 1/5 of the image
                * 'color' : color of scalebar and label, by default black or white, whichever shows on the image
            theme: {'light','dark'}, default 'light'
                Use either 'light' or 'dark' theme
            title: string or None, default 'auto'
                Title of the plot; 'auto' is the channel and the direction, None leaves it out
            show_mask: boolean, default False
                Shows the mask of the channel in translucent red
            colorbar: boolean, default True
                Add a color bar
            color_range: 'full', 'p1' or (low, high), default 'full'
                Range of the color map: all values, 1-99 % of them, or limits in the unit of the channel (m, V, ...)
            figsize: (width, height) in inches, default 6 wide and as high as the image needs
            dpi: scalar, default 300
            ax: matplotlib axes, default None
                Draws on these axes, e.g. of plt.subplots, instead of a new figure
            profile: result of line_profile or a list of them, default None
                Draws where the line cuts are on the image
            **kwargs:
                Passed to matplotlib imshow

        Return
        ------
            figure
                matplotlib figure object

        """
        key = self._key(direction, channel)
        raw = self.array(direction, channel)
        factor, label = self._scaled(raw, self.unit(direction, channel), scale)
        d = raw * factor
        size_x, size_y = self.size(direction, channel)
        length, length_label = surface3d.display_unit("m", np.array([max(size_x, size_y)]))
        x, y = size_x * length, size_y * length
        origin = self.origin
        show_bar = colorbar or cbar is not None

        fit_figure = ax is None and figsize is None
        if fit_figure:     # margins in inches around an image with the proportions of the scan
            left, right, bottom, top = 0.75, (1.1 if show_bar else 0.25), 0.55, 0.35
            height = min(max((6.0 - left - right) * y / x + bottom + top, 2.2), 8.0)
            figsize = (6.0, height)
        fig, ax, color = self._plot_axes(ax, figsize, dpi, theme)
        if fit_figure:
            fig.subplots_adjust(left=left / 6.0, right=1 - right / 6.0, bottom=bottom / height, top=1 - top / height)

        if "norm" not in kwargs:
            low, high = surface3d._range(raw, color_range) if isinstance(color_range, str) else color_range
            kwargs.setdefault("vmin", low * factor)
            kwargs.setdefault("vmax", high * factor)
        kwargs.setdefault("interpolation", "nearest")
        im = ax.imshow(d, extent=(0, x, 0, y), cmap=surface3d.colormap(cmap), origin=origin, **kwargs)

        mask = self.masks.get(self._channel(direction, channel))
        if show_mask and mask is not None:
            overlay = np.zeros(mask.shape + (4,))
            overlay[..., :3] = surface3d.MASK_COLOR
            overlay[..., 3] = mask * surface3d.MASK_ALPHA
            ax.imshow(overlay, extent=(0, x, 0, y), origin=origin, interpolation="nearest")

        for cut in (profile if isinstance(profile, (list, tuple)) else [profile] if profile is not None else []):
            ax.plot([cut.start[0] * length, cut.end[0] * length], [cut.start[1] * length, cut.end[1] * length],
                    color="#00e5ff", linewidth=1.5)
        ax.set_xlim(0, x)
        ax.set_ylim(0, y)

        self._color_axes(ax, color, fontsize)
        if scalebar:
            settings = scalebar if isinstance(scalebar, dict) else {}
            bar = settings.get("length", None)
            bar = bar * 1e-6 if bar is not None else _round_length(size_x / 5)
            c = settings.get("color") or _visible_on(im, d, origin)
            pad = y * 0.05
            dx = bar * length
            ax.plot([x - dx - pad, x - pad], [pad, pad], linewidth=5, color=c, solid_capstyle="butt")
            scale_bar, unit_bar = surface3d.display_unit("m", np.array([bar]))
            ax.text(x - pad - dx / 2, pad + y * 0.03, f"{bar * scale_bar:.3g} {unit_bar}", ha="center", va="bottom",
                    fontsize=fontsize, color=c)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xlabel(f"X [{length_label}]", fontsize=fontsize, color=color)
            ax.set_ylabel(f"Y [{length_label}]", fontsize=fontsize, color=color)
        if title == "auto":
            title = f"{key[2]} - {direction}"
        if title:
            ax.set_title(title, fontsize=fontsize, color=color)
        if show_bar:
            cb = fig.colorbar(im, cax=ax.inset_axes([1.03, 0.0, 0.045, 1.0]))    # as high as the image, moves with it
            text = cbar if cbar is not None else f"{key[2]} [{label}]" if label else key[2]
            cb.set_label(text, size=fontsize, color=color)
            cb.ax.tick_params(labelsize=fontsize, color=color, labelcolor=color)
            cb.outline.set_edgecolor(color)

        return fig

    def thumb(self, direction="Forward", channel="Z-Axis", cmap="afmhot", color_range="full", figsize=(3, 3), dpi=150,
              ax=None, **kwargs):
        """Plots an image to use as a Thumbnail

        Parameters
        ----------

            direction: {'Forward','Backward'}, default 'Forward'
                Scan direction to use
            channel: string, default 'Z-Axis'
                Name of channel to use, see .channels
            cmap: string or colormap, default 'afmhot'
                Name from matplotlib or of a gradient of this package ('Gold'), or a colormap
            color_range: 'full', 'p1' or (low, high), default 'full'
                Range of the color map, see image
            figsize: (width, height) in inches, default (3, 3)
            dpi: scalar, default 150
            ax: matplotlib axes, default None
                Draws on these axes, e.g. of plt.subplots, instead of a new figure
            **kwargs:
                Passed to matplotlib imshow

        Return
        ------
            figure
                matplotlib figure object

        """
        fig, ax, _ = self._plot_axes(ax, figsize, dpi, "light")
        data = self.array(direction, channel)
        if "norm" not in kwargs:
            low, high = surface3d._range(data, color_range) if isinstance(color_range, str) else color_range
            kwargs.setdefault("vmin", low)
            kwargs.setdefault("vmax", high)
        ax.imshow(data, origin=self.origin, cmap=surface3d.colormap(cmap), **kwargs)
        ax.axis("off")
        return fig

    def hist(
        self,
        direction="Forward",
        channel="Z-Axis",
        bins="sqrt",
        scale=None,
        cmap="afmhot",
        xlabel=None,
        fontsize=10,
        density=False,
        theme="light",
        color_range="full",
        figsize=(6, 4),
        dpi=300,
        ax=None,
        **kwargs,
    ):
        """Plots the histogram of a channel with its bars in the colors of the image

        Parameters
        ----------

            channel: string, default 'Z-Axis'
                Name of channel to use, see .channels
            bins: int or {'sqrt','freedman','rice','sturges','doane','quantile'}, default 'sqrt'
            scale: scalar, default None
                Factor for the data before plotting; by default a readable unit from the channel's unit (nm, mV, ...)
            cmap: string or colormap, default 'afmhot'
                Name from matplotlib or of a gradient of this package ('Gold'), or a colormap
            xlabel: string, default None
                Label of the x axis, by default the channel and its unit
            density: boolean, default False
            theme: {'light','dark'}, default 'light'
            color_range: 'full', 'p1' or (low, high), default 'full'
                Range of the color map, see image; it colors the bars, all of them are shown
            figsize: (width, height) in inches, default (6, 4)
            dpi: scalar, default 300
            ax: matplotlib axes, default None
                Draws on these axes, e.g. of plt.subplots, instead of a new figure
            **kwargs:
                Passed to matplotlib hist; color= makes all bars this color

        Return
        ------
            figure
                matplotlib figure object

        """
        key = self._key(direction, channel)
        raw = self.array(direction, channel)
        factor, label = self._scaled(raw, self.unit(direction, channel), scale)
        data = (raw * factor).reshape(-1)
        data = data[np.isfinite(data)]
        if isinstance(bins, str):
            bins = n_bins(data, method=bins)

        fig, ax, color = self._plot_axes(ax, figsize, dpi, theme)
        low, high = surface3d._range(raw, color_range) if isinstance(color_range, str) else color_range
        colormap = surface3d.colormap(cmap)

        def shades(edge):
            centers = 0.5 * (edge[:-1] + edge[1:])
            return colormap(np.clip((centers - low * factor) / ((high - low) * factor), 0.0, 1.0))

        if kwargs:             # anything of matplotlib's hist
            _, edge, patches = ax.hist(data, bins=bins, density=density, **kwargs)
            if "color" not in kwargs:
                for patch, shade in zip(patches, shades(edge)):
                    patch.set_facecolor(shade)
        else:                  # all bars as one object, much faster with the many bars of a large image
            counts, edge = np.histogram(data, bins=bins, density=density)
            _draw_bars(ax, edge, counts, shades(edge))

        self._color_axes(ax, color, fontsize)
        if xlabel is None:
            xlabel = f"{key[2]} [{label}]" if label else key[2]
        ax.set_xlabel(xlabel, fontsize=fontsize, color=color)
        ax.set_ylabel("Density" if density else "Counts", fontsize=fontsize, color=color)
        return fig

    @staticmethod
    def _finish_axes(ax, color, fontsize, xlabel, ylabel, title=None, legend=False):
        process._color_axes(ax, color, fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize, color=color)
        ax.set_ylabel(ylabel, fontsize=fontsize, color=color)
        if title:
            ax.set_title(title, fontsize=fontsize, color=color)
        if legend:
            ax.legend(fontsize=fontsize, frameon=False, labelcolor=color)

    @staticmethod
    def _labeled(name, label):
        return f"{name} [{label}]" if label else name

    @staticmethod
    def plot_profile(profile, scale=None, steps=None, cd=None, title=None, theme="light", fontsize=10, figsize=(6, 4),
                     dpi=300, ax=None, **kwargs):
        """Plots a line_profile, with the spread across its width as a band

        Parameters
        ----------

            profile: result of line_profile
            scale: scalar, default None
                Factor for the heights; by default a readable unit from the unit of the profile
            steps: result of step_heights, default None
                Draws its fitted plateaus
            cd: result of critical_dimension, default None
                Marks the edges of its features
            title: string, default None
            theme: {'light','dark'}, default 'light'
            fontsize: scalar, default 10
            figsize: (width, height) in inches, default (6, 4)
            dpi: scalar, default 300
            ax: matplotlib axes, default None
                Draws on these axes, e.g. of plt.subplots, instead of a new figure
            **kwargs:
                Passed to matplotlib plot

        Return
        ------
            figure
                matplotlib figure object

        """
        fig, ax, color = process._plot_axes(ax, figsize, dpi, theme)
        factor, label = process._scaled(profile.mean[np.isfinite(profile.mean)], profile.unit, scale)
        length, length_label = surface3d.display_unit("m", np.array([profile.length]))
        x = profile.distance * length
        kwargs.setdefault("color", CURVE_COLOR)
        kwargs.setdefault("linewidth", 1.0)
        line = ax.plot(x, profile.mean * factor, **kwargs)[0]
        if np.nanmax(profile.std) > 0:
            ax.fill_between(x, (profile.mean - profile.std) * factor, (profile.mean + profile.std) * factor,
                            color=line.get_color(), alpha=0.25, linewidth=0, label="spread across the width")
        if steps is not None:
            for x0, x1, z0, z1 in steps.lines:
                ax.plot([x0 * length, x1 * length], [z0 * factor, z1 * factor], color=FIT_COLOR, linewidth=1.5)
        if cd is not None:
            for feature in cd.features:
                for level, edges in feature.edges.items():
                    z = (cd.bottom + level * (cd.top - cd.bottom)) * factor
                    ax.plot([edges[0] * length, edges[1] * length], [z, z], "|", color=FIT_COLOR, markersize=6)
        ax.set_xlim(x[0], x[-1])
        process._finish_axes(ax, color, fontsize, f"Distance [{length_label}]", f"Height [{label}]" if label else "Height",
                             title)
        return fig

    def plot_bearing(self, direction="Forward", channel="Z-Axis", region="all", scale=None, title=None, theme="light",
                     fontsize=10, figsize=(6, 4), dpi=300, ax=None, **kwargs):
        """Plots the bearing (Abbott-Firestone) curve of a channel

        Parameters
        ----------

            region: {'all','masked','unmasked'}, default 'all'
            scale: scalar, default None
                Factor for the heights; by default a readable unit from the channel's unit
            title, theme, fontsize, figsize, dpi, ax:
                As for plot_profile
            **kwargs:
                Passed to matplotlib plot

        Return
        ------
            figure
                matplotlib figure object

        """
        fig, ax, color = self._plot_axes(ax, figsize, dpi, theme)
        percent, height = self.bearing_curve(direction, channel, region)
        factor, label = self._scaled(height, self.unit(direction, channel), scale)
        kwargs.setdefault("color", CURVE_COLOR)
        ax.plot(percent, height * factor, **kwargs)
        ax.set_xlim(0, 100)
        name = self._key(direction, channel)[2]
        self._finish_axes(ax, color, fontsize, "Pixels above the height [%]", self._labeled(name, label), title)
        return fig

    def plot_histogram_fit(self, direction="Forward", channel="Z-Axis", model="gauss", bins=100, low=None, high=None,
                           region="all", scale=None, cmap="afmhot", title=None, theme="light", fontsize=10,
                           figsize=(6, 4), dpi=300, ax=None):
        """Plots the histogram of a channel with a fit of histogram_fit, and each of its peaks

        Parameters
        ----------

            model: {'gauss','gauss2','gauss3','lorentz','voigt'}, default 'gauss'
            bins: int, default 100
            low, high: scalar
                Range of the fit, in the unit of the channel (m, V, ...)
            region: {'all','masked','unmasked'}, default 'all'
            scale: scalar, default None
                Factor for the values; by default a readable unit from the channel's unit
            cmap: string or colormap, default 'afmhot'
            title, theme, fontsize, figsize, dpi, ax:
                As for plot_profile

        Return
        ------
            figure
                matplotlib figure object

        """
        fit = self.histogram_fit(direction, channel, model, bins, low, high, region)
        fig, ax, color = self._plot_axes(ax, figsize, dpi, theme)
        factor, label = self._scaled(fit.x, self.unit(direction, channel), scale)
        edge = np.append(fit.x - fit.bin_width / 2, fit.x[-1] + fit.bin_width / 2) * factor
        share = (fit.x - fit.x[0]) / (np.ptp(fit.x) or 1.0)
        _draw_bars(ax, edge, fit.counts, surface3d.colormap(cmap)(share))
        x = np.linspace(fit.x[0], fit.x[-1], 500)
        ax.plot(x * factor, fit.curve(x), color=FIT_COLOR, linewidth=1.5, label="Fit")
        if len(fit.peaks) > 1:
            for i, peak in enumerate(fit.peaks):
                ax.plot(x * factor, fit.curve(x, i), color=FIT_COLOR, linewidth=1.0, linestyle="--",
                        label=f"{peak.center * factor:.3g} {label}".strip())
        name = self._key(direction, channel)[2]
        self._finish_axes(ax, color, fontsize, self._labeled(name, label), "Counts", title, legend=True)
        return fig

    def plot_psd(self, direction="Forward", channel="Z-Axis", fit="abc_noise", low=None, high=None, bins=60, title=None,
                 theme="light", fontsize=10, figsize=(6, 4), dpi=300, ax=None, **kwargs):
        """Plots the radial power spectral density with a fit of psd_fit

        Parameters
        ----------

            fit: {'abc_noise','abc','broken'} or None, default 'abc_noise'
                Model to fit, None for the data alone
            low, high: scalar
                Range of the fit in 1/m
            bins: int, default 60
            title, theme, fontsize, figsize, dpi, ax:
                As for plot_profile
            **kwargs:
                Passed to matplotlib plot for the data

        Return
        ------
            figure
                matplotlib figure object

        """
        k, power = self.psd(direction, channel, bins)
        unit = self.unit(direction, channel)
        factor, label = (1e30, "nm\u00b2\u00b7\u00b5m\u00b2") if unit == "m" else (1e12, f"{unit}\u00b2\u00b7\u00b5m\u00b2")
        result = self.psd_fit(direction, channel, fit, low, high, bins) if fit else None
        fig, ax, color = self._plot_axes(ax, figsize, dpi, theme)
        kwargs.setdefault("color", CURVE_COLOR)
        kwargs.setdefault("marker", ".")
        kwargs.setdefault("linewidth", 0.8)
        ax.loglog(k * 1e-6, power * factor, label="Data", **kwargs)
        if result is not None:
            ax.loglog(result.f * 1e-6, result.curve(result.f) * factor, color=FIT_COLOR, linewidth=1.5, label=f"Fit: {fit}")
        self._finish_axes(ax, color, fontsize, "Spatial frequency [1/\u00b5m]", f"PSD [{label}]", title, legend=True)
        return fig

    def plot_acf(self, direction="Forward", channel="Z-Axis", region="all", cmap="RdBu_r", colorbar=True, title=None,
                 theme="light", fontsize=10, figsize=None, dpi=300, ax=None, **kwargs):
        """Plots the areal autocorrelation of a channel as an image

        Parameters
        ----------

            region: {'all','masked','unmasked'}, default 'all'
            cmap: string or colormap, default 'RdBu_r'
            colorbar: boolean, default True
            title, theme, fontsize, dpi, ax:
                As for plot_profile
            figsize: (width, height) in inches, default (5.5, 4.5)
            **kwargs:
                Passed to matplotlib imshow, by default the color range is -1 to 1

        Return
        ------
            figure
                matplotlib figure object

        """
        acf, tau_x, tau_y = self.acf(direction, channel, region)
        fig, ax, color = self._plot_axes(ax, figsize or (5.5, 4.5), dpi, theme)
        length, label = surface3d.display_unit("m", np.array([tau_x.max(), tau_y.max()]))
        if "norm" not in kwargs:
            kwargs.setdefault("vmin", -1.0)
            kwargs.setdefault("vmax", 1.0)
        extent = (tau_x[0] * length, tau_x[-1] * length, -tau_y[-1] * length, -tau_y[0] * length)    # y up
        im = ax.imshow(acf, extent=extent, origin="upper", cmap=surface3d.colormap(cmap), **kwargs)
        self._finish_axes(ax, color, fontsize, f"\u03c4x [{label}]", f"\u03c4y [{label}]", title)
        if colorbar:
            cb = fig.colorbar(im, cax=ax.inset_axes([1.03, 0.0, 0.045, 1.0]))
            cb.set_label("Autocorrelation", size=fontsize, color=color)
            cb.ax.tick_params(labelsize=fontsize, color=color, labelcolor=color)
            cb.outline.set_edgecolor(color)
        return fig

    def plot3d(self, direction="Forward", channel="Z-Axis", color=None, material=None, mode=None, cmap="afmhot",
               color_range="full", light=(45.0, 45.0), rotation=-30.0, tilt=55.0, z_scale=None, zoom=1.0,
               perspective=True, axes=True, labels=True, colorbar=True, show_mask=False, max_points=512, theme="light",
               fontsize=10, figsize=None, dpi=300, ax=None):
        """Plots the channel in 3D, as the 3D view of the Surface Analyzer

        Parameters
        ----------

            direction: {'Forward','Backward'}, default 'Forward'
                Scan direction to use
            channel: string, default 'Z-Axis'
                Name of channel that gives the heights
            color: string or (direction, channel), default None
                Colors the surface by this channel, e.g. phase on the topography
            material: string or (ambient, diffuse, specular, shininess), default None
                Draws a lit surface of this material instead of a color map: Gray, Matte white, Brass, Bronze,
                Chrome, Copper, Gold, Silver, Pewter, Pearl, Jade, Ruby, Turquoise or Obsidian
            mode: {'gradient','overlay','lighting','overlay_light'}, default from color and material
                Colors from the heights, colors from another channel, a material under a light, or another
                channel's colors under a light
            cmap: string or colormap, default 'afmhot'
                Name from matplotlib or of a gradient of this package ('Gold'), or a colormap
            color_range: 'full', 'p1' or (low, high), default 'full'
                Range of the color map: all values, 1-99 % of them, or limits in the unit of the channel
            light: (phi, theta), default (45, 45)
                Direction of the light in degrees: around the vertical, above the plane
            rotation, tilt: scalar, default -30, 55
                View in degrees: around the vertical, from looking straight down (tilt 0)
            z_scale: scalar or 'true', default None
                Height of the full data range relative to the longer image side; 'true' for true proportions, by
                default these but at least 0.15 and at most 0.6, as AFM heights would look flat
            zoom: scalar, default 1
                Enlarges the surface in the figure
            perspective: boolean, default True
            axes, labels, colorbar: boolean, default True
                Draw the axes, their labels, the color bar (not for a material)
            show_mask: boolean, default False
                Shows the mask of the channel in translucent red
            max_points: int, default 512
                Largest number of points per side, higher is slower; larger images are averaged down
            theme: {'light','dark'}, default 'light'
            fontsize: scalar, default 10
            figsize: (width, height) in inches, default 6 wide and as high as the surface needs
            dpi: scalar, default 300
                Resolution of the surface, which is a picture; axes and labels stay sharp
            ax: matplotlib axes, default None
                Draws on these axes, e.g. of plt.subplots, instead of a new figure

        Return
        ------
            figure
                matplotlib figure object

        """
        state = self._view(direction, channel)
        color_state, color_unit = None, None
        if color is not None:
            key = tuple(color) if isinstance(color, (tuple, list)) else (direction, color)
            color_state, color_unit = self._view(*key), self.unit(*key)
        return surface3d.figure(state, color_state, mode, cmap, color_unit, color_range, material, light, rotation, tilt,
                                z_scale, zoom, perspective, axes, labels, colorbar, show_mask, max_points, theme,
                                fontsize, figsize=figsize, dpi=dpi, ax=ax)


def _add_operation_methods():
    """One method per operation of ops.py: 'level.facet' becomes process.level_facet()."""
    base = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    for name, default in (("direction", "Forward"), ("channel", "Z-Axis"), ("inplace", True)):
        base.append(inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default))

    def make(operation, spec):
        def method(self, direction="Forward", channel="Z-Axis", inplace=True, **params):
            return self.apply(operation, direction, channel, inplace, **params)

        method.__name__ = operation.replace(".", "_")
        method.__qualname__ = f"process.{method.__name__}"
        method.__doc__ = f"{spec.label}.\n\n{inspect.getdoc(spec.func) or ''}"
        keywords = [inspect.Parameter("axis" if key == "direction" else key, inspect.Parameter.KEYWORD_ONLY,
                                      default=value) for key, value in spec.defaults.items()]
        if "direction" in spec.defaults:
            method.__doc__ += "\n\nThe rows or columns choice is the keyword axis here (direction is the scan direction)."
        method.__signature__ = inspect.Signature(base + keywords)
        return method

    for operation, spec in ops.OPS.items():
        setattr(process, operation.replace(".", "_"), make(operation, spec))


_add_operation_methods()


def toPandaDF(user_dict):
    df = pd.DataFrame.from_dict(
        {
            (i, j): user_dict[i][j]
            for i in user_dict.keys()
            for j in user_dict[i].keys()
        },
        orient="index",
    ).transpose()
    return df


def _draw_bars(ax, edge, counts, colors):
    """Histogram bars as one object, much faster than single patches with the many bars of a large image."""
    from matplotlib.collections import PolyCollection

    left, right = edge[:-1], edge[1:]
    bars = np.stack([np.stack([left, right, right, left], 1), np.stack([0 * counts, 0 * counts, counts, counts], 1)], axis=2)
    ax.add_collection(PolyCollection(bars, facecolors=colors, edgecolors="face", linewidths=0.2))
    ax.set_xlim(edge[0], edge[-1])
    ax.set_ylim(0, counts.max() * 1.05 if counts.max() > 0 else 1)


def _round_length(length):
    """The largest length of 1, 2 or 5 times a power of ten that is not above the given length."""
    power = 10.0 ** math.floor(math.log10(length))
    return next(m * power for m in (5, 2, 1) if m * power <= length * (1 + 1e-9))


def _visible_on(im, data, origin):
    """Black or white, whichever shows better on the lower right corner of an image."""
    ny, nx = data.shape
    rows = slice(0, max(1, ny // 5)) if origin == "lower" else slice(-max(1, ny // 5), None)
    corner = im.to_rgba(data[rows, -max(1, nx // 3):])[..., :3]
    return "k" if np.mean(corner @ np.array([0.299, 0.587, 0.114])) > 0.5 else "w"


def n_bins(x, method="sqrt"):
    n = len(x)
    if method == "freedman":  # freedman-diaconis rule
        h = 2 * scipy_stats.iqr(x) / np.cbrt(n)
        k = int(np.ceil(np.ptp(x) / h)) if h > 0 else int(np.ceil(np.sqrt(n)))
    elif method == "rice":
        k = int(np.ceil(2 * np.cbrt(n)))
    elif method == "sturges":
        k = int(np.ceil(np.log2(n)) + 1)
    elif method == "doane":
        g = scipy_stats.skew(x)
        sigma = np.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
        k = int(1 + np.log2(n) + np.log2(1 + np.abs(g) / sigma))
    elif method == "quantile":
        k = np.quantile(x, np.arange(0, 1.01, 0.025))
    else:
        k = int(np.ceil(np.sqrt(n)))
    return k
