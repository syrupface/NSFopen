# -*- coding: utf-8 -*-
"""
Tests of NSFopen. Run from the repository root with: python -m unittest discover tests

NHF files are written here with the layouts of real files, so the tests need no
measurement data apart from the examples of the package.
"""

import os
import pathlib
import tempfile
import unittest

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")

from NSFopen.process import OpError, colors, process  # noqa: E402
from NSFopen.read import NHFFile, nhf_spec, read  # noqa: E402

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "NSFopen" / "example"
SAPPHIRE = str(EXAMPLES / "Calculate_Roughness" / "sapphire.nid")


def _int32(values, low, high):
    """Physical values to the raw int32 values of a 2.x file."""
    raw_low, raw_high = -(2.0 ** 31) + 1, 2.0 ** 31 - 1
    return np.round((np.asarray(values, float) - low) / (high - low) * (raw_high - raw_low) + raw_low).astype(np.int32)


def _channel(group, key, name, values, low=-1e-5, high=1e-5, unit="meter (m)", dtype=np.int32, block=None):
    fill = np.iinfo(dtype).min
    data = group.create_dataset(key, data=values, dtype=dtype, fillvalue=fill)
    data.attrs.update({"signal_name": name, "signal_id": name.lower(), "signal_selected": name.lower(),
                       "signal_calibration_min": low, "signal_calibration_max": high,
                       "signal_calibration_unit": unit, "signal_unit": "integer (int)",
                       "dataset_element_type": dtype().dtype.name})
    if block:
        data.attrs["dataset_block_size_source"] = block
    return data


def _root(path):
    f = h5py.File(path, "w")
    f.attrs["nsf_file_version_major"] = np.uint32(2)
    f.attrs["nsf_file_version_minor"] = np.uint32(3)
    return f


def _image_file(path, nx=8, ny=4, direction="up", stop_after=None):
    """Row r, column c has the height r * 1e-9 + c * 1e-10."""
    with _root(path) as f:
        g = f.create_group("group_0000")
        g.attrs.update({"group_type": "image_line_based", "measurement_name": "Image 1", "scan_direction": direction,
                        "rect_axis_range": [ny * 1e-7, nx * 1e-7]})
        s = g.create_group("subgroup_0000")
        s.attrs.update({"segment_name": "Forward", "rect_axis_size": [ny, nx]})
        z = np.add.outer(np.arange(ny) * 1e-9, np.arange(nx) * 1e-10).ravel()
        raw = _int32(z, -1e-5, 1e-5)
        if stop_after is not None:
            raw[stop_after:] = np.iinfo(np.int32).min
        _channel(s, "dataset_0000", "Z-Axis", raw)


def _spec_file(path, nx=3, ny=2, counts=None):
    """Curve n holds n + its sample number in meters * 1e-9, with snake_bottom_left order; lengths may differ."""
    number = nx * ny
    counts = list(counts) if counts else [5] * number
    with _root(path) as f:
        g = f.create_group("group_0000")
        g.attrs.update({"group_type": "spectroscopy", "measurement_name": "Grid", "rect_axis_size": [ny, nx],
                        "rect_axis_range": [ny * 1e-6, nx * 1e-6],
                        "spm_probe_calibration_deflection_sensitivity": 2e-8,
                        "spm_probe_calibration_spring_constant": 0.5})
        s = g.create_group("subgroup_0000")
        s.attrs["segment_name"] = "Retract 1"
        z = np.concatenate([(n + np.arange(c)) * 1e-9 for n, c in enumerate(counts)])
        _channel(s, "dataset_0000", "Position Z", _int32(z, -1e-5, 1e-5), block="B").attrs[
            "signal_data_pattern"] = "snake_bottom_left"
        _channel(s, "dataset_0001", "Deflection", _int32(z * 1e8, -5.0, 5.0), -5.0, 5.0, "volt (V)", block="B").attrs[
            "signal_data_pattern"] = "snake_bottom_left"
        sizes = s.create_dataset("dataset_0002", data=np.array(counts, np.uint64))
        sizes.attrs["dataset_block_size_id"] = "B"


def _nma_file(path, nx=3, ny=2, samples=20, old_names=False):
    """Free wave: a cosine with a level of 0.5 V. Pixel n has an interaction of (n + 1) * 0.01 V on every sample and a
    deflection baseline of 0.2 V."""
    wave = np.cos(2 * np.pi * np.arange(samples) / samples) + 0.5
    with _root(path) as f:
        g = f.create_group("group_0000")
        g.attrs.update({"group_type": "wavemode_nma", "measurement_name": "NMA", "rect_axis_range": [1e-6, 1e-6],
                        "scan_direction": "up", "spm_probe_calibration_deflection_sensitivity": 2e-8,
                        "spm_probe_calibration_spring_constant": 0.5, "wavemode_frequency": 1e4,
                        "wavemode_free_wave_amplitude_correction": 2.0, "wavemode_free_wave_phase_delay": 0.0})
        low, high = -5.0, 5.0
        scale = (2.0 ** 15 - 1) / high
        _channel(g, "dataset_0000", "NMA Free Wave" if old_names else "WaveMode Free Wave",
                 np.round(wave * scale).astype(np.int16), low, high, "volt (V)", np.int16)
        _channel(g, "dataset_0001", "WaveMode Free Wave Baseline", np.round(np.array([0.5]) * scale).astype(np.int16),
                 low, high, "volt (V)", np.int16)
        for k, scan in enumerate(("Forward", "Backward")):
            s = g.create_group(f"subgroup_000{k}")
            s.attrs.update({"segment_name": scan, "rect_axis_size": [ny, nx]})
            _channel(s, "dataset_0000", "Topography", _int32(np.arange(nx * ny) * 1e-9, -1e-5, 1e-5))
            fast = np.concatenate([wave - 0.5 + 0.2 + (n + 1) * 0.01 for n in range(nx * ny)])
            _channel(s, "dataset_0002", "NMA Deflection Baseline", np.round(np.full(nx * ny, 0.2) * scale).astype(np.int16),
                     low, high, "volt (V)", np.int16)
            _channel(s, "dataset_0001", "NMA Deflection" if old_names else "Fast In Deflection (16bit)",
                     np.round(fast * scale).astype(np.int16), low, high, "volt (V)", np.int16)


class Files(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def path(self, name):
        return os.path.join(self.folder.name, name)


class TestNHFImages(Files):
    def test_axis_order_of_non_square_image(self):
        path = self.path("a.nhf")
        _image_file(path, nx=8, ny=4)
        with read(path) as afm:
            z = afm.data["Image"]["Forward"]["Z-Axis"]
            self.assertEqual(z.shape, (4, 8))
            self.assertAlmostEqual(z[2, 3], 2e-9 + 3e-10, delta=1e-14)
            self.assertAlmostEqual(afm.param.size_x, 8e-7)
            self.assertAlmostEqual(afm.param.size_y, 4e-7)
            self.assertEqual(afm.units[("Image", "Forward", "Z-Axis")], "m")

    def test_stopped_scan_is_nan(self):
        path = self.path("a.nhf")
        _image_file(path, nx=4, ny=4, stop_after=6)
        with read(path) as afm:
            z = afm.data["Image"]["Forward"]["Z-Axis"]
        self.assertEqual(np.isnan(z).sum(), 10)
        self.assertTrue(np.isfinite(z[0]).all())

    def test_calibration_graph_selects_signal_and_unit(self):
        path = self.path("a.nhf")
        with _root(path) as f:
            g = f.create_group("group_0000")
            g.attrs.update({"group_type": "image_line_based", "measurement_name": "I",
                            "rect_axis_range": [1e-6, 1e-6]})
            s = g.create_group("subgroup_0000")
            s.attrs.update({"segment_name": "Forward", "rect_axis_size": [1, 2]})
            d = _channel(s, "dataset_0000", "Deflection", np.array([10, 20], np.int32), 0.0, 1.0, "volt (V)")
            d.attrs["signal_selected"] = "force"
            c = f.create_group("group_0001")
            c.attrs["group_type"] = "calibration"
            for key, source, sink, unit0, unit1, a, b in (("n0", "deflection", "volt", "integer (int)", "volt (V)", 2.0, 0.0),
                                                          ("n1", "volt", "force", "volt (V)", "newton (N)", 3.0, 1.0)):
                n = c.create_dataset(key, data=0)
                n.attrs.update({"signal_calibration_id": key, "signal_source_id": source, "signal_sink_id": sink,
                                "signal_source_unit": unit0, "signal_sink_unit": unit1,
                                "signal_mapping_type": "linear", "signal_mapping_parameters": [a, b],
                                "signal_invertibility": "right"})
        with NHFFile(path) as nhf:
            channel = nhf.select().segments["Forward"].channels["Deflection"]
            np.testing.assert_allclose(channel.read(), [10 * 2 * 3 + 1, 20 * 2 * 3 + 1])
            self.assertEqual(channel.unit, "N")
            np.testing.assert_allclose(channel.read(unit="V"), [20, 40])
            with self.assertRaises(ValueError):
                channel.read(unit="Hz")


class TestSpectroscopy(Files):
    def test_snake_order_and_lazy_access(self):
        path = self.path("s.nhf")
        _spec_file(path)
        with read(path) as afm:
            spec = afm.spec
            self.assertEqual(spec.shape, (2, 3))
            self.assertEqual(spec.n_points, 6)
            channel = afm.data["Spec"]["Retract 1"]["Position Z"]
            first = lambda point: channel[point][0] * 1e9
            self.assertAlmostEqual(first(4), 4, places=4)
            self.assertAlmostEqual(first((0, 2)), 2, places=4)
            self.assertAlmostEqual(first((1, 2)), 3, places=4)     # line 1 runs from right to left
            self.assertAlmostEqual(first((1, 0)), 5, places=4)
            self.assertEqual(np.asarray(channel).shape, (2, 3, 5))
            self.assertAlmostEqual(channel.array()[1, 2, 0] * 1e9, 3, places=4)
            self.assertAlmostEqual(channel.last()[1, 0] * 1e9, 9, places=4)

    def test_curves_of_different_length(self):
        path = self.path("s.nhf")
        _spec_file(path, counts=[2, 3, 4, 5, 6, 7])
        with read(path) as afm:
            channel = afm.spec["Retract 1"]["Position Z"]
            self.assertEqual([c.size for c in channel.curves()], [2, 3, 4, 5, 6, 7])
            self.assertEqual(channel[3].size, 5)
            padded = channel.array()
            self.assertEqual(padded.shape, (2, 3, 7))
            self.assertTrue(np.isnan(padded[0, 0, 2:]).all())

    def test_force_curve_uses_probe_calibration(self):
        path = self.path("s.nhf")
        _spec_file(path)
        with read(path) as afm:
            z, force = afm.spec.force_curve("Retract 1", 1)
            volt = afm.spec["Retract 1"]["Deflection"][1]
        np.testing.assert_allclose(force, volt * 2e-8 * 0.5)
        self.assertEqual(z.size, 5)

    def test_incomplete_grid_warns(self):
        path = self.path("s.nhf")
        _spec_file(path, nx=3, ny=3, counts=[5, 5, 5, 5])
        with self.assertWarns(UserWarning):
            afm = read(path)
        with self.assertRaises(IndexError):
            afm.spec["Retract 1"]["Position Z"][2, 2]
        self.assertTrue(np.isnan(afm.spec["Retract 1"]["Position Z"].array()[2, 2]).all())
        afm.close()

    def test_nhf_spec_interface(self):
        path = self.path("s.nhf")
        _spec_file(path)
        afm = nhf_spec(path)
        self.assertEqual(afm.seg_names, ["Retract 1"])
        self.assertEqual(afm.param.number, 6)
        self.assertEqual(afm.data("Retract 1", "Deflection", 2).size, 5)
        afm._afm.close()


class TestWaveModeNMA(Files):
    def check(self, old_names):
        path = self.path("n.nhf")
        _nma_file(path, old_names=old_names)
        with read(path) as afm:
            nma = afm.nma
            self.assertEqual((nma.samples, nma.n_points, nma.shape), (20, 6, (2, 3)))
            self.assertAlmostEqual(nma.sampling_rate, 2e5)
            curve = nma.curve((1, 2))                     # pixel 5, no snake order in images
            np.testing.assert_allclose(curve.interaction, 6 * 0.01 * 2.0, atol=1e-3)
            np.testing.assert_allclose(curve.force, curve.interaction * 2e-8 * 0.5)
            self.assertEqual(curve.turn, 10)              # the cosine is lowest in the middle
            self.assertEqual(curve.advance[0].size + curve.retract[0].size, 20)
            np.testing.assert_allclose(nma.curve((1, 2), baseline=False).interaction, (0.06 - 0.3) * 2.0, atol=1e-3)
            self.assertAlmostEqual(curve.z.min(), -0.5 * 2e-8, delta=1e-11)
            np.testing.assert_allclose(nma.map(scan="Backward"), np.arange(1, 7).reshape(2, 3) * 0.02, atol=1e-3)
            self.assertNotIn(nma.fast_channel, afm.data["Image"]["Forward"].index)
            self.assertIn("Topography", afm.data["Image"]["Forward"].index)

    def test_current_names(self):
        self.check(False)

    def test_older_names(self):
        self.check(True)


class TestProcess(unittest.TestCase):
    def setUp(self):
        self.afm = process(SAPPHIRE)

    def test_inplace_false_leaves_channel_alone(self):
        before = self.afm.array().copy()
        out = self.afm.flatten(inplace=False)
        np.testing.assert_array_equal(self.afm.array(), before)
        self.assertFalse(np.array_equal(out, before))
        self.assertIsNone(self.afm.flatten())
        np.testing.assert_allclose(self.afm.array(), out)

    def test_every_operation_method_exists(self):
        for name in process.operations():
            self.assertTrue(callable(getattr(self.afm, name.replace(".", "_"))), name)

    def test_axis_keyword_reaches_the_operation(self):
        rows = self.afm.level_lines(inplace=False, axis="rows")
        columns = self.afm.level_lines(inplace=False, axis="columns")
        self.assertFalse(np.array_equal(rows, columns))
        np.testing.assert_allclose(np.median(rows, axis=1).std(), 0, atol=1e-9)

    def test_mask_drives_fit_and_statistics(self):
        self.afm.flatten()
        self.afm.mask_threshold(mode="above", low=90.0, unit="percentile")
        mask = self.afm.masks[("Forward", "Z-Axis")]
        self.assertAlmostEqual(mask.mean(), 0.1, delta=0.01)
        table = self.afm.statistics(region="unmasked", texture=False)
        self.assertEqual(table.loc["Pixels", "value"], (~mask).sum())
        self.assertEqual(table.loc["Sa", "unit"], "m")
        self.assertEqual(table["value"].dtype, float)  # else pandas prints small heights as 0.0
        self.assertNotIn(" 0.0 ", table.loc[["Sa"]].to_string())

    def test_grains_need_a_mask_and_table_matches(self):
        with self.assertRaises(ValueError):
            self.afm.grains()
        self.afm.flatten()
        self.afm.mask_otsu()
        result = self.afm.grains(min_pixels=4)
        self.assertEqual(len(result.table), result.labels.max())
        self.assertEqual(result.labels.shape, self.afm.array().shape)
        self.assertEqual((result.labels > 0).sum(), result.table["Pixels"].sum())

    def test_line_profile_physical_and_pixel_coordinates_agree(self):
        x, y = self.afm.size()
        ny, nx = self.afm.array().shape
        by_m = self.afm.line_profile(((0.5 / nx) * x, (10.5 / ny) * y), ((nx - 0.5) / nx * x, (10.5 / ny) * y))
        by_px = self.afm.line_profile((0, 10), (nx - 1, 10), unit="px")
        self.assertAlmostEqual(by_m.length, by_px.length)
        self.assertAlmostEqual(by_m.mean[0], by_px.mean[0], delta=1e-15)
        self.assertAlmostEqual(by_m.mean[-1], by_px.mean[-1], delta=1e-15)
        np.testing.assert_allclose(by_px.mean, self.afm.array()[10], atol=1e-12)

    def test_crop_updates_size(self):
        x, _ = self.afm.size()
        nx = self.afm.array().shape[1]
        self.afm.geometry_crop(x0=0, y0=0, x1=nx // 2, y1=nx // 2)
        self.assertEqual(self.afm.array().shape, (nx // 2, nx // 2))
        self.assertAlmostEqual(self.afm.size()[0], x / 2)

    def test_bad_parameter_is_an_error_with_a_message(self):
        with self.assertRaises(OpError):
            self.afm.apply("level.plane", nonsense=1)

    def test_plots_return_figures(self):
        import matplotlib.pyplot as plt

        self.afm.mask_otsu()
        for fig in (self.afm.image(cbar="Z [nm]", show_mask=True), self.afm.image(scalebar=True, theme="dark"),
                    self.afm.thumb(), self.afm.hist(), self.afm.plot3d(max_points=32, dpi=72)):
            self.assertIsInstance(fig, plt.Figure)
            plt.close(fig)

    @staticmethod
    def _picture(fig):
        import matplotlib.pyplot as plt

        picture = fig.axes[0].images[0].get_array().copy()
        plt.close(fig)
        return picture

    def test_plot3d_seen_from_above_is_the_picture_of_image(self):
        """ A bump at the lower left of the picture, on a wider than high channel, stays there in the 3D view. """
        afm = process(SAPPHIRE)
        ny, nx = 64, 128
        z = np.zeros((ny, nx))
        z[8:16, 16:32] = 1e-9                               # row 0 is the bottom for an upward scan
        z = z if afm._row0_is_bottom else z[::-1]
        afm._store("Forward", "Z-Axis", z)
        afm._sizes[("Forward", "Z-Axis")] = (2e-6, 1e-6)
        picture = self._picture(afm.plot3d(tilt=0, rotation=0, perspective=False, axes=False, colorbar=False, dpi=72,
                                           cmap="gray", figsize=(6, 4)))
        drawn = np.argwhere(picture.min(axis=2) < 0.99)     # gray 0 is the floor, the background is white
        rows, columns = picture.shape[:2]
        top, left = drawn.min(axis=0)
        bottom, right = drawn.max(axis=0)
        self.assertAlmostEqual((right - left) / (bottom - top), 2.0, delta=0.05)    # the proportions of the image
        bump = np.argwhere(picture.min(axis=2) > 0.99)
        bump = bump[(bump[:, 0] > top) & (bump[:, 0] < bottom) & (bump[:, 1] > left) & (bump[:, 1] < right)]
        center = bump.mean(axis=0)
        self.assertAlmostEqual((center[1] - left) / (right - left), 24 / 128, delta=0.02)
        self.assertAlmostEqual((bottom - center[0]) / (bottom - top), 12 / 64, delta=0.02)

    def test_plot3d_options(self):
        channel = "Z-Axis"
        options = dict(max_points=48, dpi=72, figsize=(6, 4))
        self.afm.mask_otsu()
        heights = self._picture(self.afm.plot3d(**options))
        lit = self._picture(self.afm.plot3d(material="gold", **options))
        self.assertGreater(np.abs(heights - lit).mean(), 0.01)
        self.assertGreater(lit[..., 0].mean(), lit[..., 2].mean())                  # gold is not blue
        masked = self._picture(self.afm.plot3d(show_mask=True, **options))
        self.assertGreater(np.abs(masked - heights).mean(), 0.001)
        other = self._picture(self.afm.plot3d(color=("Backward", channel), cmap="Gold", **options))
        self.assertGreater(np.abs(other - heights).mean(), 0.01)
        dark = self._picture(self.afm.plot3d(theme="dark", **options))
        self.assertEqual(tuple(dark[0, 0]), (0, 0, 0))
        for bad in (dict(material="Wood"), dict(mode="sparkle"), dict(mode="overlay"), dict(cmap="Nothing")):
            with self.assertRaises(ValueError):
                self.afm.plot3d(**bad, **options)

    def test_plot3d_leaves_out_nan_pixels(self):
        afm = process(SAPPHIRE)
        z = afm.array().copy()
        z[:, :] = 0.0
        z[z.shape[0] // 2 - 20:z.shape[0] // 2 + 20, z.shape[1] // 2 - 20:z.shape[1] // 2 + 20] = np.nan
        afm._store("Forward", "Z-Axis", z)
        picture = self._picture(afm.plot3d(tilt=0, perspective=False, rotation=0, axes=False, colorbar=False, dpi=72,
                                           cmap="gray", max_points=64, figsize=(4, 4)))
        self.assertEqual(tuple(picture[picture.shape[0] // 2, picture.shape[1] // 2]), (1, 1, 1))
        self.assertLess(picture.min(), 0.5)

    def test_channels_and_the_default_height_channel(self):
        afm = process(str(EXAMPLES / "Plane_Removal" / "circles.nid"))          # has Topography, not Z-Axis
        self.assertIn("Topography", afm.channels["Forward"])
        np.testing.assert_array_equal(afm.array(), afm.array("Forward", "Topography"))
        afm.mask_otsu()
        self.assertTrue(afm.masks[("Forward", "Topography")].any())             # one mask, by either name
        np.testing.assert_array_equal(afm._state("Forward", "Z-Axis").mask, afm.masks[("Forward", "Topography")])
        with self.assertRaisesRegex(ValueError, "Amplitude"):                   # the error names what there is
            afm.array("Forward", "Nothing")
        with self.assertRaisesRegex(ValueError, "Forward"):
            afm.array("Sideways", "Amplitude")

    def test_plot_units_follow_the_channel(self):
        import matplotlib.pyplot as plt

        afm = process(str(EXAMPLES / "Plane_Removal" / "grid.nid"))
        fig = afm.hist(channel="Amplitude")
        self.assertRegex(fig.axes[0].get_xlabel(), r"^Amplitude \[[mµ]?V\]$")
        plt.close(fig)
        fig = afm.image(channel="Amplitude")
        self.assertLess(abs(fig.axes[0].images[0].get_clim()[1]), 1e4)           # readable numbers, not 1e9 V
        self.assertRegex(fig.axes[0].child_axes[0].get_ylabel(), "Amplitude")
        plt.close(fig)
        fig = afm.image(scale=1e9, cbar="Z [nm]", colorbar=False)                # the older way still works
        self.assertEqual(fig.axes[0].child_axes[0].get_ylabel(), "Z [nm]")
        plt.close(fig)
        fig = afm.image(colorbar=False)
        self.assertEqual(len(fig.axes[0].child_axes), 0)
        plt.close(fig)

    def test_image_color_range(self):
        import matplotlib.pyplot as plt

        afm = process(str(EXAMPLES / "Plane_Removal" / "grid.nid"))
        z = afm.array("Forward", "Z-Axis")
        fig = afm.image(color_range=(z.min(), z.mean()), scale=1.0)
        self.assertEqual(fig.axes[0].images[0].get_clim(), (z.min(), z.mean()))
        plt.close(fig)
        fig = afm.image(color_range="p1", scale=1.0)
        self.assertAlmostEqual(fig.axes[0].images[0].get_clim()[1], np.percentile(z, 99))
        plt.close(fig)
        fig = afm.hist(color="r")                                                # single color, as in matplotlib
        plt.close(fig)
        fig = afm.hist(bins=1)
        plt.close(fig)

    def test_scalebar_labels(self):
        import matplotlib.pyplot as plt

        afm = process(str(EXAMPLES / "Plane_Removal" / "grid.nid"))
        for size, expected in ((1e-6, "200 nm"), (2e-6, "200 nm"), (2.5e-6, "500 nm"), (3e-6, "500 nm"), (5e-6, "1 µm"),
                               (10e-6, "2 µm"), (50e-9, "10 nm")):
            for name in ("Z-Axis",):
                afm._sizes[("Forward", name)] = (size, size)
            fig = afm.image(scalebar=True)
            self.assertEqual(fig.axes[0].texts[0].get_text(), expected, size)
            plt.close(fig)
        afm._sizes[("Forward", "Z-Axis")] = (10e-6, 10e-6)
        fig = afm.image(scalebar={"length": 1.5, "color": "w"})
        self.assertEqual(fig.axes[0].texts[0].get_text(), "1.5 µm")
        plt.close(fig)
        fig = afm.image(scalebar={"length": 0.0005})
        self.assertEqual(fig.axes[0].texts[0].get_text(), "500 pm")
        plt.close(fig)

    def test_plots_draw_on_given_axes(self):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=72)
        count = len(plt.get_fignums())
        afm = process(str(EXAMPLES / "Plane_Removal" / "grid.nid"))
        for result in (afm.image(ax=axes[0, 0]), afm.hist(ax=axes[0, 1]), afm.thumb(ax=axes[1, 0]),
                       afm.plot3d(ax=axes[1, 1], max_points=32, dpi=72)):
            self.assertIs(result, fig)
        self.assertEqual(len(plt.get_fignums()), count)                          # no new figures
        self.assertEqual(sum(bool(a.images) for a in axes.ravel()), 3)
        self.assertTrue(axes[0, 1].collections)
        plt.close("all")

    def test_plot3d_keeps_its_proportions_when_the_layout_moves_the_axes(self):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=72, layout="constrained")
        process(str(EXAMPLES / "Plane_Removal" / "grid.nid")).plot3d(ax=axes[1], max_points=32, dpi=72)
        fig.canvas.draw()
        picture = axes[1].images[0].get_array()
        box = axes[1].get_position()
        size = fig.get_size_inches()
        self.assertAlmostEqual((box.width * size[0]) / (box.height * size[1]), picture.shape[1] / picture.shape[0], delta=0.02)
        self.assertTrue(axes[1].child_axes)                                      # the color bar stays on the view
        plt.close(fig)

    def test_analysis_plots_show_the_analysis(self):
        import matplotlib.pyplot as plt

        afm = process(str(EXAMPLES / "Plane_Removal" / "grid.nid"))
        afm.level_flatten()
        x, y = afm.size()
        profile = afm.line_profile((0, y / 2), (x, y / 2), width=3)
        np.testing.assert_allclose(profile.start, (0.5 * x / 256, y / 2), atol=x / 256)
        fig = afm.image(profile=profile)
        drawn = fig.axes[0].lines[0]
        self.assertAlmostEqual(drawn.get_ydata()[0], y / 2 * 1e6, delta=0.02)   # in um, across the image
        self.assertEqual(fig.axes[0].get_xlim(), (0.0, 3.0))
        plt.close(fig)

        fig = process.plot_profile(profile)
        line = fig.axes[0].lines[0]
        np.testing.assert_allclose(line.get_ydata(), profile.mean * 1e9)
        self.assertEqual(fig.axes[0].get_ylabel(), "Height [nm]")
        plt.close(fig)

        fig = afm.plot_bearing()
        heights = fig.axes[0].lines[0].get_ydata()
        self.assertGreater(heights[0], heights[-1])                             # highest first
        plt.close(fig)

        fig = afm.plot_histogram_fit(model="gauss2")
        self.assertEqual(len(fig.axes[0].lines), 3)                             # the fit and its two peaks
        self.assertEqual(fig.axes[0].get_xlabel(), "Z-Axis [nm]")
        plt.close(fig)

        fig = afm.plot_psd()
        ax = fig.axes[0]
        self.assertEqual((ax.get_xscale(), ax.get_yscale()), ("log", "log"))
        self.assertEqual(len(ax.lines), 2)
        plt.close(fig)
        fig = afm.plot_psd(fit=None)
        self.assertEqual(len(fig.axes[0].lines), 1)
        plt.close(fig)

        fig = afm.plot_acf()
        image = fig.axes[0].images[0]
        ny, nx = image.get_array().shape
        self.assertAlmostEqual(float(image.get_array()[ny // 2, nx // 2]), 1.0, places=6)   # 1 at zero shift
        plt.close(fig)

    def test_plot_profile_draws_steps(self):
        import matplotlib.pyplot as plt

        afm = process(SAPPHIRE)
        ny, nx = afm.array().shape
        rng = np.random.default_rng(0)
        afm._store("Forward", "Z-Axis", np.where(np.arange(nx) < nx // 2, 0.0, 5e-9) + rng.normal(0, 2e-11, (ny, nx)))
        profile = afm.line_profile((0, 10), (nx - 1, 10), unit="px", width=5)
        steps = process.step_heights(profile)
        fig = process.plot_profile(profile, steps=steps)
        self.assertEqual(len(fig.axes[0].lines), 1 + len(steps.lines))
        plt.close(fig)

    def test_feret_of_known_shapes(self):
        import math

        from NSFopen.process.grains import _feret

        square = np.zeros((12, 12), bool)
        square[1:11, 1:11] = True
        ring = square.copy()
        ring[4:8, 4:8] = False                                                  # a hole changes nothing outside
        for region in (square, ring):
            high, _, low, _, hull = _feret(region, 0.0, 0.0, 1.0, 1.0)
            self.assertAlmostEqual(high, 10 * math.sqrt(2))
            self.assertAlmostEqual(low, 10.0)
            self.assertAlmostEqual(hull, 100.0)
        line = np.zeros((3, 8), bool)
        line[1, 1:7] = True
        high, direction, low, _, hull = _feret(line, 0.0, 0.0, 2.0, 1.0)         # pixels 2 wide, 1 high
        self.assertAlmostEqual(high, math.hypot(12, 1))
        self.assertAlmostEqual(low, 1.0)
        self.assertAlmostEqual(hull, 12.0)

    def test_cut_nid_file_says_so(self):
        import os
        import tempfile

        with open(str(EXAMPLES / "Plane_Removal" / "circles.nid"), "rb") as source:
            content = source.read()
        with tempfile.TemporaryDirectory() as folder:
            cut = os.path.join(folder, "cut.nid")
            with open(cut, "wb") as target:
                target.write(content[:len(content) // 2])
            with self.assertRaisesRegex(ValueError, "shorter"):
                process(cut)

    def test_size_is_that_of_the_image_in_a_file_with_spectroscopy(self):
        afm = process(str(EXAMPLES / "Force_Spectroscopy" / "spectroscopy.nid"))   # its first frames are curves
        self.assertEqual(afm.size(), (5e-6, 5e-6))

    def test_own_arrays_become_channels(self):
        afm = process(str(EXAMPLES / "Plane_Removal" / "grid.nid"))
        z = afm.array()
        afm.set_channel(z * 2, channel="Double", unit="m")
        np.testing.assert_array_equal(afm.array(channel="Double"), z * 2)
        self.assertEqual(afm.unit(channel="Double"), "m")
        self.assertIn("Double", afm.channels["Forward"])
        afm.level_flatten(channel="Double")                                      # usable like any other
        with self.assertRaisesRegex(ValueError, "shape"):
            afm.set_channel(z[:10], channel="Short")
        self.assertIn(afm.origin, ("lower", "upper"))

    def test_combined_channel_keeps_the_unit(self):
        afm = process(str(EXAMPLES / "Lateral_Force" / "SBS-PMMA.nid"))
        channel = "Friction force"
        afm.combine(("Forward", channel), ("Backward", channel), name="Friction")
        self.assertEqual(afm.unit(channel="Friction"), afm.unit(channel=channel))
        self.assertEqual(afm.unit(channel=channel), "V")

    def test_nid_header_is_read_as_utf8(self):
        afm = process(str(EXAMPLES / "Plane_Removal" / "grid.nid"))
        self.assertEqual(afm.unit(channel="Phase"), "°")

    def test_colors(self):
        self.assertEqual(colors("BlueBlackRed").cmap.name, "BlueBlackRed")
        self.assertEqual(colors("Gold").cmap.name, "Gold")


class TestAnalysisOnRealImage(unittest.TestCase):
    """Every analysis method runs on a periodic grating and gives a sensible result."""

    @classmethod
    def setUpClass(cls):
        cls.afm = process(str(EXAMPLES / "Plane_Removal" / "grid.nid"))
        cls.afm.level_flatten()
        cls.x, cls.y = cls.afm.size()

    def test_statistics_and_distributions(self):
        table = self.afm.statistics()
        self.assertGreater(table.loc["Sq", "value"], table.loc["Sa", "value"])
        self.assertGreater(table.loc["Sal", "value"], 0)
        percent, height = self.afm.bearing_curve()
        self.assertEqual(percent.size, height.size)
        self.assertTrue((np.diff(height) <= 0).all())
        self.assertIn("inclination", self.afm.slope_distributions())
        self.assertGreater(len(self.afm.histogram_fit(model="gauss2").peaks), 1)

    def test_spectra_and_correlation(self):
        k, power = self.afm.psd()
        self.assertEqual(k.size, power.size)
        fit = self.afm.psd_fit(model="broken")
        self.assertIn("Transition", fit.summary())
        acf, tau_x, tau_y = self.afm.acf()
        self.assertEqual(acf.shape, (tau_y.size, tau_x.size))
        self.assertGreater(self.afm.line_functions(axis="y")["tau"].size, 1)
        self.assertGreater(len(self.afm.rms_by_scale([8, 16])), 0)
        amplitude, kx, ky = self.afm.spectrum()
        self.assertEqual(amplitude.shape, (ky.size, kx.size))
        self.afm.filter_notch(inplace=False, points=[(kx[-5], ky[-5])], radius=abs(kx[1] - kx[0]))

    def test_lines(self):
        profile = self.afm.line_profile((0, self.y / 2), (self.x, self.y / 2), width=3)
        self.assertAlmostEqual(profile.length, self.x)
        self.assertAlmostEqual(profile.angle, 0.0)
        rough = process.profile_roughness(profile)
        self.assertGreater(rough.params["Rq"][0], 0)

    def test_step_and_critical_dimension_of_known_shapes(self):
        afm = process(SAPPHIRE)
        ny, nx = afm.array().shape
        rng = np.random.default_rng(0)
        z = np.where(np.arange(nx) < nx // 2, 0.0, 5e-9) + rng.normal(0, 2e-11, (ny, nx))
        afm._store("Forward", "Z-Axis", z)
        profile = afm.line_profile((0, 10), (nx - 1, 10), unit="px", width=5)
        steps = process.step_heights(profile).steps
        self.assertEqual(len(steps), 1)
        self.assertAlmostEqual(steps[0].height, 5e-9, delta=1e-10)

        period = nx // 4
        z = np.where((np.arange(nx) % period) < period // 2, 0.0, 5e-9) + rng.normal(0, 2e-11, (ny, nx))
        afm._store("Forward", "Z-Axis", z)
        profile = afm.line_profile((0, 10), (nx - 1, 10), unit="px", width=5)
        cd = process.critical_dimension(profile)
        self.assertAlmostEqual(cd.height, 5e-9, delta=1e-10)
        self.assertAlmostEqual(cd.pitch, period * afm.size()[0] / nx, delta=afm.size()[0] / nx)

    def test_filters_and_glitches_keep_the_shape(self):
        shape = self.afm.array().shape
        for call in (self.afm.filter_median, self.afm.filter_gaussian, self.afm.filter_basic, self.afm.glitch_spikes,
                     self.afm.glitch_scars, self.afm.glitch_auto_lines, self.afm.level_median):
            self.assertEqual(call(inplace=False).shape, shape, call.__name__)
        self.assertEqual(self.afm.filter_fft(inplace=False, cutoff=self.x / 10).shape, shape)

    def test_second_channel_and_new_channel(self):
        from scipy.ndimage import gaussian_filter

        afm = process(SAPPHIRE)
        channel = "Z-Axis"
        z = gaussian_filter(np.random.default_rng(0).normal(size=afm.array().shape), 3.0) * 1e-9
        afm._store("Forward", channel, z)
        afm._store("Backward", channel, np.roll(z, -3, axis=1))     # the backward scan lies 3 pixels to the left
        dx, dy, score = afm.find_shift(("Forward", channel), ("Backward", channel))
        self.assertAlmostEqual(dx, 3.0, delta=0.1)
        self.assertAlmostEqual(dy, 0.0, delta=0.1)
        self.assertGreater(score, 0.5)
        half = afm.combine(("Forward", channel), ("Backward", channel), name="Half")
        np.testing.assert_array_equal(afm.array("Forward", "Half"), half)
        scaled = afm.rescale_z_to_reference(inplace=False, reference=("Backward", channel))
        self.assertEqual(scaled.shape, half.shape)


class TestProcessNHF(Files):
    def test_non_square_image_and_down_scan(self):
        path = self.path("a.nhf")
        _image_file(path, nx=8, ny=4, direction="down")
        afm = process(path)
        self.assertEqual(afm.size(), (8e-7, 4e-7))
        afm.plane_level()
        np.testing.assert_allclose(afm.array(), 0, atol=1e-13)
        self.assertFalse(afm._row0_is_bottom)
        afm.afm.close()

    def test_rotation_is_as_seen_in_the_picture(self):
        path = self.path("a.nhf")
        _image_file(path, nx=8, ny=4)
        afm = process(path)
        before = afm.array().copy()
        afm.geometry_rotate(turn="cw")
        # upward scan: row 0 is the bottom, so the left column of the picture becomes its top line
        np.testing.assert_array_equal(afm.array()[-1], before[:, 0])
        self.assertEqual(afm.size(), (4e-7, 8e-7))
        afm.afm.close()


if __name__ == "__main__":
    unittest.main()
