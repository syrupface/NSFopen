# Change log

## 2.4.0 (not yet released)

Compared to **2.3.0**.

### Force spectroscopy analysis
Added, in `NSFopen.process` (the calculations of the Spectroscopy page of NSSA, the NSF Surface and Spectroscopy
Analyzer, so a script and the program give the same numbers):
- `contact`: elastic contact of a tip pressed into a flat sample. Models Hertz, Sneddon (exact), DMT, JKR and
  Oliver-Pharr; tip shapes sphere, cone, four-sided pyramid, flat cylinder punch and hyperboloid. `fit_elastic`,
  `fit_jkr`, `fit_oliver_pharr` give the Young's modulus, reduced modulus, contact point, adhesion and the fit quality.
- `spectroscopy`: `SpecFile` opens the force curves and force maps of NID and NHF files. `ForceSettings` and
  `ForceAnalysis` process them: smoothing, baseline (offset, tilt, or an interference sine with its period found),
  contact point, indentation, adhesion, pull-off, adhesion and dissipated energy, and the model fit. `Results` has
  statistics, maps, masks, fits of different models on different regions, a quality filter (leave out fits below an R²)
  and CSV export. Maps can be analyzed partly (every n-th line and column, or a region).
- `iv`: I-V curves with dI/dV, normalized conductance, zero-bias conductance and resistance, gap and rectification.

### Color maps
- 12 maps are defined in `colormaps.py` (`NANOSURF`): Sky (the default for heights), Warm, Flare, Caribbean, Strata,
  Gold, Digitalis, Neon, Plum, Cold, Spectrum and Maple, plus BlueBlackRed. `register()` adds them and their reversed
  versions (`Sky_r`) to matplotlib; `colormap(name)` returns any of them or a matplotlib map.
- The 59 Gwyddion gradient files that 2.3.0 included are removed: they are GPL, NSFopen is MIT. An installed Gwyddion's
  gradients are found on Windows, macOS and Linux (`gwyddion_dirs`, `gwyddion_gradients`, `register(gwyddion=True)`), and
  `colors(name)` uses them when the name is not one of the 12.

### Changes that can affect existing scripts
| What | 2.3.0 | 2.4.0 |
|---|---|---|
| `colors(name)` with a Gwyddion name | 59 gradient files in the package | 12 maps defined in the code; `Pm3d`, `Blend2`, `Code-V` open as `Flare`, `Strata`, `Spectrum`; the others only with an installed Gwyddion or `gwy_dir=` |
| `colormaps.GRADIENTS` | The gradient folder | Removed |

### Examples
- New `Force_Spectroscopy/Contact_Models.py`: a force map fitted with `process.spectroscopy`, with statistics, the
  quality filter, maps of the modulus and adhesion, and the stiffest curve with its fit.

### Packaging
- Credentials (`tokens.json`, `.env`, `*.token`) in the example folders are never packaged.

## 2.3.0 (November 2, 2026)

Compared to **2.2.4**, the latest version on PyPI (August 2024).

### Highlights
- `pip install NSFopen` now works by itself: the `nanosurf` and `scikit-learn` packages are no longer needed.
- NID files are read about 5 times faster, and line flattening is about twice as fast.
- NHF spectroscopy files of any size open immediately, and their curves are read when asked for.
- WaveMode NMA files are read, with a curve for every pixel and maps of any quantity.
- `process` has many new image analysis functions, brought over from the Nanosurf Surface Analyzer (now NSSA).
- Figures take their units and labels from the file, can be placed in subplots, and there is a new 3D plot.

### Reading files
Added:
- New NHF reader that needs only h5py, for file versions 1.1 and 2.0 to 2.6. Images are identical to those of Nanosurf's
  reader on versions 1.1, 2.0, 2.3, 2.4 and 2.6.
- Spectroscopy (`afm.spec`): segments, grid position, other units through the file's calibration, force curves.
- WaveMode NMA (`afm.nma`): `curve()` gives Z, interaction and force of a pixel; `map()` builds per-pixel maps.
- `afm.units` holds the unit of every channel. `afm.close()` releases an NHF file; `read` also works in a `with` block.
- `read(..., measurement=)` picks a measurement of an NHF file.

Fixed:
- 2.2.4 needed `nanosurf` to be imported but did not declare it, and failed on NHF files with `nanosurf` 2.0.7.
- NID headers were read as Latin-1, so units and sizes came out as `Â°` and `3Âµm`.
- Duplicate `Wait` segments of older NHF files overwrote each other: the second is now `Wait 2`.
- A third NID channel of the same name overwrote the second: they are now `Amplitude`, `Amplitude2`, `Amplitude3`.
- A truncated NID file now says so.

### Processing images
Added:
- Leveling: `level_flatten`, `level_poly2d`, `level_lines`, `level_align_rows`, `level_facet`, `level_three_point`,
  `level_median`, `level_rolling_ball`, `level_shift`.
- Filters: `filter_basic`, `filter_median`, `filter_gaussian`, `filter_fft`, `filter_notch`.
- Glitches: `glitch_spikes`, `glitch_lines`, `glitch_auto_lines`, `glitch_scars`, `glitch_step_lines`, `glitch_fill_mask`.
- Masks, kept per channel: `set_mask` and `mask_threshold`, `mask_otsu`, `mask_watershed`, `mask_outliers`, `mask_scars`,
  `mask_shape`, `mask_floodfill`, `mask_grow`, `mask_shrink`, `mask_invert`, `mask_clear`, `mask_remove_small`,
  `mask_fill_holes`. Leveling, filters and statistics can use the mask.
- Geometry and scale: `geometry_crop`, `geometry_flip`, `geometry_rotate`, `geometry_resample`, `geometry_square_pixels`,
  `rescale_scale`, `rescale_z_to_reference`, `rescale_xy`.
- Statistics after ISO 25178: `statistics`, `bearing_curve`, `histogram_fit`, `slope_distributions`.
- Lines: `line_profile`, `step_heights`, `critical_dimension`, `masked_step_height`, and `profile_roughness` after
  ISO 21920.
- Spectra and texture: `psd`, `psd_fit`, `spectrum`, `acf`, `line_functions`, `rms_by_scale`, `lattice`.
- Grain analysis: `grains`.
- Two channels: `combine`, `find_shift`.
- Access: `channels`, `array`, `set_channel`, `size`, `unit`, `origin`, and `apply` and `operations`.

Fixed:
- `NSFopen.process` could not be imported, because `colormaps.py` imported itself.
- `flatten(inplace=False)` and similar calls changed the data as well as returning the result.
- `plane_level` was wrong on non-square images.
- The physical sizes of non-square NHF images were swapped.

### Figures
Added:
- `plot3d`, a 3D view after the Surface Analyzer's, with materials, lighting and overlays of another channel. It needs no
  graphics card.
- `plot_profile`, `plot_psd`, `plot_histogram_fit`, `plot_bearing`, `plot_acf`.
- `ax=` to draw into subplots, and `figsize`, `dpi`, `theme` and `color_range` on the plots.
- `image(profile=)` shows where a line cut lies.
- Color maps by name, including the Gwyddion gradients now included with the package: `cmap='Gold'`.

Fixed:
- The default scale bar had zero length on scans of about 2.5 µm or less, and non-integer lengths were labelled wrongly.
- `hist(color=)` raised an error.

### Changes that can affect existing scripts
| What | 2.2.4 | 2.3.0 |
|---|---|---|
| NHF force curves `[row, col]` | Position in the file | True grid position, row 0 at the bottom |
| NHF force curves | Arrays | Read on access; `np.asarray(curves)` gives the array |
| `image` and `hist`, `scale` | `1e9` for every channel | A readable unit from the channel; `scale=1e9` still forces it |
| `hist`, `xlabel` | `Z-Axis [nm]` for every channel | The channel and its unit |
| `image`, color bar | Only with `cbar=` | Always; `colorbar=False` turns it off |
| `image`, pixels | matplotlib's default | `interpolation='nearest'`; can be overridden |
| `remove_outliers` | Nearest neighbors with scikit-learn | Interpolation from the pixels around; the numbers differ, and `neighbors` is gone |
| Unknown channel | pandas `KeyError` | `ValueError` that lists the channels |
| Default channel | `Z-Axis` only | Falls back to `Topography` in files that have no `Z-Axis` |
| `NSFopen.read.checkFileType` | Present | Removed |

`stats()` keeps its results, including `Sku` as excess kurtosis (0 for a Gaussian surface). The new `statistics()` gives
the ISO value of `Sku` (3 for a Gaussian surface).

### Examples
- All are rewritten around `process`, and there is a new `Grain_Analysis`.
- **`Spring_Constant`: the result changes from 14.5 to 24.0 N/m.** The old fit used a power formula on amplitude data and
  found Q = 253; the file records Nanosurf's own Q of 442, and the corrected fit finds 417.
- `Plotting_Data`: the image was drawn upside down.
- `Force_Spectroscopy`: the map axes were labelled µm but showed pixels.
- `Sharpen_Image`: it no longer passes the data through an 8-bit JPEG, and needs neither OpenCV nor PIL.
- `Plane_Removal`: it crashed on non-square images, and its clipping changed the data.
- `Lateral_Force`: the offset between the scans is measured instead of set by hand.
- Renamed without spaces: `Plotting_Data.py`, `Lateral_Force_Processing.py`, `Sharpen_Image.py`.

### Requirements
- matplotlib 3.6 or newer; the other requirements are unchanged (numpy, scipy, pandas, h5py, Python 3.9 or newer).
- Removed: `nanosurf` and `scikit-learn`, which 2.2.4 used without declaring them.

## Earlier versions
The earlier releases have no change notes. Their list and dates are on
[PyPI](https://pypi.org/project/NSFopen/#history).
