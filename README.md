# NSFopen
Open and analyze Nanosurf NID and NHF files in Python: images, force spectroscopy and WaveMode NMA.

## What's new in 2.4.1
A fix: reading NID files no longer gives a pandas warning (pandas 3.1 and newer).

## What's new in 2.4.0
Compared to 2.3.0:
- **Force curve analysis:** `process.spectroscopy` and `process.contact` fit force curves and force maps with Hertz,
  Sneddon, DMT, JKR and Oliver-Pharr for sphere, cone, pyramid, flat punch and hyperboloid tips, with baseline and
  interference removal, statistics, maps and CSV; `process.iv` analyzes I-V curves. The same calculations as the
  Spectroscopy page of NSSA (NSF Surface and Spectroscopy Analyzer).
- **Color maps of its own:** 12 named maps (`Sky`, `Warm`, `Flare`, ...) defined in the code; `register()` makes them
  usable by name in matplotlib. The Gwyddion gradient files are no longer included, but an installed Gwyddion's are
  found on Windows, macOS and Linux.

## What's new in 2.3.0
Compared to 2.2.4:
- **Works by itself:** the `nanosurf` and `scikit-learn` packages are no longer needed.
- **NHF spectroscopy and WaveMode NMA:** files of any size open immediately, and curves are read when asked for.
- **Image analysis from NSSA (then called Nanosurf Surface Analyzer):** leveling, filters, masks, ISO statistics, line
  cuts, step heights, critical dimension, PSD fits, lattice and grains.
- **Better figures:** units and labels from the file, subplots with `ax=`, a new 3D plot, and plots of the analysis results.
- **Examples rewritten,** with several bugs fixed. The `Spring_Constant` result changes from 14.5 to 24.0 N/m, because the
  old fit used the wrong formula.

Some defaults changed, which can affect existing scripts:
- NHF force curves `[row, col]` now give the true grid position.
- `image` and `hist` choose their unit from the channel; `scale=1e9` still forces nm.
- `image` shows a color bar unless `colorbar=False`.
- `remove_outliers` gives different numbers and no longer takes `neighbors`.

`CHANGELOG.md`, in the source distribution, has the full list.

## Installing
From PyPi
```
pip install NSFopen
```

Needs numpy, scipy, pandas, h5py and matplotlib 3.6 or newer. Examples are in the example folder; `NSFopen_help` shows where. Each
is a notebook with its `.py` and `.html`, to run from its own folder:

| Example | Shows |
|---|---|
| `Plotting_Data` | Leveling with a mask, figures for a report, subplots, 3D |
| `Calculate_Roughness` | Surface statistics, height distribution, bearing curve, PSD, roughness along a line |
| `Plane_Removal` | Leveling all the files of a folder |
| `Grain_Analysis` | Finding features and measuring each of them |
| `Lateral_Force` | Friction from the forward and backward scans |
| `Force_Spectroscopy` | Fitting a force map with the DMT model; `Contact_Models.py`: the same with `process.spectroscopy`, maps and the quality filter |
| `Spring_Constant` | Sader method from a thermal tune |
| `Sharpen_Image` | Your own processing on the arrays, stored back as channels |

## Reading files
```python
from NSFopen.read import read

afm = read('filename.nid')  # or .nhf
afm.data['Image']['Forward']['Z-Axis']  # an image, in SI units
afm.param                               # parameters of the file
afm.units[('Image', 'Forward', 'Z-Axis')]
```
Row 0 of an image is the first line scanned, which is the bottom of the picture for an upward scan.

### Spectroscopy (.nhf)
Files of any size open immediately: curves are read when you ask for them.
```python
afm = read('grid.nhf')
afm.spec.segments, afm.spec.shape, afm.spec.n_points

deflection = afm.spec['Retract 1']['Deflection']
deflection[10]        # 11th curve in acquisition order
deflection[2, 3]      # curve at row 2, column 3 of the grid, row 0 at the bottom
deflection.to('m')[2, 3]   # in another unit, if the file holds its calibration
deflection.curves()   # all curves, which may differ in length
deflection.array()    # all curves as (rows, columns, samples), padded with NaN

z, force = afm.spec.force_curve('Retract 1', (2, 3))
afm.close()
```
`afm.data['Spec']['Retract 1']['Deflection']` is the same object.

### WaveMode NMA (.nhf)
The scalar channels are in `afm.data` as for any image, the per-pixel curves in `afm.nma`.
```python
afm = read('nma.nhf')
curve = afm.nma.curve((100, 200), scan='Forward')  # row, column
curve.z, curve.interaction, curve.force
z_advance, force_advance = curve.advance
z_retract, force_retract = curve.retract

adhesion = afm.nma.map()  # most negative interaction of every pixel, in V
stiff = afm.nma.map(lambda block: block.max(axis=1), scan='Backward')  # any function of (pixels, samples)
```

## Processing images
```python
from NSFopen.process import process

afm = process('filename.nhf')
afm.level_flatten()                     # align the lines, then subtract a plane
afm.mask_otsu()                         # two-level image: mask the upper level
afm.flatten(fit_on='exclude')           # level the lines on what is not masked

afm.statistics(region='unmasked')       # Sa, Sq, Ssk, Sku, Sdq, Sdr, Sal, Str, ... with units
grains = afm.grains(split='height')     # one row per grain in grains.table
profile = afm.line_profile((0, 1e-6), (2e-6, 1e-6), width=5)   # end points in m
afm.step_heights(profile).steps
afm.critical_dimension(profile).features
afm.profile_roughness(profile).params   # ISO 21920
afm.psd_fit().summary()                 # corner, power law, noise floor, resolution
afm.lattice().summary()

afm.image(show_mask=True)
afm.hist()
afm.plot3d()                            # 3D, colored by height
afm.plot3d(material='Gold', tilt=60)    # 3D, a material under a light
afm.plot3d(color='Phase')               # 3D, colored by another channel
```
`afm.array()` gives a channel as an array and `afm.set_channel(values, channel='Name', unit='m')` stores your own array
as a channel, for all the functions below.

All processing works on one channel, by default `direction='Forward'` and the height, which files call `Z-Axis` or
`Topography`; `afm.channels` lists them all. It changes the channel unless `inplace=False`, when the new array is
returned instead.

| Group | Methods |
|---|---|
| Leveling | `flatten`, `plane_level`, `level_flatten`, `level_poly2d`, `level_lines`, `level_align_rows`, `level_facet`, `level_three_point`, `level_median`, `level_rolling_ball`, `level_shift` |
| Filters | `denoise`, `filter_basic`, `filter_median`, `filter_gaussian`, `filter_fft`, `filter_notch` |
| Glitches | `remove_outliers`, `glitch_spikes`, `glitch_lines`, `glitch_auto_lines`, `glitch_scars`, `glitch_step_lines`, `glitch_fill_mask` |
| Masks | `set_mask`, `mask_threshold`, `mask_otsu`, `mask_watershed`, `mask_outliers`, `mask_scars`, `mask_shape`, `mask_floodfill`, `mask_grow`, `mask_shrink`, `mask_invert`, `mask_clear`, `mask_remove_small`, `mask_fill_holes` |
| Geometry and scale | `geometry_crop`, `geometry_flip`, `geometry_rotate`, `geometry_resample`, `geometry_square_pixels`, `rescale_scale`, `rescale_z_to_reference`, `rescale_xy` |
| Height statistics | `stats`, `statistics`, `bearing_curve`, `histogram_fit`, `slope_distributions` |
| Lines | `line_profile`, `step_heights`, `critical_dimension`, `profile_roughness`, `masked_step_height` |
| Spectra and texture | `psd`, `psd_fit`, `spectrum`, `acf`, `line_functions`, `rms_by_scale`, `lattice` |
| Grains | `grains` |
| Two channels | `combine`, `find_shift` |
| Plots | `image`, `thumb`, `hist`, `plot3d`, `plot_profile`, `plot_psd`, `plot_histogram_fit`, `plot_bearing`, `plot_acf` |

`help(process.level_rolling_ball)` explains a method and its parameters; `process.operations()` lists the operations
with their defaults.

## Figures
`image`, `hist`, `thumb` and `plot3d` take a channel and return a matplotlib figure, with the unit (nm, mV, ...) and
labels from the file.
```python
fig = afm.image(channel='Phase', color_range='p1', cmap='viridis')   # color range: 'full', 'p1' (1-99 %) or (low, high)
fig.savefig('phase.png')

afm.image(scalebar=True, title=None, figsize=(4, 4), dpi=600, theme='dark')

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
afm.image(ax=axes[0])
afm.hist(ax=axes[1])
afm.plot3d(material='Gold', ax=axes[2])
```

Results of the analysis have their plots too, with the same options:
```python
profile = afm.line_profile((0, 1e-6), (2e-6, 1e-6), width=5)
afm.image(profile=profile)              # where the line is
afm.plot_profile(profile, steps=afm.step_heights(profile))   # or cd=afm.critical_dimension(profile)
afm.plot_psd()                          # with a fit; fit=None for the data alone
afm.plot_histogram_fit(model='gauss2')
afm.plot_bearing()
afm.plot_acf()
```

## Color maps
```python
from NSFopen.process import colors
afm.image(cmap='Gold')           # by name
afm.image(cmap=colors('Gold').cmap)
```
The Gwyddion gradients are included with the package.
