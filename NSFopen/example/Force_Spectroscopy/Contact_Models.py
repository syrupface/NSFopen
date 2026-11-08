#!/usr/bin/env python
# coding: utf-8
""" Fitting a force map with the contact models of NSFopen.process.spectroscopy and contact (NSFopen 2.4).
The same calculations as the Spectroscopy page of the Nanosurf Surface Analyzer, so a script and the app give the same
numbers. Run it from this folder, next to spectroscopy.nid. The tip radius and Poisson ratio are assumptions: the
modulus scales with them, so set them for your own sample. """

import matplotlib.pyplot as plt
import numpy as np

from NSFopen.process import contact, spectroscopy

spec = spectroscopy.SpecFile("spectroscopy.nid")
print(f"{spec.n_curves} curves, grid {spec.grid}, k = {spec.spring_constant:.4g} N/m")
print("Models:", ", ".join(label for key, label in contact.MODELS))

# DMT on the retract curves, spherical tip, offset and tilt removed from the first 40 % of the approach
settings = spectroscopy.ForceSettings(model="dmt", shape="sphere", radius=10e-9, poisson=0.35,
                                      baseline="linear", baseline_from=0, baseline_to=40)
results = spectroscopy.ForceAnalysis(spec, settings).run()
results.min_r2 = 0.95                       # leave out fits that do not describe their curve

for name, unit, n, mean, std, median, low, high in results.statistics():
    print(f"{name:28s} N={n:4d}  mean {spectroscopy.format_value(mean, unit):>12s}  "
          f"std {spectroscopy.format_value(std, unit):>12s}")
results.write_csv("spectroscopy_results.csv")

# Maps of the modulus and the adhesion (NaN: left out or not fitted)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, name in zip(axes, ("E", "adhesion (minimum retract)")):
    values = results.map(name)
    image = ax.imshow(values, origin="lower", cmap="viridis")
    ax.set_title(f"{name} [{results.units[name]}]")
    fig.colorbar(image, ax=ax)
plt.tight_layout()
plt.show()

# The stiffest curve with its fit: force against the indentation
curve = spectroscopy.ForceAnalysis(spec, settings).curve(int(np.nanargmax(results.column("E"))))
segment = curve.retract
plt.figure()
plt.plot((segment.delta - curve.contact_delta) * 1e9, segment.force * 1e9, label="retract")
plt.plot((curve.fit.fit_delta - curve.contact_delta) * 1e9, curve.fit.fit_force * 1e9, "--", label="DMT fit")
plt.xlabel("indentation [nm]")
plt.ylabel("force [nN]")
plt.title(f"curve {curve.index + 1}: E = {spectroscopy.format_value(curve.fit.modulus, 'Pa')}")
plt.legend()
plt.show()
