# -*- coding: utf-8 -*-
""" Color maps for AFM images.

The Nanosurf color maps are defined here as color tables (position from 0 to 1, color), so they need no data files.
`register()` adds them to matplotlib, after which every plotting function takes their names, e.g.
`plt.imshow(height, cmap="Sky")`. `colormap(name)` returns a color map by name: a Nanosurf map, "BlueBlackRed", an
older name (see ALIASES) or any matplotlib map.

Gradient files of Gwyddion are not part of NSFopen, but an installed Gwyddion's are found and used (Windows, macOS,
Linux; see gwyddion_dirs): `colors("Rainbow1").cmap`, `register(gwyddion=True)` adds them to matplotlib, and
`gwyddion_gradients()` lists them. A folder of gradient files can also be given: `colors(name, gwy_dir=folder)`.
"""

import os
import pathlib
import shutil
import sys

from matplotlib.colors import LinearSegmentedColormap

HEIGHT = "Sky"                       # the default color map of height (topography) images

# name: ((position, color), ...), positions rising from 0 to 1
NANOSURF = {
    "Sky": ((0, "#000000"), (0.1, "#131432"), (0.2, "#262965"), (0.3, "#38466e"), (0.4, "#4b6477"), (0.5, "#8c6f5a"),
            (0.6, "#ca7a3f"), (0.7, "#e3a74a"), (0.8, "#fcd355"), (0.9, "#fee9ac"), (1, "#ffffff")),
    "Warm": ((0, "#000000"), (0.05, "#17090d"), (0.1, "#30131b"), (0.15, "#4a1d29"), (0.2, "#632636"),
             (0.25, "#7c3044"), (0.3, "#8c303f"), (0.35, "#9e2f39"), (0.4, "#b02f34"), (0.45, "#c22f2e"),
             (0.5, "#cc4a28"), (0.55, "#d46323"), (0.6, "#de7e1d"), (0.65, "#e0951c"), (0.7, "#e1ab1c"),
             (0.75, "#e3c01f"), (0.8, "#e5cd5f"), (0.85, "#e6dba5"), (0.9, "#e8e8e8"), (0.95, "#f4f4f4"),
             (1, "#ffffff")),
    "Flare": ((0, "#000000"), (0.025, "#0a0014"), (0.05, "#130029"), (0.075, "#1e0040"), (0.1, "#280055"),
              (0.125, "#33006c"), (0.15, "#3d0081"), (0.175, "#450092"), (0.2, "#4a009d"), (0.225, "#4f00a7"),
              (0.25, "#5400b2"), (0.275, "#5800bc"), (0.3, "#5d00c6"), (0.325, "#6200d1"), (0.35, "#6800ca"),
              (0.375, "#7000b9"), (0.4, "#7700aa"), (0.425, "#7e009b"), (0.45, "#850089"), (0.475, "#8c007a"),
              (0.5, "#940069"), (0.525, "#9b005a"), (0.55, "#a1004b"), (0.575, "#a90039"), (0.6, "#b0002a"),
              (0.625, "#b80019"), (0.65, "#bf000a"), (0.675, "#c40500"), (0.7, "#c91800"), (0.725, "#ce2800"),
              (0.75, "#d33b00"), (0.775, "#d74b00"), (0.8, "#db5b00"), (0.825, "#e06e00"), (0.85, "#e47e00"),
              (0.875, "#e99100"), (0.9, "#eda100"), (0.925, "#f2b100"), (0.95, "#f7c400"), (0.975, "#fbd400"),
              (1, "#ffe400")),
    "Caribbean": ((0, "#000000"), (0.1, "#000e1c"), (0.2, "#001d39"), (0.3, "#003d58"), (0.4, "#005d78"),
                  (0.5, "#007774"), (0.6, "#009070"), (0.7, "#27ae41"), (0.8, "#4ccc13"), (0.9, "#a7e68b"),
                  (1, "#ffffff")),
    "Strata": ((0, "#000000"), (0.025, "#030637"), (0.05, "#050c6e"), (0.075, "#061678"), (0.1, "#062071"),
               (0.125, "#062b68"), (0.15, "#063561"), (0.175, "#063e58"), (0.2, "#064546"), (0.225, "#064b36"),
               (0.25, "#065224"), (0.275, "#065914"), (0.3, "#065f0f"), (0.325, "#06660d"), (0.35, "#066c0b"),
               (0.375, "#067408"), (0.4, "#0c7c07"), (0.425, "#1e890b"), (0.45, "#33970e"), (0.475, "#45a411"),
               (0.5, "#5ab215"), (0.525, "#71bb13"), (0.55, "#8ac310"), (0.575, "#a1c30c"), (0.6, "#b1be09"),
               (0.625, "#bdb606"), (0.65, "#bdad06"), (0.675, "#bda306"), (0.7, "#bd9806"), (0.725, "#bd8f06"),
               (0.75, "#bd8506"), (0.775, "#bd7d06"), (0.8, "#bd7506"), (0.825, "#bd6b06"), (0.85, "#bd6406"),
               (0.875, "#bd5e06"), (0.9, "#bd5906"), (0.925, "#bd5406"), (0.95, "#bd4f06"), (0.975, "#c88d66"),
               (1, "#d4d4d4")),
    "Gold": ((0, "#000000"), (0.05, "#0c0400"), (0.1, "#1a0800"), (0.15, "#270d00"), (0.2, "#351100"),
             (0.25, "#421500"), (0.3, "#4f1900"), (0.35, "#5d2100"), (0.4, "#6c3000"), (0.45, "#7b3f00"),
             (0.5, "#8b4f00"), (0.55, "#995d00"), (0.6, "#a86c00"), (0.65, "#b77b00"), (0.7, "#c38d0e"),
             (0.75, "#cda021"), (0.8, "#d6b233"), (0.85, "#dfc547"), (0.9, "#e9d85a"), (0.95, "#f3ea6e"),
             (1, "#fcfc80")),
    "Digitalis": ((0, "#000000"), (0.05, "#14020d"), (0.1, "#2a041a"), (0.15, "#3f0628"), (0.2, "#550835"),
                  (0.25, "#6b0b43"), (0.3, "#8a204a"), (0.35, "#ab3752"), (0.4, "#cc4e59"), (0.45, "#e5655f"),
                  (0.5, "#e97961"), (0.55, "#ed8c62"), (0.6, "#f1a164"), (0.65, "#f5b566"), (0.7, "#ebc76f"),
                  (0.75, "#d7d87e"), (0.8, "#c5e78b"), (0.85, "#b8df90"), (0.9, "#acd393"), (0.95, "#d7e9cb"),
                  (1, "#ffffff")),
    "Neon": ((0, "#a70000"), (0.05, "#a71000"), (0.1, "#a72200"), (0.15, "#a73400"), (0.2, "#a74600"),
             (0.25, "#a75800"), (0.3, "#b87000"), (0.35, "#ca8a00"), (0.4, "#dca500"), (0.45, "#eebf00"),
             (0.5, "#fdd800"), (0.55, "#d2da00"), (0.6, "#a3dc00"), (0.65, "#74de00"), (0.7, "#44e000"),
             (0.75, "#18e203"), (0.8, "#13e833"), (0.85, "#0eee67"), (0.9, "#09f49b"), (0.95, "#05facf"),
             (1, "#00ffff")),
    "Plum": ((0, "#000000"), (0.05, "#100312"), (0.1, "#220726"), (0.15, "#340a3a"), (0.2, "#460e4e"),
             (0.25, "#571162"), (0.3, "#5f2577"), (0.35, "#673b8e"), (0.4, "#6f50a5"), (0.45, "#7665bc"),
             (0.5, "#7e7ad2"), (0.55, "#7181d6"), (0.6, "#6488da"), (0.65, "#568fdf"), (0.7, "#4996e3"),
             (0.75, "#3f9ee7"), (0.8, "#63b0ec"), (0.85, "#8bc4f1"), (0.9, "#b3d8f6"), (0.95, "#daedfa"),
             (1, "#ffffff")),
    "Spectrum": ((0, "#05059a"), (0.05, "#0216af"), (0.1, "#002bc3"), (0.15, "#004bcf"), (0.2, "#006fd3"),
                 (0.25, "#0197d0"), (0.3, "#01b4bf"), (0.35, "#00cda9"), (0.4, "#00d17f"), (0.45, "#01d354"),
                 (0.5, "#25d527"), (0.55, "#49d600"), (0.6, "#7ada00"), (0.65, "#a8de00"), (0.7, "#d1e000"),
                 (0.75, "#ebd200"), (0.8, "#f2af01"), (0.85, "#f78b01"), (0.9, "#fb6700"), (0.95, "#f43c04"),
                 (1, "#e90e0a")),
    "Cold": ((0, "#000000"), (0.1, "#0e1729"), (0.2, "#1d2e53"), (0.3, "#2b457c"), (0.4, "#2e568c"), (0.5, "#32689b"),
             (0.6, "#4489af"), (0.7, "#58acc5"), (0.8, "#9fcad6"), (0.9, "#e8e8e8"), (1, "#ffffff")),
    "Maple": ((0, "#000000"), (0.025, "#020901"), (0.05, "#031201"), (0.075, "#051d02"), (0.1, "#072603"),
              (0.125, "#093104"), (0.15, "#0b3a04"), (0.175, "#0c4305"), (0.2, "#0e4e06"), (0.225, "#105706"),
              (0.25, "#136107"), (0.275, "#266d09"), (0.3, "#3a780b"), (0.325, "#51850e"), (0.35, "#659010"),
              (0.375, "#7c9d12"), (0.4, "#8fa814"), (0.425, "#a3b316"), (0.45, "#bac018"), (0.475, "#cecb1a"),
              (0.5, "#e3d71d"), (0.525, "#e8d122"), (0.55, "#edca26"), (0.575, "#f2c32c"), (0.6, "#f7bd30"),
              (0.625, "#fcb536"), (0.65, "#fcac3a"), (0.675, "#fda33d"), (0.7, "#fe9842"), (0.725, "#fe8f46"),
              (0.75, "#ff864c"), (0.775, "#ff8c5d"), (0.8, "#ff926e"), (0.825, "#ff9982"), (0.85, "#ff9e93"),
              (0.875, "#ffa7a7"), (0.9, "#ffb8b8"), (0.925, "#ffc9c9"), (0.95, "#ffdddd"), (0.975, "#ffeeee"),
              (1, "#ffffff")),
}

BLUE_BLACK_RED = ((0.0, "#9ecae1"), (0.5, "#000000"), (1.0, "#fc9272"))

# names of earlier NSFopen versions -> today's
ALIASES = {"Pm3d": "Flare", "Blend2": "Strata", "Code-V": "Spectrum"}


def _build(name, stops):
    return LinearSegmentedColormap.from_list(name, list(stops), N=256)


def names():
    """ The names of the color maps of this module (Nanosurf maps and BlueBlackRed). """
    return list(NANOSURF) + ["BlueBlackRed"]


def colormap(name):
    """ The color map of a name: a Nanosurf map, BlueBlackRed, an older name (ALIASES) or a matplotlib map. A color
    map object is returned as it is. Raises ValueError for an unknown name. """
    import matplotlib
    if not isinstance(name, str):
        return name
    name = ALIASES.get(name, name)
    if name in NANOSURF:
        return _build(name, NANOSURF[name])
    if name == "BlueBlackRed":
        return _build(name, BLUE_BLACK_RED)
    if name in matplotlib.colormaps:
        return matplotlib.colormaps[name]
    raise ValueError(f"Unknown color map '{name}': use one of {', '.join(names())} or a matplotlib name.")


def register(gwyddion=False):
    """ Adds the Nanosurf maps and BlueBlackRed (and their reversed versions, name + '_r') to matplotlib; with
    gwyddion=True also the gradients of an installed Gwyddion (a name that is taken gets ' (Gwyddion)'). Returns the
    names. Calling it again does nothing new. """
    import matplotlib
    out = names()
    for name in out:
        if name not in matplotlib.colormaps:
            cmap = colormap(name)
            matplotlib.colormaps.register(cmap, name=name)
            matplotlib.colormaps.register(cmap.reversed(), name=name + "_r")
    if gwyddion:
        for name, path in gwyddion_gradients().items():
            target = name if name not in matplotlib.colormaps else f"{name} (Gwyddion)"
            if target not in matplotlib.colormaps:
                try:
                    cmap = LinearSegmentedColormap.from_list(target, read_gradient(path), N=256)
                except (ValueError, OSError, UnicodeError):
                    continue
                matplotlib.colormaps.register(cmap, name=target)
            out.append(target)
    return out


def gwyddion_dirs(platform=None, environ=None, home=None):
    """ The gradient folders of an installed Gwyddion that exist, the user's own first (they win over the system ones).

    Searched, after Gwyddion's documentation (resources: user folder ~/.gwyddion, on Windows %USERPROFILE%\\gwyddion;
    system folder share/gwyddion, or $GWYDDION_DATADIR/gwyddion):
        all          $GWYDDION_DATADIR/gwyddion/gradients; next to a gwyddion program on the PATH (../share/...)
        Windows      %USERPROFILE%\\gwyddion, Documents\\gwyddion; Program Files (64 and 32 bit), %LOCALAPPDATA%\\Programs:
                     Gwyddion\\share\\gwyddion\\gradients
        macOS        ~/.gwyddion; Gwyddion.app in /Applications or ~/Applications; Homebrew, MacPorts, /usr/local
        Linux, other ~/.gwyddion; /usr/share, /usr/local/share; Flatpak (system and user)
    """
    platform = platform or sys.platform
    environ = os.environ if environ is None else environ
    home = pathlib.Path(home) if home else pathlib.Path.home()
    tail = pathlib.Path("gwyddion", "gradients")
    user, system = [], []
    if environ.get("GWYDDION_DATADIR"):
        system.append(pathlib.Path(environ["GWYDDION_DATADIR"]) / tail)
    program = shutil.which("gwyddion", path=environ.get("PATH")) if environ.get("PATH") else None
    if program:
        system.append(pathlib.Path(program).resolve().parent.parent / "share" / tail)
    if platform.startswith("win"):
        profile = pathlib.Path(environ.get("USERPROFILE", home))
        user += [profile / "gwyddion" / "gradients", profile / "Documents" / "gwyddion" / "gradients"]
        roots = [environ.get(k) for k in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)")]
        roots += [str(pathlib.Path(environ["LOCALAPPDATA"]) / "Programs")] if environ.get("LOCALAPPDATA") else []
        roots += ["C:\\Program Files", "C:\\Program Files (x86)"]
        system += [pathlib.Path(r) / "Gwyddion" / "share" / tail for r in roots if r]
    elif platform == "darwin":
        user.append(home / ".gwyddion" / "gradients")
        apps = [pathlib.Path("/Applications"), home / "Applications"]
        system += [a / "Gwyddion.app" / "Contents" / "Resources" / "share" / tail for a in apps]
        system += [pathlib.Path(p) / tail for p in ("/opt/homebrew/share", "/opt/local/share", "/usr/local/share")]
    else:
        user.append(home / ".gwyddion" / "gradients")
        flatpak = pathlib.Path("app", "net.gwyddion.Gwyddion", "current", "active", "files", "share") / tail
        system += [pathlib.Path(p) / tail for p in ("/usr/share", "/usr/local/share")]
        system += [pathlib.Path("/var/lib/flatpak") / flatpak, home / ".local" / "share" / "flatpak" / flatpak]
    found = []
    for folder in user + system:
        try:
            if folder.is_dir() and folder.resolve() not in [f.resolve() for f in found]:
                found.append(folder)
        except OSError:
            pass
    return found


def gwyddion_gradients(**kwargs):
    """ {name: path} of the gradient files of an installed Gwyddion (gwyddion_dirs; the user's own win). """
    out = {}
    for folder in gwyddion_dirs(**kwargs):
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.name not in out:
                out[path.name] = path
    return out


def read_gradient(path):
    """ [(position, (r, g, b)), ...] of a Gwyddion gradient file. """
    lines = pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or not lines[0].startswith("Gwyddion resource GwyGradient"):
        raise ValueError(f"{path} is not a Gwyddion gradient file.")
    rows = [[float(v) for v in line.split()] for line in lines[1:] if line.strip()]
    return [(row[0], tuple(row[1:4])) for row in rows]


class colors:
    """ A color map by name, in .cmap.

    Parameters
    ----------
        name : string
            A Nanosurf map (NANOSURF), "BlueBlackRed", an older name (ALIASES), a matplotlib map, the name of a
            gradient file in gwy_dir, or of a gradient of an installed Gwyddion (gwyddion_dirs)
        gwy_dir : string, optional
            Folder with Gwyddion gradient files (not part of NSFopen); without it an installed Gwyddion is searched
    """

    def __init__(self, name=None, gwy_dir=None, **kwargs):
        self.name = name
        self.gwy_dir = pathlib.Path(gwy_dir) if gwy_dir else None
        self.cmap_list = {"Nanosurf": list(NANOSURF), "Custom": ["BlueBlackRed"]}
        if self.gwy_dir is not None and self.gwy_dir.is_dir():
            files = {p.name: p for p in self.gwy_dir.iterdir() if p.is_file()}
        else:
            files = gwyddion_gradients()
        if files:
            self.cmap_list["Gwyddion"] = sorted(files, key=str.lower)
        if name is not None and name in files and (self.gwy_dir is not None or name not in NANOSURF):
            self.cmap = LinearSegmentedColormap.from_list(name, read_gradient(files[name]))
            return
        try:
            self.cmap = colormap(name)
        except ValueError:
            print("Name Not Found\n")
            print("Use one of the following choices (or any matplotlib color map):\n")
            print(sorted({x for v in self.cmap_list.values() for x in v}, key=str.lower))
            self.cmap = []
