"""Package for scripting the Nanosurf control software.
Copyright (C) Nanosurf AG - All Rights Reserved (2021)
License - MIT"""

import os
import pathlib
from ._version import __version__

from .read import read

# from . process import process


def package_path() -> pathlib.Path:
    return pathlib.Path(os.path.dirname(os.path.abspath(__file__)))


def example_path() -> pathlib.Path:
    return package_path() / "example"


def help():
    print("Nanosurf NID and NHF file reader:")
    print("-------------------------------")
    print(f"Installed version: {__version__}")
    print("\nExamples are stored here:")
    print(example_path())
    for example in os.listdir(example_path()):
        print(f"  {example}")
