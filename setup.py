# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:34:04 2019

@author: nelson
"""

from setuptools import find_packages, setup
# from distutils.core import setup

setup(
    name='NSFopen',
    version='2.0.1',
    description='Read data and parameters from Nanosurf NID and NHF files',
    author='Nanosurf',
    author_email='scripting@nanosurf.com',
    url="https://www.nanosurf.com",
	packages = find_packages(),
    readme = "README.md",
    include_package_data = True,
    package_data = {
        'example': ['*.nid',
                    'Calculate_Roughness/*.*',
                    'Lateral_Force/*.*',
                    'Plotting_Data/*.*',
                    '2D_Plane_Removal/*.*',
                    'Sharpen_Image/*.*',
                    'ForceSpectroscopy/*.*']},

    )
