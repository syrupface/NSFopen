# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 05:53:03 2020

@author: nelson
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from NSFopen import read

file="M01232-Lattice-MFM-m8000000028.nid"
# file="image.nid"

data=read(file,verbose=False).data
