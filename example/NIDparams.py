# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 09:30:11 2020

@author: Edward
"""

import argparse
from NSFopen import read

parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()
afm=read(args.filename)


print('Spring Constant: %3.2f N/m' % afm.param.Tip.SpringConstant)
print('Deflection Sensitivity: %3.1f nm/V' % (afm.param.Tip.Sensitivity*1e8))
print('Q-factor: %3.1f' % afm.param.Tip.Q)
print('Length: %i um' % (afm.param.Tip.Length*1e6))
print('Width: %i um' % (afm.param.Tip.Width*1e6))
