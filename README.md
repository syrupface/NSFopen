# NSFopen
Open Nanosurf NID files in Python

## Getting Started
Can be installed from PyPi by typing
```
pip install nanosurf
```

## Use
```
from nanosurf import read
afm = read('filename.nid')
data=afm.data # raw data
param=afm.param # parameters
```
