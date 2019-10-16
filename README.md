# NSFopen
Open Nanosurf NID files in Python

## Installing
Can be installed from PyPi by typing
```
pip install nanosurf
```

## Prequisites
numpy

itertools

pandas (optional)


## Example Script
Available in example folder

### Minimum example
```
from nanosurf import read
afm = read('filename.nid')
data = afm.data # raw data
param = afm.param # parameters
```
