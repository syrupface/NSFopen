# NSFopen
Open Nanosurf NID files in Python

## Installing
From source
```
python setup.py install
```
or from PyPi
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
