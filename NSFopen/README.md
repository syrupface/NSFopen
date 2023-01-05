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

pandas (recommended, required for NHF for now)

h5py (required for NHF files)


## Example Script
Available in example folder

### Example: NID File
```
from NSFopen.read import nid_read
afm = nid_read('filename.nid')
data = afm.data # raw data
param = afm.param # parameters
```
### Example: NHF File
```
from NSFopen.read import nhf_read
afm = nhf_read('filename.nid')
data = afm.data # raw data
param = afm.param # parameters
```
