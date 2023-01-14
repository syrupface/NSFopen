# NSFopen
Open Nanosurf NID and NHF files in Python

## Installing
From source
```
python setup.py --install
```
or from PyPi
```
pip install nanosurf
```

## Prequisites
numpy

pandas (recommended, required for NHF for now)

h5py (required for NHF files)


## Example Script
Available in example folder

### Example: NID File
```
from NSFopen.read import read
afm = read('filename.nid')
data = afm.data # raw data
param = afm.param # parameters
```
