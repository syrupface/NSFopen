Installation
------------

	python setup.py install

Requirements
------------

	Python 3
	numpy
	itertools
	h5py
	pandas

Example
------------

example/sample_script.py
	

Example: NID File

from NSFopen.read import nid_read
afm = nid_read('filename.nid')
data = afm.data # raw data
param = afm.param # parameters

Example: NHF File

from NSFopen.read import nhf_read
afm = nhf_read('filename.nid')
data = afm.data # raw data
param = afm.param # parameters

	