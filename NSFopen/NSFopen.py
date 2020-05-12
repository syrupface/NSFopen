# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:17:03 2018

Copyright (c) 2018 nelson<at>nanosurf.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


import re
import numpy as np
from itertools import chain
import time


class read():
    def __init__(self, filename=None, dataframe=True, verbose=True):
        self.filename = filename
        self.__dataframe = dataframe
        self.verbose = verbose
        # names of the channels that can be opened.  Can be added to.
        self.data_names = ['Spec forward', 'Spec backward', 'Spec fwd pause', 'Spec bwd pause',
                           'Scan forward', 'Scan backward', '2nd scan forward', '2nd scan backward',
                           'Frequency sweep', 'Frequency sweep SHO', 'Spectrum FFT', 'Spectrum Fit']
        self.data_types = [1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2]
        self.data = None
        self.param = None
        if not filename:
            print('\nFile required!\n')
            return
        self.read()

    def read(self):
        t = time.time()

        encode = 'ANSI'
        # encode = 'UTF-8'

        # isolates header from data file
        if self.verbose:
            print("Reading Header")
        
        fid = open(self.filename, 'rb')
        data_in = fid.read().split(b'\n\r\n\r')
        fid.close()

        head = np.array(data_in[0].split(b'\r\n'))
        boo = head == b''
        indices = np.nonzero(boo[1:] != boo[:-1])[0] + 1
        head = np.split(head, indices)

        head = [h for h in head if len(h) > 1]

        header = {}
        for h in head:
            title = h[0].decode(encode)[1:-1]
            data = h[1:]
            values = {}
            for d in data:
                val = d.decode(encode).split('=')
                values[val[0]] = val[1].rstrip()
            header[title] = values

        param = {}

        def from_array(key, param):
            return float(header[key][param].split(',')[4])
        
        def get_values(key, param):
            string = header[key][param]
            
            ty, val, unit = re.search(r'(\S)\[(.*)\]\*\[(.*)\]', string).groups()
            if ty == 'D':
                val = float(val)
            if ty == 'B':
                val = bool(val)
            if ty == 'L':
                val = int(val)
            if ty == 'V':
                val = [float(v) for v in val.split(',')]
                unit = [str(u) for u in unit.split(',')]
            return {'Value': val, 'Unit': unit}
        
        groupNames = []
        dataSet = []
        
        ds = 'DataSet'
        grCnt = int(header[ds]['GroupCount'])
        for gr in range(grCnt):
            groupNames.append(header[ds]['Gr' + str(gr) + '-Name'])
            gc = int(header[ds]['Gr' + str(gr) + '-Count'])
            
            for g in range(gc):
                key = 'Gr' + str(gr) + '-Ch' + str(g)
                if key in header[ds]:
                    dataSet.append(header[ds][key])

        terms = ['Frame', 'Points', 'Lines', 'SaveBits',
                 'Dim0Min', 'Dim0Range', 'Dim0Unit',
                 'Dim1Min', 'Dim1Range', 'Dim1Unit',
                 'Dim2Min', 'Dim2Range', 'Dim2Unit',
                 'Dim2Name']
        
        types = [0, 1, 1, 1,
                 2, 2, 0,
                 2, 2, 0,
                 2, 2, 0,
                 0]

        # 0 = string, 1 = integer, 2 = float
        typ = [lambda x:str(x), lambda x:int(x), lambda x:float(x)]
     
        for term, ty in zip(terms, types):
            val = []
            for d in dataSet:
                h = header[d]
                if term in h:
                    val.append(typ[ty](h[term]))

            param[term] = val

        # for spectroscopy with maps
        dataPoints = []
        for d in dataSet:
            h = header[d]
            linePoints = []
            if 'LineDim0Min' in h:
                for key, value in h.items():
                    if re.match(r'LineDim\d*Points', key):
                        linePoints.append(int(value))
            dataPoints.append(np.array(linePoints))
        param['LinePoints'] = dataPoints
        
        cantilever = {}
        key = 'DataSet\\Calibration\\Cantilever'
        if key in header:
            cantilever = {'Manufacturer': header[key]['Manufacturer'], 'Name': header[key]['Name']}

            propcount = int(header[key]['PropCount'])
            for i in range(propcount):
                cantilever['Prop' + str(i)] = get_values(key, 'Prop' + str(i))

        key = 'DataSet\\Calibration\\Scanhead'
        if key in header:
            sens = from_array(key, 'In5')
            cantilever['Sensitivity'] = {'Value': sens / 10, 'Unit': 'm'}

        mapTable = {}  
        key = 'DataSet\\SpecInfos'
        if key in header:
            n = int(header[key]['SubSectionCount'])
            secNames = [header[key]['SubSection' + str(i)] for i in range(n)]
            
            specMode = header[key + '\\' + secNames[0]]['SpecMode']
            
            count = int(header[key + '\\' + secNames[1]]['Count'])
            
            mapTable = [header[key + '\\' + secNames[1]][specMode[:3] + str(i)].split(';') for i in range(count)]
            mapTable = np.array(mapTable).astype(np.float)
            
        scanOffset = {}
        key = 'DataSet\\Parameters\\Imaging'
        if key in header:
            scanOffset = get_values(key, 'ScanOffset')
        
        thermal = {}
        key = 'DataSet-Info'
        if '-- Thermal Tuning --' in header[key]:
            Q = float(header[key]['Q Factor:'])
            Freq = header[key]['Frequency:']
            SpringK = header[key]['Spring Constant:']
            Peak = header[key]['Peak Value:']
            
            gr = re.search(r'(\d+\.\d+)(\S*)', Freq).groups()
            Freq = {'Value': gr[0], 'Unit': gr[1]}
            
            gr = re.search(r'(\d+\.\d+)\s(\S*)', SpringK).groups()
            SpringK = {'Value': gr[0], 'Unit': gr[1]}
            
            gr = re.search(r'(\d+\.\d+)(\S*)', Peak).groups()
            Peak = {'Value': gr[0], 'Unit': gr[1]}
            
            thermal = {'Frequency': Freq, 'Q Factor': Q, 'Spring Constant': SpringK, 'Peak': Peak}
        
        required_size = int(sum([a*b*c/8 for a, b, c in zip(param['Points'],
                                                            param['Lines'],
                                                            param['SaveBits'])]))
        
        q = float(2**param['SaveBits'][0])
        z0 = float(q/2)

        if param['SaveBits'][0] == 16:
            dt = np.int16
        elif param['SaveBits'][0] == 32:
            dt = np.int32
        
        if self.verbose:
            print('Reading Data')
        
        fid = open(self.filename, 'rb')
        fid.seek(-required_size, 2)
        data_in = np.fromfile(fid, count=required_size, dtype=dt).astype(float)
        data_in = (data_in + z0) / q
        fid.close()

        data_in = np.split(data_in, np.cumsum([a * b for a, b in zip(param['Points'],
                                                                     param['Lines'])]))
        data_in.pop()

        data = []
        # reshape and rescale data
        for datain, pts, lns, zran, zmin in zip(data_in,
                                                param['Points'],
                                                param['Lines'],
                                                param['Dim2Range'],
                                                param['Dim2Min']):

            data.append(datain.reshape((lns, pts)).__mul__(zran).__add__(zmin))

        data_crop = []
        for numP, dSet in zip(param['LinePoints'], data):
            if numP.any():
                temp = []
                for n, d in zip(numP, dSet):
                    temp.append(d[:n])
                data_crop.append(temp)
            else:
                data_crop.append(dSet)
        data = np.array(data_crop)

        # this is the index of the channels that will be output.
        idx = [idx for idx, frame in enumerate(param['Frame']) if frame in self.data_names]

        # only take data from approved list in name
        data = data[idx]
        frames = np.array(param['Frame'])[idx]
        channel = np.array(param['Dim2Name'])[idx]

        out = {}
        for frame, chan, dat in zip(frames, channel, data):
            if frame not in out:
                out[frame] = {}
            if chan not in out[frame]:
                out[frame][chan] = {}
            out[frame][chan] = dat
    
        image = {}
        spec = {}
        spectrum = {}

        for name, value in out.items():
            dt = self.data_types[self.data_names.index(name)]
            if dt == 0:
                if name == 'Scan forward':
                    image['Forward'] = value
                elif name == 'Scan backward':
                    image['Backward'] = value
                elif name == '2nd scan forward':
                    image['2nd Forward'] = value
                elif name == '2nd scan backward':
                    image['2nd Backward'] = value
                else:
                    pass
            if dt == 1:
                if name == 'Spec forward':
                    spec['Forward'] = value
                elif name == 'Spec backward':
                    spec['Backward'] = value
                elif name == 'Spec fwd pause':
                    spec['Pause Forward'] = value
                elif name == 'Spec bwd pause':
                    spec['Pause Backward'] = value
                else:
                    pass
            if dt == 2:
                if name == 'Spectrum FFT':
                    spectrum['FFT'] = value
                elif name == 'Spectrum Fit':
                    spectrum['Fit'] = value
                else:
                    pass

        x = dict(zip(['min', 'range', 'units'],
                     [param['Dim0Min'], param['Dim0Range'], param['Dim0Unit']]))
        y = dict(zip(['min', 'range', 'units'],
                     [param['Dim1Min'], param['Dim1Range'], param['Dim1Unit']]))
        z = dict(zip(['min', 'range', 'units'],
                     [param['Dim2Min'], param['Dim2Range'], param['Dim2Unit']]))
        specMap = dict(zip(['maps'], [mapTable]))
        offset = dict(zip(['offset'], [scanOffset]))

        parameters = dict(zip(['Tip', 'X', 'Y', 'Z', 'SpecMap', 'OffSet', 'Tune', 'Header Dump'],
                              [cantilever, x, y, z, specMap, offset, thermal, header]))

        dataout = [image, spec, spectrum]
        datanames = ['Image', 'Spec', 'Sweep']

        valid_data = [val for num, val in enumerate(dataout) if val]
        valid_names = [datanames[num] for num, val in enumerate(dataout) if val]

        dataout = dict(zip(valid_names, valid_data))
        t_end = time.time() - t

        if self.verbose:
            print('Elapsed Time: %3.2f sec\n' % t_end)

        if self.__dataframe:
            # import pandas as pd
            self.data = self.__toPandaDF(dataout).unstack(level=0)
            # self.specunits = pd.DataFrame(specUnits)
            self.param = self.__toPandaSeries(parameters)
        else:
            self.data = dataout
            self.param = parameters

    @staticmethod
    def __toPandaDF(user_dict):
        import pandas as pd
        df = pd.DataFrame.from_dict({(i, j): user_dict[i][j]
                                     for i in user_dict.keys()
                                     for j in user_dict[i].keys()}, orient='index').transpose()
        return df

    @staticmethod
    def __toPandaSeries(user_dict):
        import pandas as pd
        df = pd.Series({(i, j): user_dict[i][j]
                        for i in user_dict.keys()
                        for j in user_dict[i].keys()})
        return df

# if __name__=="__main__":
#     filename='..\\example\\spec.nid'
#     afm=read(filename)
#     afm.data
