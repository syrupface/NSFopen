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

class read:
    
    def __init__(self, filename=None,dataframe=True):
        self.filename=filename
        self.__dataframe=dataframe
        self.data=None
        self.param=None
        if not filename: 
            print('\nFile required!\n')
            return
        self.read()

    def read(self):
        t=time.time()
        
        names=['Spec forward','Spec backward','Spec fwd pause','Spec bwd pause',
               'Scan forward','Scan backward',
               'Frequency sweep','Frequency sweep SHO','Spectrum FFT','Spectrum Fit'];
        
        encode='ANSI'
#        encode='UTF-8'
               
        # isolates header from data file
        fid=open(self.filename,'rb')
        data_in=fid.read().split(b'\n') 
        fid.close()
        
        print("Reading Header")
        end_of_header=0;
        while True:
            if (str(data_in[end_of_header],encode)=='\r') and (str(data_in[end_of_header+1],encode)=='\r'):
                break
            end_of_header+=1
            
        header=[str(line,encode).strip() for line in data_in[:end_of_header]];
        # END header
        
        search_terms=['^Prop0','^In5=5,TipSignalDC','^Frame','^Points','^Lines','^SaveBits',
                        '^Dim0Min','^Dim0Range','^Dim0Unit',
                        '^Dim1Min','^Dim1Range','^Dim1Unit',
                        '^Dim2Min','^Dim2Range','^Dim2Unit',
                        '^Dim2Name',
                        '^LineDim\d*Min','^LineDim\d*Range','^LineDim\d*Points',
                        '^Name=','^Manufacturer',
                        '^Prop1','^Prop2','^Prop3',
                        '^Prop4','^Prop8','^Prop9',
                        '^Map\d*=','^ScanOffset=',
                        '^Frequency:=','^Peak\sValue:=','Q\sFactor:=','Spring\sConstant:='];
        
        # Regex matches per data type
        regF=r'[-+]?\d+\.\d+|\d+' # floats or integers
        regE=r'[-]?\d+[\.\d+]*[eE][-+]?\d+|\d+\.\d+|\d+' # scientific notation or float
        regU=r'[mVN]' # units
        regS=r'[a-zA-Z\- ]+' # string i.e. channel name
        regM=r'\d+[a-zA-Z\/\(\) ]+' # mixed data
        
        
        match=[regF,'[-]?\d+[\.\d+]*[eE][-+]?\d+',regS,'\d+','\d+','\d+',
               regE,regE,regU,
               regE,regE,regU,
               regE,regE,regU,
               regS,
               regE,regE,regF,
               '\w+','\w+',
               regF,regF,regE,
               regF,regF,regE,
               'Map%*i=%e;%e;%e;%e;%*i;%*i;%*i;%*i',regE,
               regM,regM,regM,regM]
        
        # 0=string, 1=float, 2=integer
        whatType=(1,1,0,2,
                  2,2,
                  1,1,0,
                  1,1,0,
                  1,1,0,
                  0,
                  1,1,2,
                  0,0,
                  1,1,1,
                  1,1,1,
                  1,1,
                  0,0,0,0)
        
        keys=['k','sens','frame','points','lines','bits',
              'xmin','xran','xun',
              'ymin','yran','yun',
              'zmin','zran','zun',
              'zname',
              'lnmin','lnran','lnpoints',
              'tipname','tipman',
              'tipfreq','tiplen','tipwide',
              'tipQ','tipA','tipR',
              'map','offset',
              'freq','peak','Q','k']
        
        idx=[]
        for term in search_terms:
            idx.append([i for i, line in enumerate(header) if re.search(term,line)])
            
        idx[2]=idx[2][:len(idx[5])]
        
        value=[]
        for i,j in zip(idx,match):
            value.append(list(chain.from_iterable([re.findall(re.compile(j),header[x].split('=')[1]) for x in i])))
        
        channelN=len(value[5])
        
        value[3]=value[3][-channelN:]
        value[4]=value[4][-channelN:]
        
        for i,j in enumerate(whatType):
            if j==1:
                value[i]=list(map(float,value[i]))
            elif j==2:
                value[i]=list(map(int,value[i]))
                
        # make dictionary of parameter values
        param=dict(zip(keys,value))
        
        required_size=int(sum([a*b*c/8 for a,b,c in zip(param['points'],param['lines'],param['bits'])]))
        
        # determines the channel names in the file
        nameID=np.asarray([names.index(i) if i in names else -1 for i in param['frame']])
        
        q=float(2**param['bits'][0])
        z0=float(q/2)
        #precision='int' + str(param['bl'][0]);
        
        if param['bits'][0]==16:
            dt=np.int16
        elif param['bits'][0]==32:
            dt=np.int32
        
        print('Reading Data')
        fid=open(self.filename,'rb')
        fid.seek(-required_size,2)
        data_in=np.fromfile(fid,count=required_size,dtype=dt).astype(float)
        data_in=(data_in+z0)/q
        fid.close()
        
        data_in=np.split(data_in, np.cumsum([a*b for a,b in zip(param['points'],param['lines'])]))
        data_in.pop()
        
        data=[]
        #reshape and rescale data
        for datain,pts,lns,zran,zmin in zip(data_in,param['points'],param['lines'],param['zran'],param['zmin']):
            data.append(datain.reshape((lns,pts)).__mul__(zran).__add__(zmin).T)
        
        # determine if file contains spectroscopy data
        b0=(nameID<2) # spectroscopy has a name id of 0 or 1
        b1=(nameID!=-1) # and name is in list
        
        if any(b0) and any(b1):
            numFWD=sum([x and y for x,y in zip(b0,b1)])
            if param['lnpoints']:
                lnpoints=param['lnpoints']
            else:
                lnpoints=param['points'][:4]
            dimPoints=np.split(np.array(lnpoints),numFWD)
        
        #spectroscopy files
        chanNames=['Forward','Backward','Pause Fwd','Pause Bwd']
        specChan=[i for i in nameID if i<4]
        specNum=[specChan.count(j) for j in range(4)]
        
        # this removes spaces from channel names
        znames=[name.replace(" ","").replace("-","") for name in param['zname']]
        
        specData=[]
        specUnits=[]
        if any(nameID<5) and any(nameID!=-1):
            spec=[]
            for idx,num in enumerate(specNum):
                if num>0:
                    d0=np.extract(nameID==idx,data)
                    
                    temp=[]
                    for dset in d0:
                        for idx2 in range(dset.shape[1]):
                            if idx<1:
                                dP=dimPoints[idx][idx2]
                            else:
                                dP=np.shape(dset)[0]
                                
                            temp.append(dset[:,idx2][:dP])
                    
                    temp=np.split(np.asarray(temp),num)
                    
                    specData.append(dict(zip(znames,temp)))
                    specUnits.append(dict(zip(znames,param['zun'])))
        
        specUnits=dict(zip(chanNames,specUnits))
        specData=dict(zip(chanNames,specData))
            
        
     
        #image files   
        imgData={}
        if any(nameID==4) or any(nameID==5):
            trace=[]
            retrace=[]
            imgChan=[idx for idx,val in enumerate(nameID) if val==4 or val==5]
            for i in imgChan:
                if nameID[i]==4:
                    trace.append(np.asarray(data[i].T))
                elif nameID[i]==5:
                    retrace.append(np.asarray(data[i].T))
        
            if trace:
                trace=dict(zip([i for i,j in zip(znames,nameID) if j==4],trace))        
            if retrace:
                retrace=dict(zip([i for i,j in zip(znames,nameID) if j==5],retrace))    
            
            imgData=dict(zip(['Forward','Backward'],[trace,retrace]))
        
        #frequency/thermal sweeps
        sweepData={}
        if any(nameID>5):
            sweep=[]
            fit=[]
            sweepChan=[idx for idx,val in enumerate(nameID) if val>5]
            for i in sweepChan:
                if nameID[i]==6 or nameID[i]==8:
                    sweep.append(np.asarray(data[i].flatten()))
                elif nameID[i]==7 or nameID[i]==9:
                    fit.append(np.asarray(data[i].flatten()))
                    
            if sweep:
                sweep=dict(zip([i for i,j in zip(znames,nameID) if j==6 or j==8],sweep))
            if fit:
                fit=dict(zip([i for i,j in zip(znames,nameID) if j==7 or j==9],fit))
                
            sweepData=dict(zip(['Data','Fit'],[sweep,fit]))
            
            
        
        tip_params=[param['k'],param['sens'],param['tipQ'],param['tiplen'],param['tipwide'],param['tipA'],param['tipR']]
        # unnest list and remove any empty terms and replace with empty string
        tip_params=['' if not i else i[0] for i in tip_params]
        # convert sensitivity to nm/V instead of nm/10 V
        if tip_params[1]:
            tip_params[1].__mul__(1/10.0)      
        tip_names=['SpringConstant','Sensitivity','Q','Length','Width','Angle','Radius']
        
        tip=dict(zip(tip_names,tip_params))
        x=dict(zip(['min','range','units'],[param['xmin'],param['xran'],param['xun']]))
        y=dict(zip(['min','range','units'],[param['ymin'],param['yran'],param['yun']]))
        z=dict(zip(['min','range','units'],[param['zmin'],param['zran'],param['zun']]))
        spec=dict(zip(['min','range','points','map'],[param['lnmin'],param['lnran'],param['lnpoints'],param['map']]))
        offset=dict(zip(['offset'],[param['offset']]))
        tune=dict(zip(['Frequency','PeakVal','QFactor','SpringConstant'],[param['freq'],param['peak'],param['Q'],param['k']]))
    
        parameters=dict(zip(['Tip','X','Y','Z','Spec','Offset','Tune'],[tip,x,y,z,spec,offset,tune]))
        
        dataout=[specData,imgData,sweepData]
        datanames=['Spec','Image','Sweep']
        
        valid_data=[val for num,val in enumerate(dataout) if val]
        valid_names=[datanames[num] for num,val in enumerate(dataout) if val]
        
        dataout=dict(zip(valid_names,valid_data))
        t_end=time.time()-t
        
        print('Elapsed Time: %3.2f sec\n' % t_end)
#    
#        def toPandaDF(user_dict):
#            import pandas as pd
#            df=pd.DataFrame.from_dict({(i,j): user_dict[i][j] 
#                for i in user_dict.keys() 
#                for j in user_dict[i].keys()},
#                orient='index').transpose()
#            return df

    
        if self.__dataframe:
            import pandas as pd
            self.data=self.__toPandaDF(dataout).unstack(level=0)
            self.specunits=pd.DataFrame(specUnits)
            self.param=self.__toPandaSeries(parameters)
        else:
            self.data=dataout
            self.param=parameters
#        self.data=dataout
        
    @staticmethod
    def __toPandaDF(user_dict):
        import pandas as pd
        df=pd.DataFrame.from_dict({(i,j): user_dict[i][j] 
            for i in user_dict.keys() 
            for j in user_dict[i].keys()},
            orient='index').transpose()
        return df
    
    @staticmethod
    def __toPandaSeries(user_dict):
        import pandas as pd
        df=pd.Series({(i,j): user_dict[i][j] 
            for i in user_dict.keys() 
            for j in user_dict[i].keys()})
        return df
            
    

        

    

        