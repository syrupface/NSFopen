# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:37:28 2019

@author: nelson
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

file='image.nid'

encode='ANSI'

fid=open(file,'rb')
data_in=fid.read().split(b'\n') 
fid.close()

print("Reading Header")
end_of_header=0;
while True:
    if (str(data_in[end_of_header],encode)=='\r') and (str(data_in[end_of_header+1],encode)=='\r'):
        break
    end_of_header+=1
    
header=[str(line,encode).strip() for line in data_in[:end_of_header]];
df=dict()
for head in header:
    parse=re.search(r'(.*)\=(.*)',head)
    if parse:
        parse=parse.groups()
        if not parse[0].startswith('ImSoVideoTVConfiguration'):
            if re.search('\[.*\]',parse[1]):

                parse2=re.search('(\S)\[(.*)\]\*\[(.*)\]',parse[1]).groups()
                ty,val,unit=parse2
                if ty=='D':
                    val=float(val)
                if ty=='B':
                    val=bool(val)
                if ty=='L':
                    val=int(val)
                if ty=='V':
                    val=re.split(',',val)
                    val=[float(v) for v in val]
                df.update({parse[0]:{'Value':val,'Unit':unit}})
            else:
                df.update({parse[0]:{'Value':parse[1],'Unit':unit}})

