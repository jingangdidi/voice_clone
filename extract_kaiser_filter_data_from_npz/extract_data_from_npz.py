#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import numpy as np

# save list to txt
def SaveList(name, outlist):
    with open(name, 'w') as f:
        for i in outlist:
            f.write(i+'\n')

# extract data from python resampy package (resampy/data/)
for f in ['kaiser_fast', 'kaiser_best']:
    data = np.load(f+'.npz')
    print(data.files) # print array name

    print(data['half_window'])
    SaveList(f+'_half_window.txt', [str(i) for i in data['half_window']])

    print(data['precision'])
    SaveList(f+'_precision.txt', [str(data['precision'])])

    print(data['rolloff'])
    SaveList(f+'_rolloff.txt', [str(data['rolloff'])])
