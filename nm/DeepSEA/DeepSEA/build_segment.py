import sys
import os
import pandas as pd
from segment import *       
import numpy as np
import h5py    
import sys

layer = str(sys.argv[1])
kernel = str(sys.argv[2])

flanking = 1
if len(sys.argv) > 3:
    flanking = int(sys.argv[3])
# default 1
#layer = '10'
#kernel = '1130'


f = h5py.File('layer' + str(layer)+ '/kernel-'+str(kernel)+'.ppm.h5','r')


ppmids = []

run_rounds = 0

for a in list(f.keys()):
    if a.startswith('conact'):
        rl =  len(a[len('conact'):].split('_'))
        if rl > run_rounds:
            run_rounds = rl

for a in list(f.keys()):
    if a.startswith('conact'):
        rl =  len(a[len('conact'):].split('_'))
        if rl == run_rounds:
            ppmids.append(a[len('conact'):])



I1 = []
I2 = []
sp_sizes = []
ppms = []
for ppmid in ppmids:
    I1.append(f['conact' + ppmid][0])
    I2.append(f['act' + ppmid][:].max())
    sp_sizes.append(f['index' + ppmid].shape[0])
    ppms.append(f['ppm' + ppmid][:])
    ppms[-1] = ppms[-1][:,[0,2,1,3]]

seg_ppms = [ppm_segment(ppms[i],smooth=True, sp_size=sp_sizes[i],flank = flanking, shortest = None) for i in range(len(ppms)) ]
seg_max = np.array([len(seg_ppms[i][0]) for i in range(len(seg_ppms))])



seg_ppms_lst = [seg_ppms[i][0] for i in range(len(seg_ppms))]
starts_lst = [seg_ppms[i][1] for i in range(len(seg_ppms))]
ends_lst = [seg_ppms[i][2] for i in range(len(seg_ppms))]

segsfile = 'layer' + str(layer)+ '/kernel-'+str(kernel)+'-segs.chen'
segs_to_chen(ppmids=ppmids, ppms=seg_ppms_lst, starts=starts_lst, ends=ends_lst, filepath=segsfile)
