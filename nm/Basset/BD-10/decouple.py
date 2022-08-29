import numpy as np
import h5py
import sys
import os
threads = sys.argv[1]
layer = sys.argv[2]
kernel_size = int(sys.argv[3])
motif_nb = None
if len(sys.argv) > 4:
    motif_nb = int(sys.argv[4])

from modeldef import *
from scipy import stats

kernel_nb,kernel_sz,pool_sz,input_bp, input_bps, model_list, act_model_list, gd = get_model_list(layer = int(layer), kernel = 0, weight_file='weight.hdf5')

poolsz = 1

for i in range(int(layer)-1):
    poolsz *= pool_sz[i]


if motif_nb is not None:
    os.system('bash idx.sh %d  | xargs -n 1 -P %s  bash decouple.sh %s %d' %(kernel_size, threads, layer, motif_nb))
else:
    motif_nb = 1
    while True:
        os.system('bash idx.sh %d  | xargs -n 1 -P %s  bash decouple.sh %s %d' %(kernel_size, threads, layer, motif_nb))
        print('bash idx.sh %d  | xargs -n 1 -P %s  bash decouple.sh %s %d' %(kernel_size, threads, layer, motif_nb))
        maxacts = []
        for i in range(kernel_size):
            ppmids = []
            if not os.path.exists('layer'+ layer +'/kernel-' +str(i) +'.ppm.h5'):
                continue
            f = h5py.File('layer'+ layer +'/kernel-' +str(i) +'.ppm.h5','r')
            print(i)
            for a in list(f.keys()):
                if a.startswith('conact'):
                    ppmids.append(a[len('conact'):])
            if len(ppmids) == 0:
                continue
            maxacts.append(np.max([f['conact'+ppmid][:]/np.max(f['act'+ppmid][:]) for ppmid in ppmids]))
        print(np.mean(maxacts)-1)
        if len(maxacts) < 10:
            statrs = [np.mean(maxacts) - 0.95 ,0.05]
        else:        
            statrs = stats.ttest_1samp(maxacts,popmean=0.95)
        print(statrs)
#        os.system('rm -rf layer%s/lock*' %(layer))
        if (statrs[0] > 0 and statrs[1] < 0.1) or poolsz <= 1:
            os.system('python merge.py %s %d' % (layer,motif_nb))
            break
        else:
            os.system('rm -rf layer%s/kernel*.ppm.h5' %(layer))
            os.system('rm -rf layer%s/lock*' %(layer))
            motif_nb += 1

 
