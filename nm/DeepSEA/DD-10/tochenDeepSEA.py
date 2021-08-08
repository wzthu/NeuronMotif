import h5py
import numpy as np
import sys

layer = str(sys.argv[1])

pfmsf = h5py.File('layer' + str(layer)+ '/allppm.h5','r')
pfms=pfmsf['allppm'][:]
act = pfmsf['act'][:]
conact = pfmsf['conact'][:]
spnumb = pfmsf['spnumb'][:]
pfmsf.close()

lines=[]

for i in range(int(pfms.shape[0])):
    pfm = pfms[i,:,]
    if (pfm  != 0.25).sum()==0:
        continue
    lines.append('>%04d_%d_%.4f_%.4f\n' %(i,spnumb[i], act[i], conact[i]))
    for j in range(pfm.shape[0]):
        lines.append('\t'.join(list(np.array(np.array(pfm[j,[0,2,1,3]]*1000,dtype=int),dtype=str))) + '\n')
    lines.append('\n')

with open('layer' + str(layer)+'/ppm.chen' , 'w') as f:
    f.writelines(lines)
