import sys
layer = str(sys.argv[1])
kernel = str(sys.argv[2])
if len(sys.argv)> 3
    deepsea = int(sys.argv[3])
else:
    deepsea = 0
import numpy as np
import h5py
#layer = '10'
#kernel  = '0'
f = h5py.File('layer' + str(layer)+ '/kernel-'+kernel+'.ppm.h5','r')
actlist = []
for a in list(f.keys()):
    if a.startswith('act'):
        actlist.append(a)

actlist = [a.split('act')[1] for a in actlist]
psize = np.max(np.unique(np.array([a.split('_')[0] for a in actlist],dtype=int)))+1

actlist = []
conactlist = []
codelist = []
for i in range(psize):
    for j in range(psize):
        code = str(i)+'_'+str(j)
        if 'act'+code not in list(f.keys()):
            continue
#        print(code)
        actlist.append(f['act'+code][:].max())
        conactlist.append(f['conact'+code][:])
        codelist.append(code)

import pandas as pd

df = pd.DataFrame({'code':codelist,'act':actlist,'conact':conactlist})

tt = df.shape[0]

df = df.iloc[np.argsort(df.loc[:,'act'])[::-1][0:int(tt/4)],:]

tt = df.shape[0]

df = df.iloc[np.argsort(df.loc[:,'conact'])[::-1][0:int(tt/4)],:]



lines = []
smlines = []
lines1 = []
lines2 = []
for i in range(psize):
    for j in range(psize):
        code = str(i)+'_'+str(j)
        if 'act'+code not in list(f.keys()):
            continue
        act = f['act'+code][:]
        conact = f['conact'+code][:]
        ppm = f['ppm'+code][:]
        if deepsea:
            ppm = ppm[:,[0,2,1,3]]
        spnumb = act.shape[0]
        lines.append('>%04d_%04d_%d_%.4f_%.4f\n' %(i,j,spnumb, act.max(), conact))
        for k in range(ppm.shape[0]):
            lines.append('\t'.join(list(np.array(np.array(ppm[k,:]*1000,dtype=int),dtype=str))) + '\n')
        lines.append('\n')
        smppm = (ppm*spnumb + 80*np.ones((ppm.shape[0],4))*0.25)/(spnumb + 80)
        ic = - smppm * np.log(smppm)
        smlines.append('>%04d_%04d_%d_%.4f_%.4f\n' %(i,j,spnumb, act.max(), conact))
        if np.sum((2-ic.sum(axis=1))>1)<3:
            for k in range(ppm.shape[0]):
                smlines.append('\t'.join(list(np.array(np.array(smppm[k,:]*1000,dtype=int),dtype=str))) + '\n')
            smlines.append('\n')
        else:
            for k in range(ppm.shape[0]):
                smlines.append('\t'.join(list(np.array(np.array(ppm[k,:]*1000,dtype=int),dtype=str))) + '\n')
            smlines.append('\n')
        if np.sum((2-ic.sum(axis=1))>1)<3:
            continue
        if code not in np.array(df.loc[:,'code']):
            continue
        ismtf = (2-ic.sum(axis=1))>1
        for p in range(ismtf.shape[0]-1):
            if (not ismtf[p]) and ismtf[p+1]:
                st = p-2
                ed = p
                if st < 0:
                    st = 0
                ismtf[st:ed] = True
        for p in range(ismtf.shape[0]-2,-1,-1):
            if ismtf[p] and (not ismtf[p+1]):
                st = p+1
                ed = p+3
                if ed > ismtf.shape[0] - 1:
                    ed = ismtf.shape[0] - 1
                ismtf[st:ed] = True
        lst = []
        if ismtf[0]:
            lst.append(0)
        for p in range(ismtf.shape[0]-1):
            if (not ismtf[p]) and ismtf[p+1]:
                lst.append(p+1)
            if ismtf[p] and (not ismtf[p+1]):
                lst.append(p)
        if ismtf[-1]:
            lst.append(ismtf.shape[0]-1)
        print(lst)
        for p in range(0,len(lst),2): 
            if lst[p+1]-lst[p]+1 <= 8:
                continue
            lines1.append('>%04d_%04d_%d_%.4f_%.4f_%03d_%03d\n' %(i,j,spnumb, act.max(), conact,lst[p], lst[p+1]))
            lines2.append('>%04d_%04d_%d_%.4f_%.4f_%03d_%03d\n' %(i,j,spnumb, act.max(), conact,lst[p], lst[p+1]))
            for k in range(lst[p],lst[p+1]+1):
                lines1.append('\t'.join(list(np.array(np.array(ppm[k,:]*1000,dtype=int),dtype=str))) + '\n')
                lines2.append('\t'.join(list(np.array(np.array(smppm[k,:]*1000,dtype=int),dtype=str))) + '\n')
            lines1.append('\n')
            lines2.append('\n')

f.close()


with open('layer' + str(layer) + '/kernel-' + str(kernel) + '.all.ppm.chen' , 'w') as f:
    f.writelines(lines)

with open('layer' + str(layer) + '/kernel-' + str(kernel) + '.sm.all.ppm.chen' , 'w') as f:
    f.writelines(smlines)

with open('layer' + str(layer) + '/kernel-' + str(kernel) + '.sel.ppm.chen' , 'w') as f:
    f.writelines(lines1)

with open('layer' + str(layer) + '/kernel-' + str(kernel) + '.sm.sel.ppm.chen' , 'w') as f:
    f.writelines(lines2)
