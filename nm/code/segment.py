import sys
import numpy as np

def ppm_smooth(ppm, sp_size, M = 80):
    return (ppm*sp_size + M*np.ones((ppm.shape[0],4))*0.25)/(sp_size + M)

def ppm_segment0(ppm, ic_min = 1, smooth=False, sp_size = None, M = 30, flank = 1, extend = 0, shortest = None):
    if shortest is None:
        shortest = int(flank*1.5)  + 3
    ppm0 = ppm.copy()
    ppm1 = ppm_smooth(ppm = ppm, sp_size = sp_size, M = 1)
    ic0 = - ppm1 * np.log(ppm1)
    ic0 = 2-ic0.sum(axis=1)
    if smooth:
        ppm = ppm_smooth(ppm = ppm, sp_size = sp_size, M = M)
    ic = - ppm * np.log(ppm)
    ic = 2-ic.sum(axis=1)
    sm3 = [ic[(i-1):(i+2)].mean() for i in range(1,ic.shape[0] - 2)]
    return None


def ppm_segment(ppm, ic_min = 1, smooth=False, sp_size = None, M = 30, flank = 0, extend = 0, shortest = None):
    if shortest is None:
        shortest = int(flank*1.5)  + 3
    ppm0 = ppm.copy()
    ppm1 = ppm_smooth(ppm = ppm, sp_size = sp_size, M = 1)
    ic0 = - ppm1 * np.log(ppm1)
    ic0 = 2-ic0.sum(axis=1)
    if smooth:
        ppm = ppm_smooth(ppm = ppm, sp_size = sp_size, M = M)
    ic = - ppm * np.log(ppm)
    ic = 2-ic.sum(axis=1) 
    sm3 = [ic[(i-1):(i+2)].mean() for i in range(1,ic.shape[0] - 1)]  
    s = [0]
    s.extend(sm3)
    s.append(0)
    ic = np.array(s)
#    ic1 = (ic-1)*0.99+1
#    ic = ic1/(2-ic1)
#    sm3 = [ic[(i-1):(i+2)].mean() for i in range(1,ic.shape[0] - 1)]
#    s = [0]
#    s.extend(sm3)
#    s.append(0)
#    ic = np.array(s)
    ismtf = ic > ic_min
    for p in range(ismtf.shape[0]-1):
        if (not ismtf[p]) and ismtf[p+1]:
            print(p)
            st = p-(flank-1)
            ed = p + 1
            if st < 0:
                st = 0
            ismtf[st:ed] = True
    for p in range(ismtf.shape[0]-2,-1,-1):
        if ismtf[p] and (not ismtf[p+1]):
            print(p)
            st = p + 1
            ed = p + flank + 1 
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
            lst.append(p+1)
    if ismtf[-1]:
        lst.append(ismtf.shape[0]-1)
    print(lst)
    ppm_segs = []
    starts = []
    ends = []
    ppm_segs_ics = [] 
    for p in range(0,len(lst),2): 
        if lst[p+1]-lst[p] < shortest:
            continue
        st = lst[p]
        if extend > 0:
            if p == 0:
                if st - extend < 0:
                    st = 0
                else:
                    st -= extend
            else:
                if st - extend <= lst[p-1] + extend:
                    st = int((lst[p-1] + st)/2)+1
                else:
                    st -= extend
        ed = lst[p+1]
        if extend >0:
            if p+1 == len(lst)-1:
                if ed + extend > ismtf.shape[0] - 1:
                    ed = ismtf.shape[0] - 1
                else:
                    ed += extend
            else:
                if ed + extend >= lst[p+2] - extend:
                    ed = int((lst[p+2] + ed)/2)
                else:
                    ed += extend
        ppm_segs.append(ppm0[lst[p]:(lst[p+1]),:])
        starts.append(st)
        ends.append(ed)
        ppm_segs_ics.append(ic0[lst[p]:(lst[p+1])].mean())
    return ppm_segs, starts, ends, ppm_segs_ics, ic0[ismtf == False].mean()

def segs_to_chen(ppmids, ppms, starts, ends, filepath):
    lines = []
    for i in range(len(ppms)):
        print(i)
        for j in range(len(ppms[i])):
            print(j)
            lines.append('>%s_%d_%d\n' %(ppmids[i],starts[i][j], ends[i][j]))
            ppm = ppms[i][j]
            for k in range(ppm.shape[0]):
                print(k)
                lines.append('\t'.join(list(np.array(np.array(ppm[k,:]*1000,dtype=int),dtype=str))) + '\n')
    with open(filepath, 'w') as f:
        f.writelines(lines)
        
    
