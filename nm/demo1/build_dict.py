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
if len(sys.argv) >3:
    flanking = int(sys.argv[3])
# default 1
sel_ids = []

for i in range(4,len(sys.argv)):
    sel_ids.append(sys.argv[i])

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
#    ppms[-1] = ppms[-1][:,[0,2,1,3]]

seg_ppms = [ppm_segment(ppms[i],smooth=True, sp_size=sp_sizes[i],flank = flanking, shortest = None) for i in range(len(ppms)) ]

seg_ppms_lst = [seg_ppms[i][0] for i in range(len(ppmids))]
starts_lst = [seg_ppms[i][1] for i in range(len(ppmids))]
ends_lst = [seg_ppms[i][2] for i in range(len(ppmids))]
ics_lst = [seg_ppms[i][3] for i in range(len(ppmids))]
bic_lst = [seg_ppms[i][4] for i in range(len(ppmids))]
bigger =np.array( [np.mean(ics_lst[i])-bic_lst[i] for i in range(len(ppmids))])
bigger_mx_idx = np.argsort(-np.array(bigger))[0]
bigger_mx = bigger[bigger_mx_idx]
bigger_rate = bigger/bigger_mx

I1 = np.array(I1)
I2 = np.array(I2)

I1_rate = I1/I2
I2_rate = I2/I2.max()

#I1_rate[I1_rate < 0.1] -= 10

seg_max = np.array([len(seg_ppms[i][0]) for i in range(len(seg_ppms))])
seg_max_rate = seg_max/seg_max.max()

score = 1* bigger_rate + I1_rate + I2_rate + 0* seg_max_rate


thebest = np.argsort(-score)[0]

np.array(ppmids)[np.argsort(-score)]

autodetect = False

if len(sel_ids) == 0:
    autodetect = True
    sel_ppmids = [thebest]
elif sel_ids[0] == 'all':
    sel_ppmids = list(range(len(ppmids)))
else:
    sel_ppmids = []
    for sel_id in sel_ids:
        if sel_id in ppmids:
            for i in range(len(ppmids)):
                if ppmids[i] == sel_id:
                    sel_ppmids.append(i)
                    break
    if len(sel_ppmids) == 0:
        sel_ppmids = [thebest]
        autodetect = True


autofilter = autodetect
unselect= len(ppmids)        
while True:
    seg_ppms_lst = [seg_ppms[i][0] for i in sel_ppmids]
    starts_lst = [seg_ppms[i][1] for i in sel_ppmids]
    ends_lst = [seg_ppms[i][2] for i in sel_ppmids]
    ics_lst = [seg_ppms[i][3] for i in sel_ppmids]
    bic_lst = [seg_ppms[i][4] for i in sel_ppmids]
    ppmid_lst = [ppmids[i] for i in sel_ppmids]
    dictfile = 'layer' + str(layer)+ '/kernel-'+str(kernel)+'-dict.chen'
    segs_to_chen(ppmids=ppmid_lst, ppms=seg_ppms_lst, starts=starts_lst, ends=ends_lst, filepath=dictfile)
    os.system('bash  dict_dict.sh %s %s' %(layer, kernel))
    tomtomfile = 'layer' + str(layer)+ '/kernel-'+str(kernel)+'-dict-dict/tomtom.tsv'
    tomtom = pd.read_csv(tomtomfile,sep='\t',skipfooter=3,engine='python')
    tomtom = tomtom.loc[tomtom.loc[:,'Orientation']!='-',:]
    tomtom = tomtom.loc[tomtom.loc[:,'Orientation']!='-',:]
    print(tomtom)
    print(sel_ppmids)
    tomtom = tomtom.iloc[::-1,:]
    motif_gp = []
    for i in range(tomtom.shape[0]):
        qid = tomtom.iloc[i,0] #'Query_ID'
        tid = tomtom.iloc[i,1] #'Target_ID'
        q = float(tomtom.iloc[i,5]) #q-value
    #    qid = tomtom.loc[i,'Query_ID']
    #    tid = tomtom.loc[i,'Target_ID']
    #    q = tomtom.loc[i,'q-value']
        if q < 1e-2:
            isset = False
            for gp in motif_gp:
                if qid in gp or tid in gp:
                    isset = True
                    gp.add(qid)
                    gp.add(tid)
            if not isset:
                motif_gp.append(set([tid,qid]))
    print(motif_gp)
    ppm_list = []
    st_list = []
    ed_list =  []
    ppmid_list = []
    motifnames = []
    with h5py.File('layer' + str(layer)+ '/kernel-'+str(kernel)+'-unified-dict.h5','w') as motif_file:
        for gp in motif_gp:
            motifnames.extend(list(gp))
            motif_id = list(gp)[0]
            ppmid = motif_id.split('_')[0]
            ppmid_list.append(ppmid)
            for i in range(len(ppmids)):
                if ppmid == ppmids[i]: 
                    ppmid = i
                    break
            st = int(motif_id.split('_')[1])
            ed = int(motif_id.split('_')[2])
            ppm_list.append([ppms[ppmid][st:ed,:]]) 
            st_list.append([st]) 
            ed_list.append([ed])
            motif_file[motif_id] = ppms[ppmid][st:ed,:]
    dictfile = 'layer' + str(layer)+ '/kernel-'+str(kernel)+'-unified-dict.chen'
    segs_to_chen(ppmids=ppmid_list, ppms=ppm_list, starts=st_list, ends=ed_list, filepath=dictfile)
    seg_ppms_lst = [seg_ppms[i][0] for i in range(len(seg_ppms))]
    starts_lst = [seg_ppms[i][1] for i in range(len(seg_ppms))]
    ends_lst = [seg_ppms[i][2] for i in range(len(seg_ppms))]
#    segsfile = 'layer' + str(layer)+ '/kernel-'+str(kernel)+'-segs.chen'
#    segs_to_chen(ppmids=ppmids, ppms=seg_ppms_lst, starts=starts_lst, ends=ends_lst, filepath=segsfile)
    os.system('bash  ref_dict.sh %s %s' %(layer, kernel))
#    if not autodetect:
#        break
    motif_ppms = {}
    with h5py.File('layer' + str(layer)+ '/kernel-'+str(kernel)+'-unified-dict.h5','r') as motif_file:
        for k in motif_file.keys():
            motif_ppms[k] = motif_file[k][:]
    tomtomfile = 'layer' + str(layer)+ '/kernel-'+str(kernel)+'-segs-dict/tomtom.tsv'
    tomtom = pd.read_csv(tomtomfile,sep='\t',skipfooter=3,engine='python')
    tomtom = tomtom.loc[tomtom.loc[:,'Orientation']!='-',:]
    for j in range(len(motif_gp)-1,-1,-1):
        gp =list(motif_gp[j])
        for i in  range(len(gp)-1,-1,-1):
            sm = np.sum(tomtom['q-value'][tomtom.iloc[:,1] == gp[i]]<10e-5)
            print(sm)
            if sm <=1:
                motif_gp[j].remove(gp[i])
        if len(motif_gp[j]) == 0:
            del motif_gp[j] 
    matched_long_ppm = np.unique([ name.split('_')[0] for name in tomtom['Query_ID'][tomtom['q-value'] <1e-3]])
    if not autodetect:
        break
    sel = np.array([a not in  matched_long_ppm for a in ppmids])
    sortidx = np.argsort(-score[sel])
    new_dict_id = np.arange(len(ppmids))[sel][sortidx][0]
    print(unselect)
    print(sel.sum())
    print(unselect - sel.sum())
    print(sel_ppmids)
    if unselect - sel.sum() >= 3:
        sel_ppmids.append(new_dict_id)
#        aaa.append(sel.sum())
        unselect = sel.sum()
    else:
        autodetect = False
        sel_ppmids = sel_ppmids[:-1]



segnames = []
for i in range(len(ppmids)):
    if ppmids[i] in matched_long_ppm:
        segnames.extend([ ppmids[i] + '_' + str(starts_lst[i][j]) +'_' + str(ends_lst[i][j]) for j in range(len(starts_lst[i]))])

tomtomfile = 'layer' + str(layer)+ '/kernel-'+str(kernel)+'-segs-dict/tomtom.tsv'
tomtom = pd.read_csv(tomtomfile,sep='\t',skipfooter=3,engine='python')
tomtom = tomtom.loc[tomtom.loc[:,'Orientation']!='-',:]
#tomtom = tomtom['Query_ID'][tomtom['q-value']<10e-1]
matched_set = list(tomtom['Query_ID'])
unmatched_set = []
for seg in segnames:
    if seg not in matched_set:
        unmatched_set.append(seg)

unmatched_ppmids = [ dt.split('_')[0] for dt in unmatched_set]
unmatched_starts = [ int(dt.split('_')[1]) for dt in unmatched_set]
unmatched_ends = [ int(dt.split('_')[2]) for dt in unmatched_set]

unmatched_ppmids_lst = list(np.unique(unmatched_ppmids))
unmatched_starts_lst = []
unmatched_ends_lst = []
unmatched_ppm_lst = []


ppmids_dict = {}

for i in range(len(ppmids)):
    ppmids_dict[ppmids[i]] = i

for ppmid in unmatched_ppmids_lst:
    st_tmp = []
    ed_tmp = []
    ppm_tmp = []
    for i in range(len(unmatched_ppmids)):
        if unmatched_ppmids[i] == ppmid:
            st_tmp.append(unmatched_starts[i])
            ed_tmp.append(unmatched_ends[i])
            ppm_tmp.append(ppms[ppmids_dict[ppmid]][unmatched_starts[i]:unmatched_ends[i],:])
    unmatched_starts_lst.append(st_tmp) 
    unmatched_ends_lst.append(ed_tmp)
    unmatched_ppm_lst.append(ppm_tmp)

if len(unmatched_ppmids_lst) ==0:
    exit(0)
    
redictfile = 'layer' + str(layer)+ '/kernel-'+str(kernel)+'-redict.chen'
segs_to_chen(ppmids=unmatched_ppmids_lst, ppms=unmatched_ppm_lst, starts=unmatched_starts_lst, ends=unmatched_ends_lst, filepath=redictfile)
os.system('bash redict_redict.sh %s %s' %(layer, kernel))


tomtomfile = 'layer' + str(layer)+ '/kernel-'+str(kernel)+'-redict-redict/tomtom.tsv'
tomtom = pd.read_csv(tomtomfile,sep='\t',skipfooter=3,engine='python')
tomtom = tomtom.loc[tomtom.loc[:,'Orientation']!='-',:]
tomtom = tomtom.loc[tomtom['q-value']<1e-5,:]


adddict = []

while autofilter:
    ct = []
    for q in unmatched_set:
         ct.append((tomtom['Query_ID']==q).sum())
    most_match = np.argmax(np.array(ct))
    most_match_seg = unmatched_set[most_match]
    if ct[most_match] < len(ppmids)*0.3:
        autofilter = False
    else:
        most_match_seg = unmatched_set[most_match]
        adddict.append(most_match_seg)
        rs = list(tomtom['Query_ID'][tomtom['Target_ID']==most_match_seg])
        rs.extend(list(tomtom['Target_ID'][tomtom['Query_ID']==most_match_seg]))
        sel = [ tomtom.iloc[i,0] in rs or  tomtom.iloc[i,1] in rs for i in range(tomtom.shape[0])]
        tomtom = tomtom.loc[~np.array(sel),:]

for add in adddict:
    motif_gp.append(set([add]))

finaldict = [np.sort(list(motifs))[0] for motifs in motif_gp]
    
finaldict_ppmids = [ dt.split('_')[0] for dt in finaldict]
finaldict_starts = [ int(dt.split('_')[1]) for dt in finaldict]
finaldict_ends = [ int(dt.split('_')[2]) for dt in finaldict]

finaldict_ppmids_lst = list(np.unique(finaldict_ppmids))
finaldict_starts_lst = []
finaldict_ends_lst = []
finaldict_ppm_lst = []


ppmids_dict = {}

for i in range(len(ppmids)):
    ppmids_dict[ppmids[i]] = i

for ppmid in finaldict_ppmids_lst:
    st_tmp = []
    ed_tmp = []
    ppm_tmp = []
    for i in range(len(finaldict_ppmids)):
        if finaldict_ppmids[i] == ppmid:
            st_tmp.append(finaldict_starts[i])
            ed_tmp.append(finaldict_ends[i])
            ppm_tmp.append(ppms[ppmids_dict[ppmid]][finaldict_starts[i]:finaldict_ends[i],:])
    finaldict_starts_lst.append(st_tmp)
    finaldict_ends_lst.append(ed_tmp)
    finaldict_ppm_lst.append(ppm_tmp)


dictfile = 'layer' + str(layer)+ '/kernel-'+str(kernel)+'-unified-dict.chen'
segs_to_chen(ppmids=finaldict_ppmids_lst, ppms=finaldict_ppm_lst, starts=finaldict_starts_lst, ends=finaldict_ends_lst, filepath=dictfile)
os.system('bash ref_dict.sh %s %s' %(layer, kernel))

with h5py.File('layer' + str(layer)+ '/kernel-'+str(kernel)+'-unified-dict.h5','w') as motif_file:
    for gp in motif_gp:
        motif_id = list(gp)[0]
        ppmid = motif_id.split('_')[0]
        for i in range(len(ppmids)):
            if ppmid == ppmids[i]:
                ppmid = i
                break
        st = int(motif_id.split('_')[1])
        ed = int(motif_id.split('_')[2])
        motif_file[motif_id] = ppms[ppmid][st:ed,:]




