import sys
import os
import pandas as pd
from segment import *       
import numpy as np
import h5py    
import sys

layer = str(sys.argv[1])
kernel = str(sys.argv[2])
strict = 1
if len(sys.argv) > 3:
    strict = 0
#layer = '10'
#kernel = '1130'

motif_ppms = {}

with h5py.File('layer' + str(layer)+ '/kernel-'+str(kernel)+'-unified-dict.h5','r') as motif_file:
    for k in motif_file.keys():
        motif_ppms[k] = motif_file[k][:]
#        motif_ppms[k] = motif_ppms[k][:,[0,2,1,3]]


tomtomfile = 'layer' + str(layer)+ '/kernel-'+str(kernel)+'-segs-dict/tomtom.tsv'

tomtom = pd.read_csv(tomtomfile,sep='\t',skipfooter=3,engine='python')

tomtom = tomtom.loc[tomtom.loc[:,'Orientation']!='-',:]

tomtom = tomtom.loc[tomtom.loc[:,'Orientation']!='-',:]

pos = {}
motif = {} 

maxed = 0

for i in range(tomtom.shape[0]):
    qid = tomtom.iloc[i,0] #'Query_ID'
    tid = tomtom.iloc[i,1] #'Target_ID'
    offset = int(tomtom.iloc[i,2]) #'Optimal_offset'
    qv = float(tomtom.iloc[i,5]) #q-value
    ov = int(tomtom.iloc[i,6]) #'Overlap'
    if qv > 1e-3 or ov/motif_ppms[tid].shape[0] < 0.5:
        continue
    strs = qid.split('_')
    ppmid = strs[0]
    st = int(strs[1])
    ed = int(strs[2])
    if ed > maxed:
        maxed = ed
    if pos.get(ppmid) is None:
        pos[ppmid] = []
        motif[ppmid] = []
    pos[ppmid].append(st - offset)
    motif[ppmid].append(tid)

motif_single_pos = {}
motif_pair_pos = {}
motif_pair_dis = {}
pos_rename = {}
motif_rename = {}
motif_sort = {}

for k in pos.keys():
    st = np.array(pos[k],dtype=int)
    mtf = np.array(motif[k],dtype=str)
    mtf = mtf[st.argsort()]
    st.sort() 
    for i in range(st.shape[0]):
        motif_name = mtf[i]
        if motif_single_pos.get(motif_name) is None:
            motif_single_pos[motif_name] =  []
        motif_single_pos[motif_name].append(st[i])   
    for i in range(st.shape[0]-1):
        if st[i+1] - st[i] < motif_ppms[mtf[i]].shape[0]:
            continue
        pair_name = mtf[i] + '-' +  mtf[i+1]
        if motif_pair_dis.get(pair_name) is None:
            motif_pair_dis[pair_name] = []
            motif_pair_pos[pair_name] = []
        motif_pair_dis[pair_name].append(st[i+1] - st[i]) 
        motif_pair_pos[pair_name].append(st[i])
    mtf0 = list(mtf)
    for motif_name in motif_ppms.keys():
        sel = mtf == motif_name
        count = 1
        for j in np.arange(sel.shape[0])[sel]:
            mtf0[j] += '_'
            mtf0[j] += str(count)
            count += 1
    pos_rename[k] = st
    motif_rename[k] = mtf0
    motif_sort[k] = list(mtf)

all_motifs = []
mx = 0
for k,v in  zip(motif_rename.keys(),motif_rename.values()):
    if len(v)> mx :
        mx = len(v)
        print(k)
        print(v)
    all_motifs.extend(v)

all_motifs = np.unique(all_motifs)

seqs = {}
mcbs = {}
for mtfsz in range(mx,0,-1):
    for k in motif_sort.keys():
        if len(motif_sort[k]) != mtfsz:
            continue
        print(k)
        motif_cb_names = motif_sort[k]
        motif_cb_name = '-'.join(motif_cb_names)
        if seqs.get(motif_cb_name) is not None:
            seqs[motif_cb_name].add(k)
#            mcbs[motif_cb_name].append(motif_cb_names)
        else:
            aligned = False
            for s in seqs.keys():
                mcb = mcbs[s]
                mr = 0
                m = 0
                while mr < len(mcb) and m < len(motif_cb_names):
                    if mcb[mr] == motif_cb_names[m]:
                        mr += 1
                        m += 1
                    else:
                        mr += 1
                if m == len(motif_cb_names):
                    aligned = True
                    seqs[s].add(k)
#                    mcbs[motif_cb_name].append(motif_cb_names)
            if aligned == False:
                seqs[motif_cb_name]=set([k])
                mcbs[motif_cb_name]= motif_cb_names

def list_all(lst, x = 0, x_pos=[]):
    if x == 0:
        one_lst = lst.copy()
        for i in range(len(x_pos)-1,-1,-1):
            one_lst.insert(x_pos[i],'X')
        return [one_lst]
    lst_lst = []
    for i in range(len(lst)+1):
        new_x_pos = x_pos.copy()
        new_x_pos.append(i)
        lst_lst.extend(list_all(lst=lst,x=x-1,x_pos=new_x_pos))
    return lst_lst

mcb_dist = {}
mcb_pos = {}
mcb_match = {}

for mcb_name in mcbs.keys(): 
    mcb = mcbs[mcb_name]
    ppmids = seqs[mcb_name]
    mcb_poss = []
    for k in pos_rename.keys():
        values = pos_rename[k]
        if len(values) == len(mcb):
            matched = True
            for i in range(len(mcb)):
                if mcb[i] != motif_sort[k][i]:
                    matched = False
                    break
            if matched:
                mcb_poss.append(values)
    ave_pos = np.zeros(len(mcb))
    new_mcb_poss = []
    new_dist = []
    new_match = []
    for i in range(len(mcb)):
        ave_pos[i] = np.mean([a[i] for a in mcb_poss])     
        new_mcb_poss.append([])
        if i < len(mcb)-1:
            new_dist.append([])
    for ppmid in ppmids:
        motif_seq = motif_sort[ppmid]
        all_motif_seqs = list_all(motif_seq,x = len(mcb)-len(motif_seq))
        scores = []
        for one_motif_seq in all_motif_seqs:
            matched = True
            sc = 0
            count = 0
            for i in range(len(mcb)):
                if one_motif_seq[i] != 'X' and one_motif_seq[i] != mcb[i]:
                    matched = False
                    break
                if one_motif_seq[i] != 'X':
                    s = (pos_rename[ppmid][count] - ave_pos[i])
                    sc += s*s
                    count += 1
            if matched == True:
                scores.append(sc)
            else:
                scores.append(1e10)
        scores = np.array(scores)
        sel_id = scores.argmin()
        one_motif_seq = all_motif_seqs[sel_id]
        new_match.append(one_motif_seq)
        count = 0
        for i in range(len(mcb)):
            if one_motif_seq[i] != 'X':
                new_mcb_poss[i].append(pos_rename[ppmid][count])
                if i+1 < len(mcb)  and one_motif_seq[i+1] != 'X':
                    if pos_rename[ppmid][count+1]-pos_rename[ppmid][count]==1:
                        print(ppmid)
                    new_dist[i].append(pos_rename[ppmid][count+1]-pos_rename[ppmid][count])
                count += 1
    mcb_pos[mcb_name] = new_mcb_poss
    mcb_dist[mcb_name] = new_dist
    mcb_match[mcb_name]  = new_match
        
                
        
from modeldef import *
from utils import *
layer = int(layer)
kernel = int(kernel)
kernel_nb,kernel_sz,pool_sz,input_bp, input_bps, model_list, act_model_list, gd = get_model_list(layer = layer, kernel = int(kernel), weight_file='weight.hdf5')

submodel = model_list[-1]
presubmodel = model_list[-2]


max_pool_sz = 1
for i in range(int(layer)-1):
    max_pool_sz *= pool_sz[i]


def gen_sample(pwms, seq_size , starts):
    sp_size = starts.shape[1]
    sample = np.zeros((sp_size,seq_size, 4))
    ids = np.random.choice(4,sp_size*seq_size)
    ids = np.reshape(ids, (sp_size,seq_size))
    for i in range(4):
        sel = np.where(ids == i)
        sample[(sel[0],sel[1],i)] = 1
    for i in range(starts.shape[0]):
        pwm = pwms[i]
        pwmids = np.concatenate([np.random.choice(4,sp_size, p = pwm[j,:])[:,np.newaxis] for j in range(pwm.shape[0])], axis=1)
        pwmmotif = np.zeros((sp_size,pwm.shape[0], 4))
        for j in range(4):
            sel = np.where(pwmids == j)
            pwmmotif[(sel[0],sel[1],j)] = 1
        for j in range(starts.shape[1]):
            sample[j,starts[i,j]:(starts[i,j]+pwm.shape[0]),:] = pwmmotif[j,:,:]
    return sample

            
'''        

sample_size = 10000


for mcb_name in mcbs.keys():
    mcb = mcbs[mcb_name]
    pos_st = [min(a) for a in mcb_pos[mcb_name]]
    pos_ed = [max(a) for a in mcb_pos[mcb_name]]
    dis_min = [min(a) for a in mcb_dist[mcb_name]]
    dis_max = [max(a) for a in mcb_dist[mcb_name]]
    new_dis_min = dis_min.copy()
    new_dis_max = dis_max.copy()
    for i in range(len(new_dis_min)):
        p = [int(p) for p in mcb[i].split('_')]
        p = p[2]-p[1]
        new_dis_min[i] = max([new_dis_min[i]-max_pool_sz,p])
        new_dis_max[i] = max([new_dis_max[i]+max_pool_sz,p])
    bg_sample_dis = [np.random.choice(np.arange(dis_min[i],dis_max[i]+1),sample_size)[np.newaxis,:] for i in range(len(dis_min))]
    bg_sample_dis = np.concatenate(bg_sample_dis,axis=0)
    bg_sample_dis_sum = bg_sample_dis.sum(axis=0)
    p = [int(p) for p in mcb[-1].split('_')]
    p = p[2]-p[1]
    bg_sample_dis_sum += p
    bg_sample_dis = bg_sample_dis[:,bg_sample_dis_sum < input_bp]
    bg_sample_dis_sum = bg_sample_dis_sum[bg_sample_dis_sum < input_bp]
    bg_sample_pos = input_bp - bg_sample_dis_sum
    bg_sample_pos  = np.array([ int(np.random.choice(pos,1)) for pos in bg_sample_pos ])
    pwms = [motif_ppms[m] for m in mcb]
    bg_sample = gen_sample(pwms=pwms, seq_size=input_bp , starts=bg_sample_dis  + bg_sample_pos)
    bg_sample_act =  submodel.predict(bg_sample, batch_size=1000)[:,0,int(kernel)]
    for i in range(len(new_dis_min)):
        new_d = np.arange(new_dis_min[i],new_dis_max[i]+1)
        new_d = new_d[np.logical_or(new_d<dis_min[i],new_d>dis_max[i])]
        for d in new_d:
            fg_sample_dis = bg_sample_dis.copy()
            fg_sample_dis[i,:] = d
            fg_sample_dis_sum = fg_sample_dis.sum(axis=0)
            fg_sample_dis_sum += p
            fg_sample_dis = fg_sample_dis[:,fg_sample_dis_sum < input_bp]
            fg_sample_dis_sum = fg_sample_dis_sum[fg_sample_dis_sum < input_bp]
            fg_sample_pos = input_bp - fg_sample_dis_sum
            fg_sample_pos  = np.array([ int(np.random.choice(pos,1)) for pos in fg_sample_pos ])
            fg_sample = gen_sample(pwms=pwms, seq_size=input_bp , starts=fg_sample_dis  + fg_sample_pos)
            fg_sample_act =  submodel.predict(fg_sample, batch_size=1000)[:,0,int(kernel)-1]
            print('%d\t%d\t%f'%(i,d,fg_sample_act.mean()))
        
'''        






html_txt = '''
<html lang="en">
<style>

tr td,th{

border:1px solid grey;

}

.mt{

 border-collapse:collapse;

}

</style>
    
    <br/>
    <body>
        <a href='tomtom_dict_%d/tomtom.html'>Dictionary for neuron %d in layer %d</a><br/>
        <a href='%d.html'>PWMs represented by this neuron</a><br/>
        %s<br/>
        Please wait patiently for all motif logos or patterns of CN motifs to load ...
        %s
    </body>
    Visit NeuronMotif website for full results:
    <a href="https://wzthu.github.io/NeuronMotif/">https://wzthu.github.io/NeuronMotif/</a><br/>
    The motifs/motif combinations occur in previous layers:
    %s
        <script src="https://wzthu.github.io/NeuronMotif/jseqlogo.js"></script>
        <script>
            var options = {
                "colors": jseqlogo.colors.nucleotides
            };

            %s
        </script>
</html>
'''
# <p>The neuron grammar in previous layer match to this neuron grammar:<p>
def ppm2js(ppm, ppm_id, width, height):
    ppm += 0.0001
    v = ppm.sum(axis=1)
    v = v.repeat(4).reshape(ppm.shape)
    ppm /= v
    ppm0 = ppm.copy()
    vlg = -ppm *np.log2(ppm)
    ppm = ppm *(2 -vlg.sum(axis=1)).repeat(4).reshape(ppm.shape)
    A ='"A": [' + ','.join(['%1.2f' % (p)  for p in ppm[:,0]]) + '],'
    C ='"C": [' + ','.join(['%1.2f' % (p)  for p in ppm[:,1]]) + '],'
    G ='"G": [' + ','.join(['%1.2f' % (p)  for p in ppm[:,2]]) + '],'
    T ='"T": [' + ','.join(['%1.2f' % (p)  for p in ppm[:,3]]) + ']'
    html = 'var data = {%s};' % (A+C+G+T)
    html += 'sequence_logo(document.getElementById("%s"), %d,%d, data, options);' % (ppm_id,width,height)
    return html

def get_subtree_idxs(lst):
    if not isinstance(lst, list):
        return [lst]
    all_idxs = []
    for l in lst:
        if isinstance(l, list):
            all_idxs.extend(get_subtree_idxs(l))
        else:
            all_idxs.append(l)
    return all_idxs

def get_tree_depth(lst, depth=1):
    if not isinstance(lst, list):
        return depth
    d_max = depth
    for l in lst:
        d = get_tree_depth(l, depth+1)
        if d > d_max:
            d_max = d
    return d_max


tree = {}
node = {}
leaf = {}

for mcb_name in mcbs.keys():
    subtree = {}
    mcb = mcbs[mcb_name]
    matched = mcb_match[mcb_name]
    matched = np.array(matched)
    pos_st = [min(a) for a in mcb_pos[mcb_name]]
    pos_ed = [max(a) for a in mcb_pos[mcb_name]]
    dis_min = []
    dis_max = []
    for a in mcb_dist[mcb_name]:
        m = np.mean(a)
        tmp = np.array(a)
        sel = np.abs(tmp-m) <=4
        if np.sum(sel)>len(a)*0.8:
            tmp  = tmp[sel]
        dis_min.append(int(tmp.min()))
        dis_max.append(int(tmp.max()))
#    dis_min = [min(a) for a in mcb_dist[mcb_name]]
#    dis_max = [max(a) for a in mcb_dist[mcb_name]]
    if strict == 0:
        for i in range(len(dis_min)):
            if dis_max[i]-dis_min[i] >= 4:
                if pos_st[i+1] - pos_ed[i] < dis_min[i]:
                    dis_min[i] = max(pos_st[i+1] - pos_ed[i],motif_ppms[mcb[i]].shape[0])
                if pos_ed[i+1] - pos_st[i] > dis_max[i]:
                    dis_max[i] = max(pos_ed[i+1] - pos_st[i],motif_ppms[mcb[i]].shape[0])
    dis_range = [dis_max[i] -dis_min[i] for i in range(len(dis_min))]
    dis_range = np.array(dis_range)
    dis_mean = [dis_max[i]/2 +dis_min[i]/2 for i in range(len(dis_min))]
    dis_mean = np.array(dis_mean)
    dis_range_txt = []
    for i in range(len(dis_max)):
        if dis_max[i] == dis_min[i]:
            dis_range_txt.append('Gap size: ' + str(dis_max[i]-motif_ppms[mcb[i]].shape[0])+'bp')
        else:
            dis_range_txt.append('Gap range:' + str(dis_min[i]-motif_ppms[mcb[i]].shape[0]) +'~'+ str(dis_max[i]-motif_ppms[mcb[i]].shape[0]) +'bp')
    tree_nodes = []
    node_names = {}
    leaf_names = []
    for i in range(len(mcb)):
        tree_nodes.append(i)
        node_names[str(i)] = mcb_name+'-' + str(i) 
        leaf_names.append(mcb_name+'-' + str(i))
    for i in range(len(dis_min)):
        idx = dis_range.argmin()
        if np.sum(dis_range[idx]==dis_range) > 1:
            sel_idx = np.where(dis_range[idx]==dis_range)
            idx = sel_idx[0][dis_mean[sel_idx].argmin()]
        new_node = [tree_nodes[idx],tree_nodes[idx+1]]
        allidxs1 =  get_subtree_idxs(tree_nodes[idx])
        allidxs2 =  get_subtree_idxs(tree_nodes[idx+1]) 
        X1 = (matched[:,allidxs1] == 'X').sum(axis=1)
        X2 = (matched[:,allidxs2] == 'X').sum(axis=1)
        link = None
        if np.sum(X1<len(allidxs1))==matched.shape[0]  and np.sum(X2<len(allidxs2))==matched.shape[0]:
            link = 'AND'
        elif np.sum(X1<len(allidxs1))==matched.shape[0]:
            link = 'LEFT/AND'
        elif np.sum(X2<len(allidxs2))==matched.shape[0]:
            link = 'RIGHT/AND'
        elif np.logical_or(np.logical_and(X1 == len(allidxs1), X2<len(allidxs2)), np.logical_and(X2 == len(allidxs2), X1<len(allidxs1))).sum()==matched.shape[0]:
            link = 'XOR'
        else:
            link = 'OR'
#        node_names['_'.join([str(i) for i in get_subtree_idxs(new_node)])] = link + ':' + dis_range_txt[idx]
        node_names['_'.join([str(i) for i in get_subtree_idxs(new_node)])] =  dis_range_txt[idx]
        del tree_nodes[idx+1]
        del tree_nodes[idx]
        tree_nodes.insert(idx,new_node)
        dis_range = list(dis_range)
        del dis_range[idx]
        del dis_range_txt[idx]
        dis_range = np.array(dis_range)
    tree[mcb_name] = tree_nodes[0]
    node[mcb_name] = node_names
    leaf[mcb_name] = leaf
        
nodes = node

test = False
#test = True

if not test:
    occur = {}
    for k in nodes.keys():
        occur[k]  = {}
        for k1 in nodes[k].keys():
            occur[k][k1] = [str(layer) + '_0']
                


    for tl in range(len(model_list)-1):
    #for tl in [3]:
        for i in range(kernel_nb[tl]):
            if not  os.path.exists('layer' + str(tl+1) +'/kernel-'+str(i)+'-segs.meme'):
                continue
            os.system('tomtom --norc  layer' + str(tl+1) +'/kernel-'+str(i)+'-segs.meme layer'+str(layer)+'/kernel-'+str(kernel)+'-unified-dict.meme -oc layer'+str(layer)+'/kernel-'+ str(kernel) +'-segs-segs')
            tomtom = pd.read_csv('layer'+str(layer)+'/kernel-'+ str(kernel) +'-segs-segs/tomtom.tsv',sep='\t',skipfooter=3,engine='python')
            tomtom = tomtom.loc[tomtom.loc[:,'Orientation']!='-',:]
            tomtom = tomtom.loc[tomtom.loc[:,'q-value']<1e-5,:]
            pwmid_list = [tomtom.iloc[i,0].split('_')[0] for i in range(tomtom.shape[0])]
            pwmid_list = np.array(pwmid_list)
            pwmid_unique = np.unique(pwmid_list)
            for pwmid_i in range(pwmid_unique.shape[0]):
                tg = np.array(tomtom.loc[pwmid_list == pwmid_unique[pwmid_i],'Target_ID'])
                qr = np.array(tomtom.loc[pwmid_list == pwmid_unique[pwmid_i],'Query_ID'])
                for k in nodes.keys():
                    name = np.array(k.split('-'))
                    for k1 in nodes[k].keys():
                        node_key_idx = np.array([int(s)for s in k1.split('_')])
                        node_motifs = name[node_key_idx]
                        node_motifs_uniqued = np.unique(node_motifs)
                        flag = True
                        for nm in node_motifs_uniqued:
                            if np.sum(nm == node_motifs) > np.sum(nm == tg):
                                flag = False
                                break
                        if flag:
                            occur[k][k1].append(str(tl+1)+'_' +str(i))


    start_layer = {}
    for k in occur.keys():
        start_layer[k] = {}
        for k1 in nodes[k].keys():
            if len(occur[k][k1])>0:
                start_layer[k][k1] = np.min([int(a.split('_')[0]) for a in occur[k][k1]])
            else:
                start_layer[k][k1] = -1
                        

html_table = ''        

jsscripts = []
for mcb_name in mcbs.keys():
    mcb = mcbs[mcb_name]
    for i in range(len(mcb)):
        jsscripts.append(ppm2js(motif_ppms[mcb[i]], mcb_name+'-'+str(i),motif_ppms[mcb[i]].shape[0]*8,50))

for mcb_name in mcbs.keys():
    mcb = mcbs[mcb_name]
    for k in nodes[mcb_name].keys():
        node_idx = [ int(v) for v in k.split('_')]
        for i in range(len(node_idx)):
            jsscripts.append(ppm2js(motif_ppms[mcb[node_idx[i]]], mcb_name+'-'+str(node_idx[i])+'-node-'+k,motif_ppms[mcb[node_idx[i]]].shape[0]*8,50))


jsscripts = '\n'.join(jsscripts)


html_rows = []

html_table = '<table border=0><tr>'
mcb_i = 0
for mcb_name in mcbs.keys():
    mcb = mcbs[mcb_name]
    queue = []
    queue_depth = [1]
    queue.append(tree[mcb_name])
    counter = 0
    html_row = ''
    max_depth = get_tree_depth(tree[mcb_name])
    logo_widths = np.array([motif_ppms[mcb[i]].shape[0] + 8 for i in range(len(mcb))])
    if max_depth == 1:
        queue_flag = [1]
    else:
        queue_flag = [0]
    while len(queue) > 0:
        if counter%len(mcb) == 0:
            html_row += '<tr>'
        node = queue[0]
        d = queue_depth[0]
        flag = queue_flag[0]
        del queue[0]
        del queue_depth[0]
        del queue_flag[0]
        if isinstance(node,list):
            node_idx = get_subtree_idxs(node)
            node_idx_int = np.array(node_idx)
            node_idx = [str(i) for i in node_idx]
            left_branch = node_idx[0]
            for i in range(len(node_idx)):
                if '_'.join(node_idx[0:(i+1)]) in list(nodes[mcb_name].keys()):
                    left_branch = node_idx_int[0:(i+1)]
            total_len = logo_widths[node_idx_int].sum()
            left_len = logo_widths[left_branch].sum()
            left_space = ''.join(['&nbsp']*int(left_len*0.1))
            span = len(node_idx)
            counter += span
            if not test:
                oclayer = str(start_layer[mcb_name]['_'.join(node_idx)])
                if int(oclayer) < int(layer):
                    oclayer += '~'
                    oclayer += str(layer)
            else:
                oclayer = ''               
            html_row += '<td align=center colspan="%d">%s</td>'%(span,left_space+nodes[mcb_name]['_'.join(node_idx)]+'<br/>'+left_space+'combination occurs in <br/>'+ left_space+ ' layer '+oclayer+'<br/>'+left_space+'<img src="https://wzthu.github.io/NeuronMotif/split.png" height="30px" width="%dpx">' % (int(total_len*6)))
            for subnode in node:
                queue.append(subnode)
                queue_depth.append(d+1)
                queue_flag.append(1)
        else:
            counter += 1
#            check = False
#            for subnode in queue:
#                if isinstance(subnode,list):
#                    check = True
#                    break
#            if check or d != max_depth:
            if d != max_depth:
                queue.append(node)
                queue_depth.append(d+1)
                queue_flag.append(0)
            if flag == 1:
                if not test:
                    oclayer = str(start_layer[mcb_name][str((counter-1)%len(mcb))])
                    if int(oclayer) < int(layer):
                        oclayer += '~'
                        oclayer += str(layer)
                else:
                    oclayer = ''
                html_row += ( '<td align=center valign=bottom rowspan="%d">motif occurs in<br/>layer %s<br/>&nbsp&nbsp&nbsp&nbsp<canvas id="%s"></canvas>&nbsp&nbsp&nbsp&nbsp</td>'%(max_depth - d + 1,oclayer, mcb_name+'-'+str(node)))
        if counter%len(mcb) == 0:
            html_row += '</tr>\n'
    html_row += '<tr>'
#    for m in range(len(mcb_name.split('-'))):
#        oclayer = str(start_layer[mcb_name][str(m)])
#        if int(oclayer) < int(layer):
#            oclayer += '~'
#            oclayer += str(layer)
#           html_row += '<td align=center colspan="1">%s</td>'%('occurs in layer '+oclayer)
    html_row += '</tr>\n'
    match_table = np.array(mcb_match[mcb_name])!='X'
    match_table =  match_table*1
    for i in range(match_table.shape[1]):
        for j in range(i):
            match_table[:,i]*=2
    match_value = match_table.sum(axis=1)
    match_value0 = np.unique(match_value)
    match_value0 = match_value0[match_value0>0]
    match_table = np.array([match_table[match_value==v,:][0,:]>0 for v in match_value0])
    for i in range(match_table.shape[0]):
        html_row += '<tr>'
        for j in range(match_table.shape[1]):
            color = '#F19473'
            if match_table[i,j]:
                color = '#80FD91'
            html_row += ('<td style="background-color:%s">' % (color))
            html_row += str(match_table[i,j])[0]
            html_row += '</td>'
        html_row += '</tr>'
    html_table += '<td><table class="mt">%s</table></td>'%(html_row)
    mcb_i += 1
    if mcb_i < len(mcbs.keys()):
        html_table += '<td>OR</td>'

html_table += '<tr/><table/>'

if test:
    html_whole_txt = html_txt % (int(kernel),int(kernel),int(layer),int(kernel),'ucsc_link',html_table,'',jsscripts)
    with open( 'layer'+str(layer)+'/tree.'+str(kernel)+'.test.html'  ,'w') as ftest:
        ftest.write(html_whole_txt)
    exit(0)

html_link_table_temple = '<table><tr><td align=left valign=middle>&nbsp&nbsp&nbsp%s</td></tr><tr><td>%s</td></tr></table>'

html_link_nodes = []

for mcb_name in mcbs.keys():
    mcb = mcbs[mcb_name]
    node = nodes[mcb_name]
    oc = occur[mcb_name]
    st = start_layer[mcb_name]
    for k in node.keys():
        node_list_idx = [int(n) for n in k.split('_')]
        node_motif = '&nbsp&nbsp+&nbsp&nbsp'.join(['<canvas id="%s"></canvas>'%(mcb_name+'-'+str(n)+'-node-'+ k) for n in k.split('_')])
        node_start_layers = ''
        for ly in range(st[k],int(layer),1):
            xs = np.array([int(x.split('_')[1]) for x in oc[k]])
            ll = np.array([int(x.split('_')[0]) for x in oc[k]])
            xs = xs[ll == int(ly)]
            xs = np.sort(np.unique(xs))
            if xs.shape[0]>0:
                node_start_layers += 'layer %d (%d neurons):<br/>' %(ly, int(xs.shape[0]))
            for x in list(xs):
                node_start_layers += '<a href="../layer%d/%d.html">%d</a> &nbsp' %(int(ly),int(x),int(x))
            if xs.shape[0]>0:
                node_start_layers += '<br/>'
        node_start_layers += 'this neuron in layer %d:<br/>' %(int(layer))
        node_start_layers += '<a href="../layer%d/%d.html">%d</a> &nbsp' %(int(layer),int(kernel),int(kernel))
        html_link_nodes.append(html_link_table_temple % (node_motif,node_start_layers))

bedlink = 'http://bioinfo-xwwang-thu.cn/zwei/NeuronMotif/DD-10/layer%d/%d.bed' %(int(layer),int(kernel))
ucsc_link = ''                
if bedlink is not None:
    ucsc_link = '<a href="http://genome.ucsc.edu/cgi-bin/hgTracks?db=hg19&hgt.customText=%s">Click here to see the sequences following this syntax in UCSC Genome Browser.</a>' %(bedlink)
 

html_whole_txt = html_txt % (int(kernel),int(kernel),int(layer),int(kernel),ucsc_link,html_table,'<br/>'.join(html_link_nodes),jsscripts)
with open( 'layer'+str(layer)+'/tree.'+str(kernel)+'.html'  ,'w') as ftest:
                ftest.write(html_whole_txt)

exit(0)

with h5py.File('layer'+str(layer)+'/kernel-'+str(kernel)+'.h5','r') as f:
    sample_seq = f['seq1'][:]
    value = f['value'][:]


#max_act = np.load('layer9_max_act.npy')


max_act_list = []
for ly_n in range(int(layer)-1):
    print(ly_n)
    max_act = np.zeros(kernel_nb[ly_n])
    for i in range(kernel_nb[ly_n]):
        ppmfile = 'layer'+str(layer-1)+'/kernel-'+str(i)+'.ppm.h5'
        if not os.path.exists(ppmfile):
            continue
        with h5py.File('layer'+str(layer-1)+'/kernel-'+str(i)+'.ppm.h5','r') as f:
            for a in list(f.keys()):
                if a.startswith('act'):
                    act = f[a][:].max()
                    if act > max_act[i]:
                        max_act[i] = act
    max_act_list.append(max_act)



def gen_sample(pwms, sp_size):
    pwm_nb = pwms.shape[0]
    seq_size = pwms.shape[1]
    sample = np.zeros((pwm_nb * sp_size,seq_size, 4))
    for i in range(pwm_nb):
        pwm = pwms[i,:,:]
        pwmids = np.concatenate([np.random.choice(4,sp_size, p = pwm[j,:])[:,np.newaxis] for j in range(pwm.shape[0])], axis=1)
        for j in range(4):
            sel = np.where(pwmids == j)
            sample[(i*sp_size):((i+1)*sp_size),:,j][sel] = 1
    return sample



f = h5py.File('layer' + str(layer)+ '/kernel-'+str(kernel)+'.ppm.h5','r')

long_motif_ppm = {}
long_motif_act = {}
long_motif_idx = {}

for k in list(f.keys()):
    if k.startswith('ppm'):
        long_motif_ppm[k[len('ppm'):]] = f[k][:]
        long_motif_ppm[k[len('ppm'):]] = long_motif_ppm[k[len('ppm'):]][:,[0,2,1,3]]
        long_motif_act[k[len('ppm'):]] = f['conact' + k[len('ppm'):]][:].max()
        long_motif_idx[k[len('ppm'):]] = f['index' + k[len('ppm'):]][:]

sample_sz = 1000

occur = {}
for mcb_name in mcbs.keys():
    mcb = mcbs[mcb_name]
    ppmids = list(seqs[mcb_name])
    pwms = [long_motif_ppm[ppmid][np.newaxis,:] for ppmid in ppmids ]
    pwms = np.concatenate(pwms,axis=0)
    pwm_match = mcb_match[mcb_name]
    pwm_pos = [pos[ppmid] for ppmid in ppmids]
    sel_node  = nodes[mcb_name]
    for node in sel_node.keys():
        node_idx = node.split('_')
        sample_seq_masked = sample_seq.copy()
        valid_idx = []
        for i in range(len(pwm_match)):
            count = 0
            for j in range(len(pwm_match[i])):
                if (str(j) not in node_idx) and  pwm_match[i][j] != 'X':
                    sample_seq_masked[long_motif_idx[ppmids[i]],pwm_pos[i][count]:(pwm_pos[i][count]+motif_ppms[pwm_match[i][j]].shape[0]),:] = 0.25
                    print('%dmask:%d~%d' % (j,pwm_pos[i][count],pwm_pos[i][count]+motif_ppms[pwm_match[i][j]].shape[0]))
                    count += 1
            valid_idx.append(long_motif_idx[ppmids[i]])
        valid_idx = np.concatenate(valid_idx)
        sample_seq_masked = sample_seq_masked[valid_idx,:,:]
        for tl in range(len(model_list)-1):
            max_act = max_act_list[tl]
            presubmodel = model_list[tl]
            feature_map = presubmodel.predict(sample_seq_masked, batch_size = 1000)
            feature_map = feature_map.reshape((feature_map.shape[0]*feature_map.shape[1],feature_map.shape[2]))
    #        mn =  [feature_map[feature_map[:,k]>0,k].mean() for k in range(feature_map.shape[1])]
            mx =  [feature_map[feature_map[:,k]>0,k].max() for k in range(feature_map.shape[1])]
    #        mn = np.array(mn)
            mx = np.array(mx)
            val = mx/max_act
            idx = val.argsort()[::-1]
            selmotif = [mcb[int(ni)] for ni in node_idx]
            idx1 = []
            for i in range(idx.shape[0]):
                print(i)
                if not  os.path.exists('layer' + str(tl+1) +'/kernel-'+str(idx[i])+'-segs.meme'):
                    continue
                os.system('tomtom --norc  layer' + str(tl+1) +'/kernel-'+str(idx[i])+'-segs.meme layer'+str(layer)+'/kernel-'+str(kernel)+'-unified-dict.meme -oc layer'+str(layer)+'/kernel-'+ str(kernel) +'-segs-segs')
                tomtom = pd.read_csv('layer'+str(layer)+'/kernel-'+ str(kernel) +'-segs-segs/tomtom.tsv',sep='\t',skipfooter=3,engine='python')
                tomtom = tomtom.loc[tomtom.loc[:,'Orientation']!='-',:]
                selrow = tomtom['Target_ID'] == selmotif[0]
                for ni in range(1,len(selmotif)):
                    selrow = np.logical_or(selrow, tomtom['Target_ID'] == selmotif[ni])
                if tomtom.loc[selrow,'q-value'].min() < 10e-5:
                    idx1.append(idx[i])
                if len(idx1) > 4:
                    break
            os.system('rm -rf ' + 'layer'+str(layer)+'/kernel-'+str(kernel)+'-segs-segs')
    #        idx2 = (mn/max_act).argsort()[::-1][0:5]
            if len(idx1) > 0:
                occur[mcb][node] = str(tl + 1) + '_' +str(i)
                
        
    
    




    
#html_whole_txt = html_txt % ('<table class="mt">%s</table>'%(html_row),jsscripts)

#html_whole_txt += '<br/><iframe src="test_tree.254.html" frameBorder="0" width="900" scrolling="no" height="900"></iframe>'



with open( 'layer'+str(layer)+'/tree.'+str(kernel)+'.html'  ,'w') as ftest:
                ftest.write(html_whole_txt)




