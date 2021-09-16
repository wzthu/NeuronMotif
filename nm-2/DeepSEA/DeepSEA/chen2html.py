import h5py
import numpy as np
import os
import sys
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
f = h5py.File(sys.argv[1],'r')



html_txt = '''
<html lang="en">
<style>

tr td,th{

border:1px solid black;

}

.mt{

 border-collapse:collapse;

}

</style>
    <body>
     %s
<br/>
NeuronMotif website: <a href="https://wzthu.github.io/NeuronMotif/">https://wzthu.github.io/NeuronMotif/ </a>
<br/>
Please be patient to load all motif logos or patterns in the column of CN motifs ...
<br/>
        %s
        <script src="https://wzthu.github.io/NeuronMotif/jseqlogo.js"></script>
        <script>
            var options = {
                "colors": jseqlogo.colors.nucleotides
            };

            %s
        </script>
    </body>
</html>
'''


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

ppm_ids = []
ppm_jss = []
for i in range(12):
    for j in range(12):
        if 'act%d_%d' % (i,j) in list(f.keys()):
            ppm = f['ppm%d_%d' % (i,j)][:]
            break


width=ppm.shape[0]*8
height = 50

for i in range(12):
    for j in range(12):
        if 'act%d_%d' % (i,j) not in list(f.keys()):
            continue
        ppm_id = '%04d_%04d_%.4f_%.4f_%d' % (i,j,f['act%d_%d' % (i,j)][:].max(), f['conact%d_%d' % (i,j)][0],f['act%d_%d' % (i,j)].shape[0])
        ppm = f['ppm%d_%d' % (i,j)][:]
        ppm_js = ppm2js(ppm, ppm_id, width, height)
        ppm_jss.append(ppm_js)
        ppm_ids.append('<tr><td>%s</td><td><a href="tomtom_%s.sel.ppm.meme/tomtom.html">TomtomLink</a></td><td><canvas id="%s"></canvas></td></tr>' % (ppm_id,sys.argv[2], ppm_id)) 
    
html_txt1 = html_txt % ('Neuron '+ ' '.join(['<a href="'+str(i)+'.html">'+str(i)+'</a>' for i in range(int(sys.argv[4]))]),'<table class="mt"><tr><td>Dcp1_Dcp2_ActMax_ConsensusAct_SampleSize</td><td align=center>TomtomResult</td><td align=center>CN motifs ('+str(ppm.shape[0])+' bp)</td></tr>'+'\n'.join(ppm_ids)+'</table>', '\n'.join(ppm_jss))

with open(sys.argv[3]+'.html','w') as ftest:
    ftest.write(html_txt1)
 
  


