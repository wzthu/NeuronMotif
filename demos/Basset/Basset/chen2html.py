import h5py
import numpy as np
import os
import sys
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
f = h5py.File(sys.argv[1],'r')

ppm = f['ppm0'][:]

html_txt0 = '''
<html lang="en">
    <body>
        <h2>Nucleotides</h2>
        <canvas id="logo_nt"></canvas>

        <h2>Amino acids</h2>
        <canvas id="logo_aa"></canvas>

        <script src="https://wzthu.github.io/NeuronMotif/jseqlogo.js"></script>
        <script>
            var options = {
                "colors": jseqlogo.colors.nucleotides
            };

            var data = {
                "A": [0.5, 1.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                "C": [0.8, 0.05, 0.2, 0.0, 0.5, 0.05, 0.0],
                "G": [0.01, 0.0, 0.0, 0.7, 0.0, 0.3, 0.84],
                "T": [0.2, 0.0, 0.45, 0.0, 0.3, 0.0, 0.2]
            };

            sequence_logo(document.getElementById("logo_nt"), 600, 200, data, options);


        </script>
    </body>
</html>
'''

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
Visit NeuronMotif website for more information: <a href="https://wzthu.github.io/NeuronMotif/">https://wzthu.github.io/NeuronMotif/ </a>
<br/>
Please be patient to load all motif logos or patterns in the column of CN CRMs ...
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
width=ppm.shape[0]*8
height = 50

i_max = 0
for k in list(f.keys()):
    if k.startswith('act'):
        if int(k[3:])>i_max:
            i_max= int(k[3:])


for i in range(i_max+1):
        if 'act%d' % (i) not in list(f.keys()):
            continue
        ppm_id = '%8d_%.3f_%.3f_%d' % (i,f['act%d' % (i)][:].max(), f['conact%d' % (i)][0],f['act%d' % (i)].shape[0])
        ppm = f['ppm%d' % (i)][:]
        ppm_js = ppm2js(ppm, ppm_id, width, height)
        ppm_jss.append(ppm_js)
        ppm_ids.append('<tr><td>%s</td><td><a href="tomtom_%s.sel.ppm.meme/tomtom.html">TomtomLink</a></td><td><canvas id="%s"></canvas></td></tr>' % (ppm_id,sys.argv[2], ppm_id))
    
    
html_txt1 = html_txt % ('Neuron '+ ' '.join(['<a href="'+str(i)+'.html">'+str(i)+'</a>' for i in range(int(sys.argv[4]))]) + ('<br/><a href="tree.%s.html">Click here to see syntax tree (if exist)</a>' % (sys.argv[3].split('/')[1])),'<table class="mt"><tr><td>Dcp1_Dcp2_ActMax_ConsensusAct_SampleSize</td><td align=center>TomtomResult</td><td align=center>CN CRMs ('+str(ppm.shape[0])+' bp)</td></tr>'+'\n'.join(ppm_ids)+'</table>', '\n'.join(ppm_jss))

with open(sys.argv[3]+'.html','w') as ftest:
    ftest.write(html_txt1)
 
  


