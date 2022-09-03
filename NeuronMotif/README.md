
Howw to use:

# Enter NeuronMotif environment

conda  activate NeuronMotif

# Copy code folder for your DCNN model

Copy this folder to a new directory ND (e.g.  ~/mymodel) and 'cd ND'. For exmaple:

```
cp -r . ~/mymodel
cd ~/mymodel
```

#Copy your DCNN model weight file to ND/weight.hdf5. For example:

```
cp xxx/xxx/weight.deep.h5 ~/mymodel/weight.hdf5
```

#Download motif database

Take JASPAR as an example:

```
wget --no-check-certificate  -O  motifDB.txt  https://jaspar.genereg.net/download/data/2022/CORE/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt
```

# Configure DCNN model structure of your DCNN model 

Modify ND/modeldef.py in ND according to the your CNN model strucutre.
You need to follow the comments in modeldef.py to modified the modeldef.py.
We show the examples for DeepSEA, DD-10, Basset, BD-10. It is easy to learn the way to fill your own structure from  the examples of these models.



# Run NeuronMotif 

Run NeuronMotif algorithm layer by layer with script 'run.layer.sh'. 

The parameter of run.layer.sh:

* 1st, layer number, start from 1

* 2nd, kernels number in this layer

* 3rd, the max threads number to be used. 

* 4rd, (optional) the number of flexible motifs recgnized by the convolutional neuron. Default: NeuronMotif detects automatically.

It depends on total CPU cores in this computer and is limited by memory size.
Be carefule to set large threads. The memory is easy to be stuffed.

For example:

Our server contains 28 cores server 128GB memory.

***Note***: only the last layer is done, the next layer can be started.


```
# For one line of the script, the line can be submitted to several nodes at the same time for paralleling.
# Next line can not be submitted until the tasks in all nodes are done.

bash run.layer.sh 1 128 20
bash run.layer.sh 2 128 20
bash run.layer.sh 3 160 20
bash run.layer.sh 4 160 20
bash run.layer.sh 5 256 20
bash run.layer.sh 6 256 20
bash run.layer.sh 7 384 20
bash run.layer.sh 8 384 20
bash run.layer.sh 9 512 10
Bash run.layer.sh 10 512 10

```


The HTML, PPM and tomtom results are stored in the folder of corresponding layer folder.

| Files                                            | Contains                                           |
|--------------------------------------------------|----------------------------------------------------|
| layer[Layer#]/[Neuron#].html                     | Visualization of CN CRMs                           |
| layer[Layer#]/tree.[Neuron#].html                | Visualization of syntax tree for CN CRMs           |
| layer[Layer#]/kernel-[Neuron#].all.ppm.chen      | PPMs of CN CRMs in chen format                     |
| layer[Layer#]/kernel-[Neuron#].all.ppm.meme      | PPMs of CN CRMs in meme format                     |
| layer[Layer#]/kernel-[Neuron#].h5                | sequence sample                                    |
| layer[Layer#]/kernel-[Neuron#].ppm.h5            | PPMs/activation/indicators of CN CRMs in h5 format |
| layer[Layer#]/kernel-[Neuron#]-segs.chen         | PPMs of CRMs segments                              |
| layer[Layer#]/kernel-[Neuron#]-segs-dict         | motif segment mapped to dictionary                 |
| layer[Layer#]/tomtom_dict_[Neuron#]              | motif dictionary mapped to tomtom                  |
| layer[Layer#]/kernel-[Neuron#]-unified-dict.chen | PPMs of motifs in dictionary (chen format)         |
| layer[Layer#]/kernel-[Neuron#]-unified-dict.h5   | PPMs of motifs in dictionary (h5 format)           |
| layer[Layer#]/kernel-[Neuron#]-unified-dict.meme | PPMs of motifs in dictionary (meme format)         |
