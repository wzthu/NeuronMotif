
Please read the README in the folder [code](https://github.com/wzthu/NeuronMotif/tree/master/nm/code) first.

# Enter NeuronMotif environment

```
conda  activate NeuronMotif
```

# Download the weight file

Download and rename weight file as 'weight.hdf5':

```
wget  --no-check-certificate -O weight.hdf5  http://bioinfo-xwwang-thu.cn/zwei/NeuronMotif/DCNN_weight/Basset/Basset/weight.hdf5
```

Alternatively, if the link above is not available temperately, you can download from https://cloud.tsinghua.edu.cn/d/fee522536d524eae9531/files/?p=%2FDCNN_weight%2FBasset%2FBasset%2Fweight.hdf5&dl=1 


# Download JASPAR database:

```
wget --no-check-certificate  -O  motifDB.txt  https://jaspar.genereg.net/download/data/2022/CORE/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt
```


# Run

Before running the scripts, you should adapt the number of threads in the scripts to your server or cluster.

## Through script in a server:


```
bash script.sh
```

## Through scripts in many nodes of a cluster

### Run NeuronMotif:

```
# For one line of the script, the line can be submitted to several nodes at the same time for paralleling.
# Next line can not be submitted until the tasks in all nodes are done.

bash run.layer.sh 1 300 20 
bash run.layer.sh 2 200 20
bash run.layer.sh 3 200 20
```

The HTML visualization results are organized in each layer folder of  HTML folder. You can visualize them by web browser:

| Files                                            | Contains                                           |
|--------------------------------------------------|----------------------------------------------------|
| HTML/layer[Layer#]/[Neuron#].html                | Visualization of CN CRMs                           |
| HTML/layer[Layer#]/tree.[Neuron#].html           | Visualization of syntax tree for CN CRMs           |
| HTML/layer[Layer#]/tomtom_[Neuron#].sel.ppm.meme | some motif segment mapped to database              |
| layer[Layer#]/tomtom_dict_[Neuron#]              | motif dictionary mapped to database                |




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
