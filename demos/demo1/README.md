# Demo 1

* Input: 8 bp DNA sequence
* Output: a scalar belonging to interval [0,1]
* Positive sequences: 2 ZEB1 motifs shifted by 1 bp
* Negative sequences: random sequences

Architecture of DCNN model:

* Convolutional layer 1 (3 filters, size 5)
    + Activation function (ReLU)
    + Maxpooling operation (size 2)
* Convolutional layer 2 (1 filters, size 2)
    + Activation function (sigmoid)
* Flatten



# Run this demo

Download motif database

Take JASPAR as an example:

```
wget --no-check-certificate  -O  motifDB.txt  https://jaspar.genereg.net/download/data/2022/CORE/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt
```

Then, run

```
bash script.sh
```



# Result

The HTML visualization results are organized in each layer folder of  HTML folder. You can visualize them by web browser:

| Files                                            | Contains                                           |
|--------------------------------------------------|----------------------------------------------------|
| HTML/layer[Layer#]/[Neuron#].html                | Visualization of CN CRMs                           |
| HTML/layer[Layer#]/tree.[Neuron#].html           | Visualization of syntax tree for CN CRMs           |
| HTML/layer[Layer#]/tomtom_[Neuron#].sel.ppm.meme | some motif segment mapped to database              |
| layer[Layer#]/tomtom_dict_[Neuron#]              | motif dictionary mapped to database                |



For the result in each folder, the sequence samples, position probability matrixes and diagnosis indicators are stored in folder 'layerX'.


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
