For each layer do:
1. link the ppm files of layers

```
bash runlink.sh path/to/layer10 layer10 1280
```

The parameter of runlink.sh:

1st, NeuronMotif result ppm directory xxx/xxx/xxx/layer?
2nd, layer?
3st, the number of the kernel

2. Download motif database

Take JASPAR as an example:

```
wget -O motifDB.txt  http://jaspar.genereg.net/download/CORE/JASPAR2020_CORE_vertebrates_redundant_pfms_meme.txt
```

3. run tomtom

The parameter of run.layer.tt.sh:
1st, layer number, start from 1
2nd, kernels number in this layer
3rd, the max threads number to be used.
4th, is the model train by DeepSEA dataset, yes: 1, no: 0
It depends on total CPU cores in this computer and is limited by memory size.
Be carefule to set large threads. The memory is easy to be stuffed.

For example:

# 28 cores server 128GB memory

***Note: only the last layer is done, the next layer can be started.***
```
bash run.layer.sh 1 320 28 1
bash run.layer.sh 2 480 28 1
bash run.layer.sh 3 960 28 1
```
