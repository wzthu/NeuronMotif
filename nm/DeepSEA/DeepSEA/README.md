
Please read the README in the folder [code](https://github.com/wzthu/NeuronMotif/tree/master/nm/code) first.

# Enter NeuronMotif environment

```
conda  activate NeuronMotif
```

# Download the weight file

```
wget -O weight.hdf5 https://cloud.tsinghua.edu.cn/d/fee522536d524eae9531/files/?p=%2FDCNN_weight%2FDeepSEA%2FDeepSEA%2Fweight.hdf5&dl=1
```

# Run NeuronMotif:

```
sh run.layer.sh 1 320 20
sh run.layer.sh 2 480 20
sh run.layer.sh 3 960 20
```

```
python tochenDeepSEA.py 1
python tochenDeepSEA.py 2
python tochenDeepSEA.py 3
```

# Visulization

Download JASPAR database:

```
wget -O  motifDB.txt  http://jaspar.genereg.net/download/CORE/JASPAR2020_CORE_vertebrates_redundant_pfms_meme.txt
```

Match the discovered motif to JASPAR database:

```
sh vis.layer.sh 1 28
sh vis.layer.sh 2 28
sh vis.layer.sh 3 28
```


