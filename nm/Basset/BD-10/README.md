
Please read the README in the folder [code](https://github.com/wzthu/NeuronMotif/tree/master/nm/code) first.

# Enter NeuronMotif environment

```
conda  activate NeuronMotif
```

# Download the weight file

```
wget -O weight.hdf5 https://cloud.tsinghua.edu.cn/d/fee522536d524eae9531/files/?p=%2FDCNN_weight%2FBasset%2FBD-10%2Fweight.hdf5&dl=1
```

# Run NeuronMotif:

```
sh run.layer.sh 1 128 20
sh run.layer.sh 2 128 20
sh run.layer.sh 3 160 20
sh run.layer.sh 4 160 20
sh run.layer.sh 5 256 20
sh run.layer.sh 6 256 20
sh run.layer.sh 7 384 20
sh run.layer.sh 8 384 20
sh run.layer.sh 9 512 20
sh run.layer.sh 10 512 20
```

```
python tochen.py 1
python tochen.py 2
python tochen.py 3
python tochen.py 4
python tochen.py 5
python tochen.py 6
python tochen.py 7
python tochen.py 8
python tochen.py 9
python tochen.py 10
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
sh vis.layer.sh 4 28
sh vis.layer.sh 5 28
sh vis.layer.sh 6 28
sh vis.layer.sh 7 28
sh vis.layer.sh 8 28
sh vis.layer.sh 9 28
sh vis.layer.sh 10 25
```

