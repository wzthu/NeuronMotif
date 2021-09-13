
Please read the README in the folder [code](https://github.com/wzthu/NeuronMotif/tree/master/nm/code) first.

# Enter NeuronMotif environment

```
conda  activate NeuronMotif
```

# Download the weight file

Download and rename weight file as 'weight.hdf5':

```
wget  --no-check-certificate -O weight.hdf5  http://bioinfo-xwwang-thu.cn/zwei/NeuronMotif/DCNN_weight/Basset/BD-5/weight.hdf5
```

Alternatively, if the link above is not available temperately, you can download from https://cloud.tsinghua.edu.cn/d/fee522536d524eae9531/files/?p=%2FDCNN_weight%2FBasset%2FBD-5%2Fweight.hdf5&dl=1


# Run NeuronMotif:

```
bash run.layer.sh 1 64 20
bash run.layer.sh 2 128 20
bash run.layer.sh 3 256 20
bash run.layer.sh 4 384 20
bash run.layer.sh 5 512 20
```

```
python tochen.py 1
python tochen.py 2
python tochen.py 3
python tochen.py 4
python tochen.py 5
```

# Visulization

Download JASPAR database:

```
wget -O  motifDB.txt  http://jaspar.genereg.net/download/CORE/JASPAR2020_CORE_vertebrates_redundant_pfms_meme.txt
```

Match the discovered motif to JASPAR database:

```
bash vis.layer.sh 1 28
bash vis.layer.sh 2 28
bash vis.layer.sh 3 28
bash vis.layer.sh 4 28
bash vis.layer.sh 5 28
```


