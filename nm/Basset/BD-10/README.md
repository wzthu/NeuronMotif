
Please read the README in the folder [code](https://github.com/wzthu/NeuronMotif/tree/master/nm/code) first.

# Enter NeuronMotif environment

```
conda  activate NeuronMotif
```

# Download the weight file

Download and rename weight file as 'weight.hdf5':

```
wget  --no-check-certificate -O weight.hdf5  http://bioinfo-xwwang-thu.cn/zwei/NeuronMotif/DCNN_weight/Basset/BD-10/weight.hdf5
```

Alternatively, if the link above is not available temperately, you can download from  https://cloud.tsinghua.edu.cn/d/fee522536d524eae9531/files/?p=%2FDCNN_weight%2FBasset%2FBD-10%2Fweight.hdf5&dl=1

# Run

Before running the scripts, you should adapt the number of threads in the scripts to your server or cluster.

## Through script in a server:


```
bash script.sh
```

## Through scripts in many nodes of a cluster

## Run NeuronMotif:

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
bash run.layer.sh 9 512 20
bash run.layer.sh 10 512 20
```

```
# These scripts do not spend a long time, paralleling in not necessary.

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

## Visulization

Download JASPAR database:

```
wget -O  motifDB.txt  http://jaspar.genereg.net/download/CORE/JASPAR2020_CORE_vertebrates_redundant_pfms_meme.txt
```

Match the discovered motif to JASPAR database:

```
# These scripts can be submitted to different nodes at the same time. Order is not required.

bash vis.layer.sh 1 28
bash vis.layer.sh 2 28
bash vis.layer.sh 3 28
bash vis.layer.sh 4 28
bash vis.layer.sh 5 28
bash vis.layer.sh 6 28
bash vis.layer.sh 7 28
bash vis.layer.sh 8 28
bash vis.layer.sh 9 28
bash vis.layer.sh 10 25
```


