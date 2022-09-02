
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

