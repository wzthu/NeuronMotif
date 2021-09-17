How to use:

# Enter NeuronMotif environment

conda  activate NeuronMotif

# Copy code folder for your DCNN model

Copy this folder to a new directory ND and 'cd ND'. For exmaple:

```
cp -r . ~/mymodel
cd ~/mymodel
```

# Configure weight of your DCNN model

Copy your DCNN model weight file to ND/weight.hdf5. For example:

```
cp xxx/xxx/weight.deep.h5 ~/mymodel/weight.hdf5
```

# Configure DCNN model structure of your DCNN model 

Modify ND/modeldef.py in ND according to the your DCNN model strucutre.
You need to follow the comments in modeldef.py to modified the modeldef.py.
We show the examples for DeepSEA, DD-10, Basset, BD-5, BD-10. It is easy to learn the way to fill your own structure from  the examples of these models.



# Run NeuronMotif 

Run NeuronMotif algorithm layer by layer with script 'run.layer.sh'. 

The parameter of run.layer.sh:

* 1st, layer number, start from 1

* 2nd, kernels number in this layer

* 3rd, the max threads number to be used. 

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
bash run.layer.sh 9 512 20
bash run.layer.sh 10 512 20
```

# Convert to motif file

The parameter is the layer number:

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

For model generated from  DeepSEA data please run the following scripts instead.

```
# These scripts do not spend a long time, paralleling in not necessary.

python tochenDeepSEA.py 1
python tochenDeepSEA.py 2
python tochenDeepSEA.py 3
```


The motif file is xxx.chen in corresponding layer folder

# Visualization

These steps are optional for NeuronMotif.

## Download motif database

Take JASPAR as an example:

```
wget -O  motifDB.txt  http://jaspar.genereg.net/download/CORE/JASPAR2020_CORE_vertebrates_redundant_pfms_meme.txt
```


## Visualization through run visualization script 'vis.layer.sh'.

The parameter of run.layer.sh:
* 1st, layer number, start from 1
* 2nd, the max threads number to be used.
* It depends on total CPU cores in this computer and is limited by memory size.
Be carefule to set large threads. The memory is easy to be stuffed.

For example:

Our server contains  28 cores server 128GB memory

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
bash vis.layer.sh 10 28
```

The results of  HTML files in vis folder of corresponding layer folder.

