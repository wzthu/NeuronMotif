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
We show the examples for DeepSEA, DD-10, Basset, BD-5, BD-10. You can learn the examples of these models.



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

The script can be submited at the same directory at each node of the cluster.

```
sh run.layer.sh 1 320 20  
sh run.layer.sh 2 480 5
sh run.layer.sh 3 960 5
```

# Convert to motif file

The arameter is the layer number:

```
python tochen.py 1
python tochen.py 2
python tochen.py 3
```

For model generated from  DeepSEA data please run the following scripts instead.

```
python tochenDeepSEA.py 1
python tochenDeepSEA.py 2
python tochenDeepSEA.py 3
```


The motif file is xxx.chen in corresponding layer folder

# Visualization

These steps are optional for NeuronMotif.

## Install visualization dependences

Download and install meme-suit to get tomtom first:

version: 5.1.0, later version does not support >100 bp receptive field.

https://meme-suite.org/meme/meme-software/5.1.0/meme-5.1.0.tar.gz
https://meme-suite.org/meme/tools/tomtom

and add it to PATH environment variable or execute

```
export PATH=/path/to/meme/bin:/path/to/meme:/path/to/meme/libexec/meme-x.x.0:$PATH
```

Test following commands:

```
tomtom
```
and 

```
chen2meme
```


## Download motif database

Take JASPAR as an example:

wget http://jaspar.genereg.net/download/CORE/JASPAR2020_CORE_vertebrates_redundant_pfms_meme.txt
mv JASPAR2020_CORE_vertebrates_redundant_pfms_meme.txt  motifDB.txt

## Visualization through run visualization script 'vis.layer.sh'.

The parameter of run.layer.sh:
* 1st, layer number, start from 1
* 2nd, the max threads number to be used.
* It depends on total CPU cores in this computer and is limited by memory size.
Be carefule to set large threads. The memory is easy to be stuffed.

For example:

Our server contains  28 cores server 128GB memory

sh vis.layer.sh 1  28
sh vis.layer.sh 2  28
sh vis.layer.sh 3  28

The results of  HTML files in vis folder of corresponding layer folder.

