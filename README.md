# NeuronMotif

This repository implements the NeuronMotif algorithm in ["NeuronMotif: Deciphering cis-regulatory codes by layerwise de-mixing of deep neural networks"](https://www.biorxiv.org/content/10.1101/2021.02.10.430606v1) by Zheng Wei et al.

This repository includes the code for generating the published result.

Feel free to contact Zheng Wei for any issues or difficulties:
+ Email: weiz(at)tsinghua.edu.cn
+ GitHub: https://github.com/wzthu/NeuronMotif/issues

The code for training the deep convolutional neural network (DCNN) models that are used in this work are stored in [dcnn](https://github.com/wzthu/NeuronMotif/tree/master/dcnn).

The code implements the NeuronMotif algorithm for decoupling the mentioned DCNN **once** are stored in [nm](https://github.com/wzthu/NeuronMotif/tree/master/nm). It also includes interpreting and visualizing codes.

The code implements the NeuronMotif algorithm for decoupling the mentioned DCNN **twice** are stored in [nm-2](https://github.com/wzthu/NeuronMotif/tree/master/nm-2). It depends on the sequence sampling result  generated by the code in [nm](https://github.com/wzthu/NeuronMotif/tree/master/nm). It also includes interpreting and visualizing codes. 

The code of NeuronMotif does not depend on GPU. We have parallelized the code so that it can make full use of the computing resources in a computer cluster.

If you are interested in a part of the work, you can read README file in the corresponding folder for the instruction of downloading the data and running the code.

# Introduction of NeuronMotif

NeuronMotif is an algorithm for interpreting the patterns learned by deep convolutional neurons (DCN) in DCNN. For a DCNN based on DNA sequences, it can convert the substructure weight of a DCN into DNA sequence motifs. DNA can be considered a new language so the task of NeuronMotif is to uncover the motif and motif grammar embedded in DCNs.

![](https://github.com/wzthu/NeuronMotif/blob/master/Goal.jpg)

# Results of NeuronMotif

Result is available at [NeuronMotif website](https://wzthu.github.io/NeuronMotif/).


# Installation

## Install anaconda

Download and install (anaconda)[https://www.anaconda.com/products/individual].

## Create the  NeuronMotif environment

NeuronMotif is implemented and tested in python 3.6.

```
conda create -n NeuronMotif python=3.6
```

## Install dependent packages

```
conda activate NeuronMotif
pip install h5py==2.10.0
pip install matplotlib
pip install pandas
pip install numpy
pip install sklearn
pip install tensorflow==1.15.0
pip install keras==2.3.1

```


## Install visualization dependences

Download and install meme-suit to get tomtom first:

version: 5.1.0, later version does not support >100 bp receptive field.

Download:

```
wget https://meme-suite.org/meme/meme-software/5.1.0/meme-5.1.0.tar.gz

```

Install:
```
tar zxf meme-5.1.0.tar.gz
cd meme-5.1.0
./configure --prefix=$HOME/meme --enable-build-libxml2 --enable-build-libxslt
make
make test
make install
```

and add it to PATH environment variable or execute

```
export PATH=$HOME/meme/bin:$HOME/meme/libexec/meme-5.1.0:$PATH
```

Test following commands:

```
tomtom -h
```
and

```
chen2meme  -h
```

## Clone this repository:

```
git clone https://github.com/wzthu/NeuronMotif
```


## Ready to use

This command line should be executed every time you enter the terminal before using NeuronMotif

```
conda activate NeuronMotif
```

