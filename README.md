# Introduction of NeuronMotif

NeuronMotif is an algorithm for interpreting the patterns learned by deep convolutional neurons (DCN) in DCNN. For a DCNN based on DNA sequences, it can convert the substructure weight of a DCN into DNA sequence motifs. DNA can be considered a new language so the task of NeuronMotif is to uncover the motif and motif grammar embedded in DCNs.

![](https://github.com/wzthu/NeuronMotif/blob/master/Goal.jpg)


# NeuronMotif Software

This repository implements the NeuronMotif algorithm in ["NeuronMotif: Deciphering cis-regulatory codes by layerwise de-mixing of deep neural networks"](https://www.biorxiv.org/content/10.1101/2021.02.10.430606v1) by Zheng Wei et al.

This repository includes the code for generating the result in the paper. We also provide the instructions for users to apply NeuronMotif to their own models. 

Feel free to contact Zheng Wei for any issues or difficulties:
+ Email: weiz(at)tsinghua.edu.cn
+ GitHub: https://github.com/wzthu/NeuronMotif/issues

* Input of NeuronMotif:
    + Architecture of CNN Model
    + H5 weight file of CNN model

* Ouput of NeuronMotif:
    + HTML of CN cis-regulatory module (CRM) 
    + HTML of syntax rule for each CN
    + Position probability matrix of discovered motifs and CN CRMs
    + The discovered motifs matched to JASPAR database


The contains in the directorise:

| Directory             | Contains                                               |
|-----------------------|--------------------------------------------------------|
| NeuronMotif           |  NeuronMotif software                                    |
| demos/demo1           | Simple NeuronMotif demo applied to 2-layer toy model   |
| demos/demo2           | Simple NeuronMotif demo applied to 4-layer toy model   |
| demos/DeepSEA/DeepSEA | NeuronMotif applied to DeepSEA model                   |
| demos/DeepSEA/DD-10   | NeuronMotif applied to DD-10 model                     |
| demos/Basset/Basset   | NeuronMotif applied to Basset model                    |
| demos/Basset/BD-10    | NeuronMotif applied to BD-10 model                     |
| dcnn                  | The code for training the CNN models used in this work |


Result of DeepSEA/DD-10/Basset/BD-10 is available at [NeuronMotif website](https://wzthu.github.io/NeuronMotif/).


Before using this repository, users should install the dependent software following the instruction at the end of this file.


# Installation

We test this repository in a CentOS 7 computer cluster with 4 nodes, each of which contains 28 cores and 128 GB memory. About 10 TB is consumed for obtaining the result of all examples in the manuscripts. The installation lasts for ~10 min if there are no more dependent packages.

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
```

***Note: if there are some warnings in the configuring, HTML related perl dependent should be solved:***

File::Which missing.

HTML::Template missing.

HTML::TreeBuilder missing.

JSON missing.

These warnings can be solved by:
```
export PERL_MM_USE_DEFAULT=1
perl -MCPAN -e 'install "File::Which"'
perl -MCPAN -e 'install "HTML::Template"'
perl -MCPAN -e 'install "HTML::TreeBuilder"'
perl -MCPAN -e 'install "JSON"'
perl -MCPAN -e 'install "XML::Simple"'
perl -MCPAN -e 'install "XML::Parser::Expat"'
```

Compile the code:

```
make
make test
make install
```
***Note: make sure tomtom in the meme suit can work in the make test.***

The erros of other tools can be ignored.

Add it to PATH environment variable by appending following command line to  ~/.bashrc:

```
export PATH=$HOME/meme/bin:$HOME/meme/libexec/meme-5.1.0:$PATH
```

and execute:

```
source  ~/.bashrc
```

Test following commands:

```
tomtom
```
and

```
chen2meme  -h
```

## Clone this repository:

```
git clone https://github.com/wzthu/NeuronMotif.git
```


## Ready to use

This command line should be executed every time you enter the terminal before using NeuronMotif

```
conda activate NeuronMotif
```

To use NeuronMotif, please go to directory [NeuronMotif](https://github.com/wzthu/NeuronMotif/tree/master/nm) and read the instructions in README.md.

To run the demos or apply NeuronMotif to DeepSEA/DD-10/Basset/BD-10, please go to correspoinding directory and read the README.md in the directory.
