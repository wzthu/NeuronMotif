# Demo 1

* Input: 8 bp DNA sequence
* Output: a scalar belonging to interval [0,1]
* Positive sequences: 2 ZEB1 motifs shifted by 1 bp
* Negative sequences: random sequences

Architecture of DCNN model:

* Convolutional layer 1 (3 filters, size 5)
    + Activation function (ReLU)
    + Maxpooling operation (size 2)
* Convolutional layer 2 (1 filters, size 2)
    + Activation function (sigmoid)
* Flatten



# Run this demo

Download motif database

Take JASPAR as an example:

```
wget --no-check-certificate  -O  motifDB.txt  https://jaspar.genereg.net/download/data/2022/CORE/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt
```

Then, run

```
bash script.sh
```



# Result

For the result in each folder, the sequence samples, position probability matrixes and diagnosis indicators are stored in folder 'layerX'. 
