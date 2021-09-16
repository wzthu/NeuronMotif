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


# Result

For the result in each folder, the sequence samples, position probability matrixes and diagnosis indicators are stored in folder 'layerX'. 
The corresponding visualization results (HTML files) and the patterns matched to the JASPAR database by using Tomtom are shown in 'layerX_visualization'

The results of layers 1-2 will be shown. The decoupling algorithm in NeuronMotif is applied once to the DCNN model.
