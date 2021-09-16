# Demo 2
* Input: 82 bp DNA sequence
* Output: a scalar belonging to interval [0,1]
* Positive sequences: random sequences with 2 CTCF motifs. One CTCF motif starts at the 1st-16th bp. The other CTCF motif starts at 47th-62th bp
* Negative sequences: random sequences with 2 CTCF motifs. One CTCF motif starts at the 1st-16th bp or 47th-62th bp. The other CTCF motif starts at 17th-46th bp

Architecture of DCNN model:


* Convolutional layer 1 (5 filters, size 7)
    + BatchNormalization
    + Activation function (ReLU)
    + Maxpooling operation (size 2)
* Convolutional layer 2 (5 filters, size 3)
    + BatchNormalization
    + Activation function (ReLU)
    + Maxpooling operation (size 2)
* Convolutional layer 3 (6 filters, size 3)
    + BatchNormalization
    + Activation function (ReLU)
    + Maxpooling operation (size 2)
* Convolutional layer 4 (6 filters, size 3)
    + BatchNormalization
    + Activation function (ReLU)
    + Maxpooling operation (size 2)
* Convolutional layer 5 (1 filters, size 3)
    + BatchNormalization
    + Activation function (sigmoid)
* Flatten

# Result

For the result in each folder, the sequence samples, position probability matrixes and diagnosis indicators are stored in folder 'layerX'. 
The corresponding visualization results (HTML files) and the patterns matched to the JASPAR database by using Tomtom are shown in 'layerX_visualization'

The results of layers 1-5 will be shown. The decoupling algorithm in NeuronMotif is applied once to the DCNN model.
