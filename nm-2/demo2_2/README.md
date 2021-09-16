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

Only the receptive field of a convolutional neuron in layer 5 (82 bp) is long enough to put two 16-bp shifting CTCF motifs (19 bp). The decoupling algorithm in NeuronMotif is applied twice to the convolutional neuron in layer 5 of the DCNN model. The sample sequences depend on the result of Demo 2. 

Only the results of layer 5 will be shown. This process can be applied to other layers but is not necessary. 
