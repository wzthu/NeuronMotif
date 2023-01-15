


# bash run.layer.sh <layer> <the number of kernels> <threads> <optional: the number of flexible motifs> &> .layer<layer>.log

#<layer> the layer number
#<the number of kernels> the number of kernels in the layer
#<threads> the number of neuron can be demixed at the same time
#<optional: the number of hard-syntax motif or motif combination>
#If you know the maximun number of hard-syntax motifs recognized by the neuron in the layer, setting this parameter is benificial to improving motif quality.  
#Examples:
#1 hard syntax motif CTCF
#1 hard syntax motif CTCF-[5bp]-CTCF
#2 hard-syntax motifs  CTCF-[5bp]-CTCF-[3~5bp]-CTCF 
#3 hard-syntax motifs  CTCF-[3~5bp]-CTCF-[3~5bp]-CTCF



bash run.layer.sh 1 128 20  &> .layer1.log
bash run.layer.sh 2 128 20  &> .layer2.log
bash run.layer.sh 3 160 20  &> .layer3.log
bash run.layer.sh 4 160 20  &> .layer4.log
bash run.layer.sh 5 256 20  &> .layer5.log
bash run.layer.sh 6 256 20  &> .layer6.log
bash run.layer.sh 7 384 20  &> .layer7.log
bash run.layer.sh 8 384 20  &> .layer8.log
bash run.layer.sh 9 512 10  &> .layer9.log
bash run.layer.sh 10 512 10 &> .layer10.log

