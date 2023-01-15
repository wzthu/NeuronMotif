


# bash run.layer.sh <layer> <the number of kernels> <threads> <optional: the number of flexible motifs> &> .layer<layer>.log


bash run.layer.sh 1 5 28   &> .layer1.log
bash run.layer.sh 2 5 28   &> .layer2.log
bash run.layer.sh 3 6 28   &> .layer3.log
bash run.layer.sh 4 6 28   &> .layer4.log
bash run.layer.sh 5 1 28 2 &> .layer5.log

