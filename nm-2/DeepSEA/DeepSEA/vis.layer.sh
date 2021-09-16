layer=$1
kernels=$2
threads=$3
deepsea=$4
bash idx.sh  $kernels  | xargs -n 1 -P $threads  bash   vis.sh $layer $deepsea $kernels
#python merge.py $layer
