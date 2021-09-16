layer=$1
kernels=$2
threads=$3
deepsea=$4
sh idx.sh  $kernels  | xargs -n 1 -P $threads  sh   vis.sh $layer $deepsea
#python merge.py $layer
