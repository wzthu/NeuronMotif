layer=$1
kernels=$2
threads=$3
bash idx.sh  $kernels  | xargs -n 1 -P $threads  bash   run.sh $layer
#python merge.py $layer
