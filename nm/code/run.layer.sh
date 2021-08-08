layer=$1
kernels=$2
threads=$3
sh idx.sh  $kernels  | xargs -n 1 -P $threads  sh   run.sh $layer
python merge.py $layer
