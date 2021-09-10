layer=$1
kernels=$2
threads=$3
isDeepSEA=$4
bash idx.sh  $kernels  | xargs -n 1 -P $threads  bash   run.tomtom.sh $layer $isDeepSEA
#python merge.py $layer
