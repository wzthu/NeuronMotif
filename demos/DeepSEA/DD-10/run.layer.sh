layer=$1
kernels=$2
threads=$3
motif_nb=$4
bash idx.sh  $kernels  | xargs -n 1 -P $threads  bash   sampling.sh $layer 
#rm -rf layer$layer/lock*
python decouple.py  $threads  $layer $kernels  $motif_nb 
bash idx.sh  $kernels  | xargs -n 1 -P $threads  bash   visualize.sh $layer  $kernels
bash idx.sh  $kernels  | xargs -n 1 -P $threads bash tree.sh $layer 
