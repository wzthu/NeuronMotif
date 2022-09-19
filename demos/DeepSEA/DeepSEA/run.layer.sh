layer=$1
kernels=$2
threads=$3
motif_nb=$4
bash idx.sh  $kernels  | xargs -n 1 -P $threads  bash   sampling.sh $layer 
python decouple.py  $threads  $layer $kernels  $motif_nb 
bash idx.sh  $kernels  | xargs -n 1 -P $threads  bash   visualize.sh $layer  $kernels
bash idx.sh  $kernels  | xargs -n 1 -P $threads bash tree.sh $layer 
mkdir HTML
rm -rf HTML/layer$layer
mkdir HTML/layer$layer
cp layer$layer/*.html HTML/layer$layer/
cp -r layer$layer/tomtom_*.sel.ppm.meme HTML/layer$layer/
cp -r layer$layer/tomtom_dict_* HTML/layer$layer/
