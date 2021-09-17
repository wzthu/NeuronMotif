cp ../../../nm/DeepSEA/DD-10/modeldef.py ./
cp ../../../nm/DeepSEA/DD-10/weight.hdf5 ./
bash runlink.sh ../../../../nm/DeepSEA/DD-10/layer10 layer10 1280
wget -O motifDB.txt  http://jaspar.genereg.net/download/CORE/JASPAR2020_CORE_vertebrates_redundant_pfms_meme.txt
bash run.layer.sh 10 1280 5
bash vis.layer.sh 10 1280 28 1
