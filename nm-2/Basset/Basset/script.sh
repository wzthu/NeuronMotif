cp ../../../nm/Basset/Basset/modeldef.py ./
cp ../../../nm/Basset/Basset/weight.hdf5 ./

bash runlink.sh ../../../../nm/Basset/Basset/layer3 layer3 200

wget -O motifDB.txt  http://jaspar.genereg.net/download/CORE/JASPAR2020_CORE_vertebrates_redundant_pfms_meme.txt

bash run.layer.sh 3 200 5
bash vis.layer.sh 3 200 28 0
