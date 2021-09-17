cp ../../../nm/DeepSEA/DeepSEA/modeldef.py ./
cp ../../../nm/DeepSEA/DeepSEA/weight.hdf5 ./
bash runlink.sh ../../../../nm/DeepSEA/DeepSEA/layer3 layer3 960
wget -O motifDB.txt  http://jaspar.genereg.net/download/CORE/JASPAR2020_CORE_vertebrates_redundant_pfms_meme.txt
bash run.layer.sh 3 960 5
bash vis.layer.sh 3 960 28 1
```
