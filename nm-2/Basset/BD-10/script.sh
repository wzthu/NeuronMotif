cp ../../../nm/Basset/BD-10/modeldef.py ./
cp ../../../nm/Basset/BD-10/weight.hdf5 ./
bash runlink.sh ../../../../nm/Basset/BD-10/layer10 layer10 512
wget -O motifDB.txt  http://jaspar.genereg.net/download/CORE/JASPAR2020_CORE_vertebrates_redundant_pfms_meme.txt

bash run.layer.sh 10 512 5
bash vis.layer.sh 10 512 28 0
```
