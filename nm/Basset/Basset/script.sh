
conda  activate NeuronMotif
wget  --no-check-certificate -O weight.hdf5  http://bioinfo-xwwang-thu.cn/zwei/NeuronMotif/DCNN_weight/Basset/Basset/weight.hdf5

bash run.layer.sh 1 300 20 
bash run.layer.sh 2 200 20
bash run.layer.sh 3 200 20

python tochen.py 1
python tochen.py 2
python tochen.py 3

wget -O  motifDB.txt  http://jaspar.genereg.net/download/CORE/JASPAR2020_CORE_vertebrates_redundant_pfms_meme.txt

bash vis.layer.sh 1 28
bash vis.layer.sh 2 28
bash vis.layer.sh 3 28
```


