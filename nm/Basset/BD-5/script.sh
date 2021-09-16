conda  activate NeuronMotif
wget  --no-check-certificate -O weight.hdf5  http://bioinfo-xwwang-thu.cn/zwei/NeuronMotif/DCNN_weight/Basset/BD-5/weight.hdf5

bash run.layer.sh 1 64 20
bash run.layer.sh 2 128 20
bash run.layer.sh 3 256 20
bash run.layer.sh 4 384 20
bash run.layer.sh 5 512 20
python tochen.py 1
python tochen.py 2
python tochen.py 3
python tochen.py 4
python tochen.py 5
wget -O  motifDB.txt  http://jaspar.genereg.net/download/CORE/JASPAR2020_CORE_vertebrates_redundant_pfms_meme.txt
bash vis.layer.sh 1 28
bash vis.layer.sh 2 28
bash vis.layer.sh 3 28
bash vis.layer.sh 4 28
bash vis.layer.sh 5 28


