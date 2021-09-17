
conda  activate NeuronMotif
wget  --no-check-certificate -O weight.hdf5  http://bioinfo-xwwang-thu.cn/zwei/NeuronMotif/DCNN_weight/Basset/BD-10/weight.hdf5

bash run.layer.sh 1 128 20
bash run.layer.sh 2 128 20
bash run.layer.sh 3 160 20
bash run.layer.sh 4 160 20
bash run.layer.sh 5 256 20
bash run.layer.sh 6 256 20
bash run.layer.sh 7 384 20
bash run.layer.sh 8 384 20
bash run.layer.sh 9 512 5
bash run.layer.sh 10 512 5

python tochen.py 1
python tochen.py 2
python tochen.py 3
python tochen.py 4
python tochen.py 5
python tochen.py 6
python tochen.py 7
python tochen.py 8
python tochen.py 9
python tochen.py 10

wget -O  motifDB.txt  http://jaspar.genereg.net/download/CORE/JASPAR2020_CORE_vertebrates_redundant_pfms_meme.txt

bash vis.layer.sh 1 28
bash vis.layer.sh 2 28
bash vis.layer.sh 3 28
bash vis.layer.sh 4 28
bash vis.layer.sh 5 28
bash vis.layer.sh 6 28
bash vis.layer.sh 7 28
bash vis.layer.sh 8 28
bash vis.layer.sh 9 28
bash vis.layer.sh 10 25


