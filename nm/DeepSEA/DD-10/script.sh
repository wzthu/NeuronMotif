conda  activate NeuronMotif
wget  --no-check-certificate -O weight.hdf5  http://bioinfo-xwwang-thu.cn/zwei/NeuronMotif/DCNN_weight/DeepSEA/DD-10/weight.hdf5
bash run.layer.sh 1 128 20
bash run.layer.sh 2 128 20
bash run.layer.sh 3 160 20
bash run.layer.sh 4 160 20
bash run.layer.sh 5 256 20
bash run.layer.sh 6 320 20
bash run.layer.sh 7 512 20
bash run.layer.sh 8 640 20
bash run.layer.sh 9 1024 20
bash run.layer.sh 10 1280 20

python tochenDeepSEA.py 1
python tochenDeepSEA.py 2
python tochenDeepSEA.py 3
python tochenDeepSEA.py 4
python tochenDeepSEA.py 5
python tochenDeepSEA.py 6
python tochenDeepSEA.py 7
python tochenDeepSEA.py 8
python tochenDeepSEA.py 9
python tochenDeepSEA.py 10

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
bash vis.layer.sh 10 28


