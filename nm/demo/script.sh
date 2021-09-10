bash run.layer.sh 1 3 3
bash run.layer.sh 2 1 1
python tochen.py 1
python tochen.py 2
wget -O  motifDB.txt  http://jaspar.genereg.net/download/CORE/JASPAR2020_CORE_vertebrates_redundant_pfms_meme.txt
bash vis.layer.sh 1 3
bash vis.layer.sh 2 1
