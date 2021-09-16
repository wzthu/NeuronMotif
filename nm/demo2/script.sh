
# conda  activate NeuronMotif

bash run.layer.sh 1 5 28
bash run.layer.sh 2 5 28
bash run.layer.sh 3 6 28
bash run.layer.sh 4 6 28
bash run.layer.sh 5 1 28

python tochen.py 1
python tochen.py 2
python tochen.py 3
python tochen.py 4
python tochen.py 5

# wget -O  motifDB.txt  http://jaspar.genereg.net/download/CORE/JASPAR2020_CORE_vertebrates_redundant_pfms_meme.txt

bash vis.layer.sh 1 28
bash vis.layer.sh 2 28
bash vis.layer.sh 3 28
bash vis.layer.sh 4 28
bash vis.layer.sh 5 28

