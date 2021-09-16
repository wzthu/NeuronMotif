mkdir layer$1/vis/result
ls layer$1/vis/*.chen | xargs -n 1 basename | xargs -n 1 -P $2  bash vis.sh $1
python chen2html.py $1
