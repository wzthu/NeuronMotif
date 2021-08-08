mkdir layer${1}
lk=layer${1}/lock${2}
if [[ ! -e $lk ]]; then
    mkdir layer${1}/lock${2}
    python vis.py $1 $2
    python kmeans.py $1 $2
fi
