mkdir layer${1}
lk=layer${1}/lock${3}
if [[ ! -e $lk ]]; then
    mkdir layer${1}/lock${3}
    python kmeans.py $1 $3 $2
fi
