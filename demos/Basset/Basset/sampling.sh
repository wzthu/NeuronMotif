mkdir layer${1}
lk=layer${1}/samplinglock${2}
if [[ ! -e $lk ]]; then
    mkdir layer${1}/samplinglock${2}
    sp=layer${1}/kernel-${2}.h5
    if [[ ! -e $sp ]]; then
        python vis.py $1 $2
    fi
fi
