mkdir layer${1}
lk=layer${1}/vislock${3}
if [[ ! -e $lk ]]; then
    mkdir layer${1}/vislock${3}
    python chen2html.py layer${1}/kernel-${3}.ppm.h5 ${3}  layer${1}/${3} ${2}
    python tochen.py $1 $3
    chen2meme layer${1}/kernel-${3}.sel.ppm.chen > layer${1}/kernel-${3}.sel.ppm.meme
    tomtom layer${1}/kernel-${3}.sel.ppm.meme motifDB.txt -o layer${1}/tomtom_${3}.sel.ppm.meme
    chen2meme layer${1}/kernel-${3}.all.ppm.chen > layer${1}/kernel-${3}.all.ppm.meme
#    tomtom layer${1}/kernel-${3}.all.ppm.meme motifDB.txt -o layer${1}/tomtom_${3}.all.ppm.meme
fi
