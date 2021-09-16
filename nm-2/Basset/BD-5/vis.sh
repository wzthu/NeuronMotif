mkdir layer${1}
#lk=layer${1}/lock${4}
#if [[ ! -e $lk ]]; then
#    mkdir layer${1}/lock${4}
    python chen2html.py layer${1}/kernel-${4}.ppm.h5 $4 layer${1}/${4} $3
    python tochen.py $1 $4 $2
    chen2meme layer${1}/kernel-${4}.all.ppm.chen > layer${1}/kernel-${4}.all.ppm.meme
    chen2meme layer${1}/kernel-${4}.sm.all.ppm.chen > layer${1}/kernel-${4}.sm.all.ppm.meme
    chen2meme layer${1}/kernel-${4}.sel.ppm.chen > layer${1}/kernel-${4}.sel.ppm.meme
    chen2meme layer${1}/kernel-${4}.sm.sel.ppm.chen > layer${1}/kernel-${4}.sm.sel.ppm.meme
#    tomtom layer${1}/kernel-${4}.all.ppm.meme empty.meme -o layer${1}/tomtom_${4}.all.ppm.meme
#    tomtom layer${1}/kernel-${4}.sm.all.ppm.meme empty.meme -o layer${1}/tomtom_${4}.sm.all.ppm.meme
    tomtom layer${1}/kernel-${4}.sel.ppm.meme motifDB.txt -o layer${1}/tomtom_${4}.sel.ppm.meme
#fi
