mkdir layer${1}
#lk=layer${1}/lock${3}
#if [[ ! -e $lk ]]; then
#    mkdir layer${1}/lock${3}
    python tochen.py $1 $3 $2
    chen2meme layer${1}/kernel-${3}.all.ppm.chen > layer${1}/kernel-${3}.all.ppm.meme
    chen2meme layer${1}/kernel-${3}.sm.all.ppm.chen > layer${1}/kernel-${3}.sm.all.ppm.meme
    chen2meme layer${1}/kernel-${3}.sel.ppm.chen > layer${1}/kernel-${3}.sel.ppm.meme
    chen2meme layer${1}/kernel-${3}.sm.sel.ppm.chen > layer${1}/kernel-${3}.sm.sel.ppm.meme
#    tomtom layer${1}/kernel-${3}.all.ppm.meme empty.meme -o layer${1}/tomtom_${3}.all.ppm.meme
#    tomtom layer${1}/kernel-${3}.sm.all.ppm.meme empty.meme -o layer${1}/tomtom_${3}.sm.all.ppm.meme
    tomtom layer${1}/kernel-${3}.sel.ppm.meme JASPAR2020.txt -o layer${1}/tomtom_${3}.sel.ppm.meme
#fi
