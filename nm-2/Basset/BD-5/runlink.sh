mkdir $2
tt=$3
tt=$[tt-1]
for i in `seq 0 $tt`
do
    ln -s $1/kernel-$i.h5 $2/kernel-$i.h5
done
