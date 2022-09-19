
layer=$1
kernel=$2
mkdir layer$layer
lk=layer$layer/treelock$kernel
if [[ ! -e $lk ]]; then
    mkdir layer${1}/treelock${2}
    python build_segment.py $layer $kernel
    python build_dict.py $layer $kernel
    CUDA_VISIBLE_DEVICES=-1 python build_tree.py $layer $kernel
    tomtom layer$layer/kernel-$kernel-unified-dict.meme  motifDB.txt -oc layer$layer/tomtom_dict_$kernel
fi
