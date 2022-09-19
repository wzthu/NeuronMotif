layer=$1
kernel=$2

chen2meme layer$layer/kernel-$kernel-redict.chen > layer$layer/kernel-$kernel-redict.meme
tomtom --norc  layer$layer/kernel-$kernel-redict.meme layer$layer/kernel-$kernel-redict.meme -oc layer$layer/kernel-$kernel-redict-redict
