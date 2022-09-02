layer=$1
kernel=$2

chen2meme layer$layer/kernel-$kernel-unified-dict.chen > layer$layer/kernel-$kernel-unified-dict.meme
chen2meme layer$layer/kernel-$kernel-segs.chen > layer$layer/kernel-$kernel-segs.meme
tomtom --norc  layer$layer/kernel-$kernel-segs.meme layer$layer/kernel-$kernel-unified-dict.meme -oc layer$layer/kernel-$kernel-segs-dict
