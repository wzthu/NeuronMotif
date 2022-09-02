layer=$1
kernel=$2

chen2meme layer$layer/kernel-$kernel-dict.chen > layer$layer/kernel-$kernel-dict.meme
#chen2meme layer$layer/kernel-$kernel-test-dict.chen > layer$layer/kernel-$kernel-test-dict.meme
#tomtom --norc  layer$layer/kernel-$kernel-test-dict.meme layer$layer/kernel-$kernel-test-dict.meme -oc layer$layer/kernel-$kernel-dict-dict
tomtom --norc  layer$layer/kernel-$kernel-dict.meme layer$layer/kernel-$kernel-dict.meme -oc layer$layer/kernel-$kernel-dict-dict
