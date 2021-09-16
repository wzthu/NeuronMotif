mkdir layer${1}/vis/result/${2}
chen2meme layer${1}/vis/${2} > layer${1}/vis/result/$2/${2}.meme
tomtom  -no-ssc  layer${1}/vis/result/${2}/${2}.meme motifDB.txt  -o layer${1}/vis/result/${2}/${2}.out  > layer${1}/vis/result/${2}/${2}.stdout.txt
