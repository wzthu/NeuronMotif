DeepSEA data

Download the data for training the DCNN:

```
wget  --no-check-certificate  http://bioinfo-xwwang-thu.cn/zwei/NeuronMotif/DCNN_train/DeepSEA_data.tar.gz

```

Alternatively, if the link above is not available temperately, you can download from https://cloud.tsinghua.edu.cn/d/fee522536d524eae9531/files/?p=%2FDCNN_train%2FDeepSEA_data.tar.gz&dl=1


```
tar -xzvf DeepSEA_data.tar.gz
mv data/* ./
```

# Reference

Jian Zhou, Olga G. Troyanskaya. Predicting the Effects of Noncoding Variants with Deep learning-based Sequence Model. Nature Methods (2015).
