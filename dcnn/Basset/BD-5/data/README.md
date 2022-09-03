# Basset data

Download the data for training the DCNN:

```
wget  --no-check-certificate   http://bioinfo-xwwang-thu.cn/zwei/NeuronMotif/DCNN_train/Basset_data.tar.gz
```
Alternatively, if the link above is not available temperately, you can download from https://cloud.tsinghua.edu.cn/d/fee522536d524eae9531/files/?p=%2FDCNN_train%2FBasset_data.tar.gz&dl=1

```
tar -xzvf Basset_data.tar.gz

mv data/* ./
```

# Reference

Kelley, D. R., Snoek, J., & Rinn, J. L. (2016). Basset: learning the regulatory code of the accessible genome with deep convolutional neural networks. Genome research, 26(7), 990-999.
