# Train DCNN model

## Download data

For the models trained by Basset dataset or DeepSEA dataset, please go to the data folder in Basset folder or DeepSEA folder. 
There are instructions for downloading the training data.

## Train model

Take BD-10 model as an example, go to the model folder and run:

```
CUDA_VISIBLE_DEVICES=0 python BD-10.py
```

CUDA_VISIBLE_DEVICES is the GPU ID in your server

## Test model

Take BD-10 model as an example, go to the model folder and run:

```
CUDA_VISIBLE_DEVICES=0 python BD-10.pred.py
```
