# A Coarse-to-fine Hierarchical Fine-tuning Model for Monocular Depth Estimation
([Article PDF](https://link.springer.com/article/10.1007/s00138-024-01560-0?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=nonoa_20240606&utm_content=10.1007%2Fs00138-024-01560-0))
### Datasets
You can prepare the datasets KITTI and NYUv2 according to [here](https://github.com/cleinc/bts), and then modify the data path in the config files to your dataset locations.

### Training
First download the pretrained encoder backbone from [here](https://github.com/microsoft/Swin-Transformer), and then modify the pretrain path in the config files.

Training the NYUv2 model:
```
python nyu_train.py
```

Training the KITTI model:
```
python kitti_train.py
```


### Evaluation
Evaluate the NYUv2 model:
```
python nyu_test.py
```

Evaluate the KITTI model:
```
python kitti_test.py
```