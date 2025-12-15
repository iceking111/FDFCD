# FDRFCD:Feature Disentangling Representation and Fusion Deep Network for Remote Sensing Image Change Detection
Here, we provide the pytorch implementation of the paper: FDRFCD:Feature Disentangling Representation and Fusion Deep Network for Remote Sensing Image Change Detectio

For more ore information, please see our published paper at [Springer](https://link.springer.com/chapter/10.1007/978-981-96-9863-9_37). 

![](https://pic1.imgdb.cn/item/693fcb734a4e4213d0069d04.png)

## Requirements

```
Python 3.6
pytorch 1.6.0
torchvision 0.7.0
einops  0.3.0
```

## Installation

Clone this repo:

```shell
git clone https://github.com/iceking111/FDFCD.git
cd models
```



## Train

```python
python trainer.py
```



## Evaluate

```python
python evaluator.py
```


## Dataset Preparation

### Data structure

```
"""
Change detection data set with pixel-level binary labels；
├─A
├─B
└─label
"""
```

`A`: images of t1 phase;

`B`:images of t2 phase;

`label`: label maps;


### Data Download 

LEVIR-CD: https://justchenhao.github.io/LEVIR/

## License

Code is released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.

## Citation

If you use this code for your research, please cite our paper:

```
@InProceedings{10.1007/978-981-96-9863-9_37,
author="Zhao, Kang
and Zhao, Xinyu
and Wang, Bin
and Qin, Pinle
and Zeng, Jianchao",
editor="Huang, De-Shuang
and Pan, Yijie
and Chen, Wei
and Li, Bo",
title="FDRFCD: Feature Disentangling Representation and Fusion Deep Network for Remote Sensing Image Change Detection",
booktitle="Advanced Intelligent Computing Technology and Applications",
year="2025",
publisher="Springer Nature Singapore",
address="Singapore",
pages="435--447",
abstract="Aiming at the problem that the hybrid feature extraction and fusion adopted by current deep networks for remote sensing change detection (RSCD) easily leads to blurred boundaries of predicted categories, a Feature Disentangling Representation and Fusion Deep Network (FDRFCD) is proposed. The network uses the Mamba Out for feature extraction, which can effectively capture local and global multi-scale fused features. Meanwhile, a feature decoupling module is designed to distinguish the unique and shared features of each-phase image, improving the detection accuracy of changed and unchanged regions. To avoid information loss during the decoupling process, the decoupled features are reconstructed through the fusion module to ensure the completeness of information. Finally, a Positive Example Threshold Pull (PETP) loss regularization term is introduced to enhance the attention to positive samples. The experimental results show that this network is superior to other models in both quantitative analysis and qualitative analysis.",
isbn="978-981-96-9863-9"
}


```
