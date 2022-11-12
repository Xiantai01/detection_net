# detection_net
Unsupervised learning + detection for NEU_data
# Background
This is a PyTorch implementation of the improved Faster-R-CNN for NEU surface defect database

This project uses Contrastive Learning (SimSiam) to pre-train the backbone and preserve high-resolution features through the feature pyramid structure (FPN). In addition, lightweight modules such as shuffleNet and attention modules such as CBAM have also been explored.

# Environment
The code is developed using python 3.9 on Ubuntu 20.04. NVIDIA GPUs are needed. The code is developed and tested using 1 NVIDIA 3060 cards. Other platforms or GPU cards are not fully tested.

# Prepare datasets
It is recommended to symlink the dataset root to `./NEU-DEF`. If your folder structure is different, you may need to change the corresponding paths.

**For NEU-DEF data**, please download from [NEU-DEF download](http://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm).
Download and extract them under `./NEU-DEF`, and make them look like this:
```
detection_net
├── backbone
├── network_files
├── train_utils
|── NEU-DEF
    │-- Annotations
    |-- JPEGImages
```
**For Contrastive Learning**, please download data from Kaggle ([Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection))

# Usage

##  SimSiam
Please refer to an official PyTorch implementation of the [SimSiam](https://github.com/facebookresearch/simsiam) paper
Using the kaggle dataset, different backbones can be trained.
**Our pre-trained models are listed here**.
| backbone| pre-train ckpt|
| ------ | ------------- |
| ResNet50| link|
| lite-HRNet18 | link|

