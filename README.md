# detection_net
Unsupervised learning + detection for NEU_data
# Background
This is a PyTorch implementation of the improved Faster-R-CNN for NEU surface defect database

This project uses **Contrastive Learning** (SimSiam) to pre-train the backbone and preserve high-resolution features through the **feature pyramid structure** (FPN). In addition, lightweight modules such as **shuffleNet** and attention modules such as **CBAM** have also been explored.

# Environment

The code is developed using python 3.9 on Ubuntu 20.04. NVIDIA GPUs are needed. The code is developed and tested using 1 NVIDIA 3060 card. Other platforms or GPU cards are not fully tested.

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

## Train

### Split data
```python
python split_data.py
```
By default, the training set and test set are divided according to 8:2(You can change ratio through `val_rate`), and the names are written into train.txt and val.txt respectively.

###  SimSiam

Please refer to an official PyTorch implementation of the [SimSiam](https://github.com/facebookresearch/simsiam) paper.
Using the kaggle dataset, different backbones can be trained.

**Our pre-trained models are listed here**.

| backbone| pre-train ckpt|
| ------ | ------------- |
| ResNet50| [link]|
| Lite-HRNet18 | [link]|
|MobileNetv2|[link]|
|ShuffleNet|[link]|
| VGG16| [link]|
###  Detection_net

```python
python defectpyramid_train.py --device cuda:0 --data-path ./NEU-DEF --num-classes 6 --epochs 35 --batch_size 16
```

For better performance, you can download our pre-trained backbone, e.g. our ResNet50 and Lite-HRNet18. 

For further improvement, you can also download pre-trained **RPN** weights, e.g. [fasterrcnn_resnet50_fpn_coco.pth](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth).

Don't forget to put them in the `backbone` folder and change the corresponding paths in `defectpyramid_train.py`.

You can make them look like this

```
detection_net
├── backbone
	|--ResNet50.pth
	|--Lite-HRNet18.pth
	|--MobileNetv2.pth
	|--VGG16.pth
	|--fasterrcnn_resnet50_fpn_coco.pth
├── defectpyramid_train.py
```

## Test

```
python predict.py
```

You need to use the trained model weights, change `weight_path` and `original_img` in `predict.py`.

You can make them look like this:

```detection_net
detection_net
├── backbone
	|--ResNet50.pth
	|--Lite-HRNet18.pth
	|--MobileNetv2.pth
	|--VGG16.pth
├── save_weights
	|--Model_ResNet50.pth
	|--Model_LiteHRNet18.pth
├── predict.py
├── network_files
├── train_utils
```

Our models are listed here.

| Model             | ckpt   |
| ----------------- | ------ |
| Model_ResNet50    | [link] |
| Model_LiteHRNet18 | [link] |

# Citation

```
@inproceedings{chen2021exploring,
  title={Exploring simple siamese representation learning},
  author={Chen, Xinlei and He, Kaiming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15750--15758},
  year={2021}
}
@inproceedings{lin2017feature,
  title={Feature pyramid networks for object detection},
  author={Lin, Tsung-Yi and Doll{\'a}r, Piotr and Girshick, Ross and He, Kaiming and Hariharan, Bharath and Belongie, Serge},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2117--2125},
  year={2017}
}
@inproceedings{yu2021lite,
  title={Lite-hrnet: A lightweight high-resolution network},
  author={Yu, Changqian and Xiao, Bin and Gao, Changxin and Yuan, Lu and Zhang, Lei and Sang, Nong and Wang, Jingdong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10440--10450},
  year={2021}
}
@article{ren2015faster,
  title={Faster r-cnn: Towards real-time object detection with region proposal networks},
  author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  journal={Advances in neural information processing systems},
  volume={28},
  year={2015}
}
@inproceedings{woo2018cbam,
  title={Cbam: Convolutional block attention module},
  author={Woo, Sanghyun and Park, Jongchan and Lee, Joon-Young and Kweon, In So},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={3--19},
  year={2018}
}
@inproceedings{ma2018shufflenet,
  title={Shufflenet v2: Practical guidelines for efficient cnn architecture design},
  author={Ma, Ningning and Zhang, Xiangyu and Zheng, Hai-Tao and Sun, Jian},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={116--131},
  year={2018}
}
```
# Conclusion
If you have any questions and ideas, don't hesitate to contact me(xiangxiantai@gmail.com).


