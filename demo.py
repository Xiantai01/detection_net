import transforms
from draw_box_utils import draw_objs
from PIL import Image
import json
import matplotlib.pyplot as plt
import torchvision.transforms as ts
import random
from Xiantai_dataset import NEU_DataSet
import torch
# read class_indict
category_index = {}
try:
    json_file = open('./NEU_classes.json', 'r')
    class_dict = json.load(json_file)
    category_index = {str(v): str(k) for k, v in class_dict.items()}
except Exception as e:
    print(e)
    exit(-1)

x = torch.rand(1, 2, 2)
y = x.flip(1)
c, _, _ = y.shape
print(y.shape)
# data_transform = {
#     "train": transforms.Compose([transforms.ToTensor(),
#                                  transforms.RandomHorizontalFlip(0.5)]),
#     "val": transforms.Compose([transforms.ToTensor()])
# }
#
# # load train data set
# train_data_set = NEU_DataSet('./NEU-DEF', data_transform["train"], "train_1.txt")
# demo = train_data_set[120][-1]['area']
# # a = len(demo)
# area_list = []
# for index in range(len(train_data_set)):
#     area = train_data_set[index][-1]['area']
#     for i in range(len(area)):
#         area_list.append(area[i].item())
# m=0
# for size in area_list:
#     if 5000 < size < 10000:
#     # if size > 39000:
#         m += 1
# print(m)
# plt.hist(area_list, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
# plt.hist(x=area_list,
#          bins=1660,
#         color="steelblue",
#         edgecolor="black")
# plt.show()
# print(len(area_list))