import numpy as np
import matplotlib.pyplot as plt

file_name_1 = 'Result_faster_rcnn_.txt'
file_name_2 = 'Result_contrastive_ResNet.txt'
file_name_3 = 'Result_contrastive_HRNet.txt'
file_name_4 = 'Result_No pre-trained.txt'
x = np.linspace(0, 35, 35)
a = np.loadtxt(file_name_1, usecols=2)
b = np.loadtxt(file_name_2, usecols=2)
c = np.loadtxt(file_name_3, usecols=2)
d = np.loadtxt(file_name_4, usecols=2)

fig, ax = plt.subplots()

ax.plot(x, b, label='contrastive_ResNet')
# ax.plot(x, e, label='CBAM_ResNet')
ax.plot(x, c, label='contrastive_HRNet')
ax.plot(x, a, label='detection network')
ax.plot(x, d, label='No pre-trained')
ax.legend()
ax.set_xlabel('epoch')
ax.set_ylabel('mAP')
ax.set_title('Eval mAP')
# plt.savefig('../Contrastive mAP.svg', format='svg')
plt.show()

