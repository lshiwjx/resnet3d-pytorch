import torch
from train_deform3d.deform_resnet3d_18 import DeformResNet3d
import pandas as pd
import os
from train_res3d import util
import numpy as np

# t = torch.load('train_deform3d/off_tensor1')
# d000 = t.cpu().data[0, :, 0, 2, 2]
# f = t.cpu().data.numpy()
# fm = np.max(f)
# fmi = np.min(f)
# a = d000.numpy()
# a = np.reshape(a, (128, 3, 3, 3, 3))
# for i in range(128):
#     print(str(i) + ': ')
#     print(a[0, 0, :, :, :])
#     print(a[0, 1, :, :, :])
#     print(a[0, 2, :, :, :])
#
# b = d000.contiguous().view(128, 3, 3, 3, 3)
# util.off_show(b[0][0], 'off1')
# util.off_show(b[0][1], 'off2')
# util.off_show(b[0][2], 'off3')
# print('none')


# jester_root = '/home/lshi/Database/Jester'
# label_csv = os.path.join(jester_root, 'jester-v1-labels.csv')
# test_csv = os.path.join(jester_root, 'jester-v1-test.csv')

# train_csv = os.path.join(jester_root, 'train.csv')
# val_csv = os.path.join(jester_root, 'val.csv')

# f1 = pd.read_csv(label_csv, header=None)
# f2 = pd.read_csv(test_csv, header=None)
# f3 = pd.read_csv(train_csv, header=None)
# f4 = pd.read_csv(val_csv, header=None)
# print(f3.loc[0, 0])
# print(f3.loc[0, 1])
# f4.loc[0, 1]=9224
# print(f1)
# print(f2)
# print(f3)
# print(f4)
# for i in range(len(f4)):
#     num, label = f4.iloc[i]
#     for j in range(len(f1)):
#         if f1[0][j] == label:
#             f4.loc[i, 1] = j
#             break
# f4.to_csv(os.path.join(jester_root, 'val.csv'),header=None,index=None)

# for i in range(len(f3)):
#     num, label = f3.iloc[i]
#     for j in range(len(f1)):
#         if f1[0][j] == label:
#             f3.loc[i, 1] = j
#             break
# f3.to_csv(os.path.join(jester_root, 'train.csv'),header=None,index=None)
