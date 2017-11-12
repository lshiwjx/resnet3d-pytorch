"""
    Do some test
"""
import argparse
import os
import time

from torch.utils.data import DataLoader

from data_set import dataset
from model import resnet3d_18
import pandas as pd
import torch.nn.functional as f
from torch.autograd import Variable
import torch

# x = torch.FloatTensor([[[[1,2],[3,4]]]])
# a = torch.FloatTensor([[[1,0,1],[0,1,0]]])
# g = f.affine_grid(Variable(a), x.size())
# g = g.data.numpy()
# y = f.grid_sample(x, g)
# x = x.numpy()
# y = y.data.numpy()

# t = time.time()
# t1 = time.time()
# print(time.time()-t)
# t1 = time.time()
# t = time.time()-t
# print('')
# t = time.time()-t
# print(time.time()-t1)
# print('test time')
r = pd.read_csv('/home/lshi/Database/ChaLearn/model_test.csv', header=None)
for i in r.values:
    pass
r.pop(1)
for i, path in enumerate(sorted(os.listdir('/home/lshi/Database/ChaLearn/train'))):
    r.loc[i, 0] = '/home/lshi/Database/ChaLearn/train/' + path
    r.loc[i, 2] = int(r.loc[i, 2]) - 1
    print('now for ', i)
print(r)
# l = pd.read_csv('/home/lshi/Database/ChaLearn/train1.csv', header=None)
# for i in range(len(r)):
#     j = r.loc[i, 2]
#     t = l.loc[int(j), 0]
#     r.loc[i, 1] = t

r.to_csv('/home/lshi/Database/ChaLearn/train.csv', header=None, index=None)

# a= torch.load('/opt/model/cifar10_base_100_0.001_0.0005_p3-20000.state')
# parser = argparse.ArgumentParser()
# parser.add_argument('-class_num', default=27)
# parser.add_argument('-batch_size', default=32)
# parser.add_argument('-pre_trained_model', default='deform_jes_l3b-15742.state')
# parser.add_argument('-clip_length', default=32)
# parser.add_argument('-crop_shape', default=[100, 100])  # must be same for rotate
# parser.add_argument('-mean', default=[0.45, 0.43, 0.41])  # cha[124,108,115]ego[114,123,125]ucf[101,97,90]k[]
# parser.add_argument('-std', default=[0.23, 0.24, 0.23])
# parser.add_argument('-device_id', default=[0])
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
#
# args = parser.parse_args()
# m = resnet3d_18.DeformResNet3d(args.class_num, args.clip_length, args.crop_shape)
# a = m.state_dict()
# m = m.cuda()
# b = m.state_dict()
# print('test save model')

#
# params = []
# for key, value in m.named_parameters():
#     if key[8:16] == 'conv_off':
#         params += [{'params': [value], 'lr': 0.01}]
#     else:
#         params += [{'params': [value], 'lr': 0.1}]
#
# for a in m.named_parameters():
#     print(a)
# a = list(m.parameters())
# a = m.named_children()
# b = dict(m.named_parameters())
# c = list(m.modules())
# d = dict(m.named_modules())
# e = list(m.children())
# f = dict(m.named_children())
# b = args.std
# t = time.time()
# f = pd.read_csv('/home/lshi/Database/Jester/jester-v1-test.csv', header=None)
# data_set = dataset.JesterImageFolder('model_test', args)
# data_set_loaders = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)
# print(time.time() - t)
# clip, labels = next(iter(data_set_loaders))
# print(time.time() - t)
