"""
    Do some test
"""
import torch
from train_res3d.resnet3d_18 import DeformResNet3d
import pandas as pd
import os
from train_res3d import util
import numpy as np
import time
from data_set import dataset
from train_res3d import resnet3d_18, train_val_model
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-class_num', default=27)
parser.add_argument('-batch_size', default=16)
parser.add_argument('-pre_trained_model', default='deform_jes_l3b-15742.state')
parser.add_argument('-clip_length', default=32)
parser.add_argument('-crop_shape', default=[100, 100])  # must be same for rotate
parser.add_argument('-mean', default=[0.45, 0.43, 0.41])  # cha[124,108,115]ego[114,123,125]ucf[101,97,90]k[]
parser.add_argument('-std', default=[0.23, 0.24, 0.23])
parser.add_argument('-device_id', default=[0])
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

args = parser.parse_args()
a = args.class_num
b = args.std
t = time.time()
f = pd.read_csv('/home/lshi/Database/Jester/jester-v1-test.csv', header=None)
data_set = dataset.JesterImageFolder('test', args)
data_set_loaders = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, num_workers=10, drop_last=False)
print(time.time() - t)
clip, labels = next(iter(data_set_loaders))
print(time.time() - t)
