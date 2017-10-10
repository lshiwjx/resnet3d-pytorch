"""
This program is for fine_tuning the Alex_net model for the final fc layer with the data set of 'Caltech256'.

Data set: http://www.vision.caltech.edu/Image_Datasets/Caltech256/

Usage: You should firstly download the data and exact to the directory of fine_tuning.py.
       Make sure the name of data dir is '256_ObjectCategories'. Then run the data.py.
       Finally, run the fine_tuning.py
"""
from __future__ import print_function, division

import argparse
import os

import torch
from torch.utils.data import DataLoader

from data_set import dataset
from train_res3d import resnet3d_18, train_val_model

# params
parser = argparse.ArgumentParser()
# parser.add_argument('-class_num', default=83)
parser.add_argument('-class_num', default=27)

parser.add_argument('-batch_size', default=4)

# parser.add_argument('-pre_trained_model', default='resnet3d_max_18-14975.state')
parser.add_argument('-pre_trained_model', default='jester_std32_base-14800.state')

# parser.add_argument('-clip_length', default=32)
# parser.add_argument('-resize_shape', default=[240, 320])
# parser.add_argument('-crop_shape', default=[224, 224])  # must be same for rotate
# parser.add_argument('-mean', default=[114 / 1, 123 / 1, 125 / 1])
# parser.add_argument('-std', default=[0.229, 0.224, 0.225])

parser.add_argument('-clip_length', default=32)
parser.add_argument('-crop_shape', default=[100, 100])  # must be same for rotate
parser.add_argument('-mean', default=[0.45, 0.43, 0.41])  # cha[124,108,115]ego[114,123,125]ucf[101,97,90]k[]
parser.add_argument('-std', default=[0.23, 0.24, 0.23])

parser.add_argument('-device_id', default=[0])
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '5,4,2,1'
# parser.add_argument('-device_id', default=[0])

args = parser.parse_args()

# data_dir = '/home/lshi/Database/Ego_gesture/val/'
# data_set = dataset.EGOImageFolderPillow(data_dir, False, args)
# data_dir = '/home/lshi/Database/UCF-101/val/'
data_set = dataset.JesterImageFolder(False, args)
data_set_loaders = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True)

# model = resnet3d.resnet18(args.class_num, args.clip_length, args.crop_shape)
model = resnet3d_18.ResNet3d(args.class_num, args.clip_length, args.crop_shape)
model.load_state_dict(torch.load(args.pre_trained_model))
print('Pretrained model load finished: ', args.pre_trained_model)

use_gpu = torch.cuda.is_available()
print('Use gpu? ', use_gpu)
if use_gpu:
    model = model.cuda()

loss_function = torch.nn.CrossEntropyLoss(size_average=True)
print('Using CrossEntropy loss with average')

print('Validate')
loss, acc = train_val_model.validate(model, data_set_loaders, loss_function, args.batch_size,
                                     use_gpu, args.device_id)
