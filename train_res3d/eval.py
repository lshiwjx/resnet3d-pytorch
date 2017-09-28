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
from train_res3d import resnet3d, train_val_model

# params
parser = argparse.ArgumentParser()
# parser.add_argument('-class_num', default=83)
parser.add_argument('-class_num', default=101)

parser.add_argument('-batch_size', default=32)

# parser.add_argument('-pre_trained_model', default='resnet3d_max_18-14975.state')
parser.add_argument('-pre_trained_model', default='resnet3d_finetuning_34-513.state')

# parser.add_argument('-clip_length', default=32)
# parser.add_argument('-resize_shape', default=[240, 320])
# parser.add_argument('-crop_shape', default=[224, 224])  # must be same for rotate
# parser.add_argument('-mean', default=[114 / 1, 123 / 1, 125 / 1])
# parser.add_argument('-std', default=[0.229, 0.224, 0.225])

parser.add_argument('-clip_length', default=16)
parser.add_argument('-resize_shape', default=[120, 160])
parser.add_argument('-crop_shape', default=[112, 112])  # must be same for rotate
parser.add_argument('-mean', default=[101.6 / 255, 97.6 / 255, 90.3 / 255])
parser.add_argument('-std', default=[0.229, 0.224, 0.225])

parser.add_argument('-device_id', default=[0, 1, 2, 3])
os.environ['CUDA_VISIBLE_DEVICES'] = '7,3,6,0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '5,4,2,1'
# parser.add_argument('-device_id', default=[0])

args = parser.parse_args()

# data_dir = '/home/lshi/Database/Ego_gesture/val/'
# data_set = dataset.EGOImageFolderPillow(data_dir, False, args)
data_dir = '/home/lshi/Database/UCF-101/val/'
data_set = dataset.UCFImageFolder(data_dir, False, args)
data_set_loaders = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True)

# model = resnet3d.resnet18(args.class_num, args.clip_length, args.crop_shape)
model = resnet3d.resnet34(args.class_num, args.clip_length, args.crop_shape)
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
