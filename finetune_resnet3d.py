"""
This program is for fine_tuning the Alex_net resnet3d_model for the final fc layer with the data set of 'Caltech256'.

Data set: http://www.vision.caltech.edu/Image_Datasets/Caltech256/

Usage: You should firstly download the data and exact to the directory of fine_tuning.py.
       Make sure the name of data dir is '256_ObjectCategories'. Then run the data.py.
       Finally, run the fine_tuning.py
"""
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.serialization import load_lua
from tensorboard_logger import configure
import os
import shutil
import resnet3d
import util
import dataset
import train_val_model

# params
LR = 0.001
MOMENTUM = 0.9
LR_DECAY_RATIO = 0.1
NUM_EPOCH = 20
NUM_EPOCH_SAVE_MODEL = 2
CLASS_NUM = 101
BATCH_SIZE = 256
DEVICE_ID = [0, 1, 2, 3, 4, 5, 6, 7]
LOG_DIR = "runs/test"
LAST_MODEL = 'resnet3d-finetuning-18_1000.state'
USE_LAST_MODEL = False
ONLY_TRAIN_CLASSIFIER = False

# for tensorboard --logdir runs
if os.path.isdir(LOG_DIR) and not USE_LAST_MODEL:
    shutil.rmtree(LOG_DIR)
    print('dir removed: ', LOG_DIR)
configure(LOG_DIR)

# Date reading, setting for batch size, whether shuffle, num_workers
data_dir = '/home/lshi/Database/UCF-101/'
data_set = {x: dataset.UCFImageFolder(os.path.join(data_dir, x), (x is 'train')) for x in ['train', 'val']}
data_set_loaders = {x: torch.utils.data.DataLoader(data_set[x], batch_size=BATCH_SIZE, shuffle=True,
                                                   num_workers=50, drop_last=False) for x in ['train', 'val']}

data_set_classes = data_set['train'].classes
util.write_class_txt(data_set_classes)

# show examples of input
util.batch_show(data_set_loaders, data_set_classes)
util.clip_show(data_set_loaders, data_set_classes)

use_gpu = torch.cuda.is_available()

resnet3d_model = resnet3d.resnet18(pretrained=True)

if ONLY_TRAIN_CLASSIFIER is True:
    for param in resnet3d_model.parameters():
        param.requires_grad = False
resnet3d_model.fc = nn.Linear(512, CLASS_NUM)

global_step = 0
if USE_LAST_MODEL is True:
    resnet3d_model.load_state_dict(torch.load(LAST_MODEL))
    global_step = int(LAST_MODEL[:-6])

if use_gpu:
    resnet3d_model = resnet3d_model.cuda()

entropy_loss = nn.CrossEntropyLoss()

adam_optimizer = optim.Adam(resnet3d_model.parameters(), lr=LR)

loss, acc = train_val_model. \
    train_val_model(resnet3d_model, data_set_loaders, entropy_loss, adam_optimizer,
                    num_epoch_save=NUM_EPOCH_SAVE_MODEL, use_gpu=use_gpu, num_epochs=NUM_EPOCH,
                    global_step=global_step, device_id=DEVICE_ID, batch_size=BATCH_SIZE,
                    lr_decay_ratio=LR_DECAY_RATIO)
