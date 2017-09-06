"""
This program is for fine_tuning the Alex_net model for the final fc layer with the data set of 'Caltech256'.

Data set: http://www.vision.caltech.edu/Image_Datasets/Caltech256/

Usage: You should firstly download the data and exact to the directory of fine_tuning.py.
       Make sure the name of data dir is '256_ObjectCategories'. Then run the data.py.
       Finally, run the fine_tuning.py
"""
from __future__ import print_function, division

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboard_logger import configure, log_value

import os
import time
import shutil

import resnet3d
import dataset
import train_val_model
import util

# params
LR = 0.001
CLASS_NUM = 101
BATCH_SIZE = 400
DEVICE_ID = [0, 1, 2, 3, 4, 5, 6, 7]
WEIGHT_DECAY_RATIO = 0.0001
MAX_EPOCH = 20
NUM_EPOCH_PER_SAVE = 2
LR_DECAY_RATIO = 0.5
LOG_DIR = "./runs/test"
MODEL_NAME = 'resnet3d_finetuning_34-'
LAST_MODEL = MODEL_NAME + '0.state'
USE_LAST_MODEL = True
ONLY_TRAIN_CLASSIFIER = False

# for tensorboard --logdir runs
if os.path.isdir(LOG_DIR) and not USE_LAST_MODEL:
    shutil.rmtree(LOG_DIR)
    print('Dir removed: ', LOG_DIR)
configure(LOG_DIR)

# Date reading, setting for batch size, whether shuffle, num_workers
data_dir = '/home/lshi/Database/UCF-101/'
data_set = {x: dataset.UCFImageFolder(os.path.join(data_dir, x), (x is 'train')) for x in ['train', 'val']}
data_set_loaders = {x: DataLoader(data_set[x], batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=32, drop_last=True, pin_memory=True)
                    for x in ['train', 'val']}

data_set_classes = data_set['train'].classes
# util.write_class_txt(data_set_classes)

# show examples of input
print('Show examples of input')
# clip, classes = next(iter(data_set_loaders['train']))
# util.batch_show(clip,classes, data_set_classes)
# util.clip_show(clip,classes, data_set_classes)

# model = resnet3d.resnet18(pretrained=True)
model = resnet3d.resnet34(pretrained=True)
print('Pretrained model load finished: ', MODEL_NAME[:-2])

if ONLY_TRAIN_CLASSIFIER is True:
    print('Only train classifier with weight decay: ', WEIGHT_DECAY_RATIO)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Linear(512, CLASS_NUM)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR, weight_decay=WEIGHT_DECAY_RATIO)
else:
    print('Train all params with weight decay: ', WEIGHT_DECAY_RATIO)
    model.fc = torch.nn.Linear(512, CLASS_NUM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY_RATIO)

global_step = 0
# The name for model must be **_**-$(step).state
if USE_LAST_MODEL is True:
    model.load_state_dict(torch.load(LAST_MODEL))
    global_step = int(LAST_MODEL[:-6].split('-')[1])
    print('Training continue, last model load finished, step is ', global_step)
else:
    print('Training from scratch, step is ', global_step)

use_gpu = torch.cuda.is_available()
print('Use gpu? ', use_gpu)
if use_gpu:
    model = model.cuda()

loss_function = torch.nn.CrossEntropyLoss(size_average=True)
print('Using CrossEntropy loss with average')

lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=LR_DECAY_RATIO, patience=2,
                                 verbose=True, threshold=0.1, threshold_mode='abs', cooldown=1)
print('LR scheduler: LR:{} DecayRatio:{} Patience:2 Threshold:0.1 Before_epoch:1'
      .format(LR, LR_DECAY_RATIO))

print('Train and val begin, total epoch: ', MAX_EPOCH)
for epoch in range(MAX_EPOCH):
    time_start = time.time()
    print('Epoch {}/{}'.format(epoch, MAX_EPOCH - 1))
    print('Train')
    global_step = train_val_model.train(model, data_set_loaders['train'], loss_function, optimizer,
                                        global_step, use_gpu, DEVICE_ID)
    print('Validate')
    loss, acc = train_val_model.validate(model, data_set_loaders['val'],
                                         loss_function, use_gpu, DEVICE_ID)
    log_value('val_loss', loss, global_step)
    log_value('val_acc', acc, global_step)
    lr_scheduler.step(acc)
    time_elapsed = time.time() - time_start
    lr = optimizer.param_groups[0]['lr']
    log_value('lr', lr)
    print('validate loss: {:.4f} acc: {:.4f} lr: {}'.format(loss, acc, lr))

    # save model
    if epoch % NUM_EPOCH_PER_SAVE == 0 and epoch != 0:
        torch.save(model.state_dict(), MODEL_NAME + str(global_step) + '.state')
        print('Save model at step ', global_step)

    print('Epoch {} finished, using time: {:.0f}m {:.0f}s'.
          format(epoch, time_elapsed // 60, time_elapsed % 60))
