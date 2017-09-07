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
import argparse

import resnet3d
import dataset
import train_val_model
import util

# params
parser = argparse.ArgumentParser()
parser.add_argument('-lr', default=0.0001)
parser.add_argument('-class_num', default=101)
parser.add_argument('-batch_size', default=4)
parser.add_argument('-device_id', default=[1, 2, 3, 7])
parser.add_argument('-default_device', default=1)
parser.add_argument('-weight_decay_ratio', default=0.0001)
parser.add_argument('-max_epoch', default=20)
parser.add_argument('-num_epoch_per_save', default=2)
parser.add_argument('-lr_decay_ratio', default=0.5)
parser.add_argument('-lr_patience', default=2)
parser.add_argument('-lr_threshold', default=0.1)
parser.add_argument('-lr_delay', default=1)
parser.add_argument('-log_dir', default="./runs/test")
parser.add_argument('-model_name', default='resnet3d_finetuning_34-')
parser.add_argument('-last_model', default='resnet3d_finetuning_34-972.state')
parser.add_argument('-use_last_model', default=True)
parser.add_argument('-only_train_classifier', default=False)
parser.add_argument('-clip_length', default=16)
parser.add_argument('-mean', default=[0.485, 0.456, 0.406])
parser.add_argument('-resize_shape', default=[120, 160])
parser.add_argument('-crop_shape', default=[112, 112])
args = parser.parse_args()

# for tensorboard --logdir runs
if os.path.isdir(args.log_dir) and not args.use_last_model:
    shutil.rmtree(args.log_dir)
    print('Dir removed: ', args.log_dir)
configure(args.log_dir)

# Date reading, setting for batch size, whether shuffle, num_workers
data_dir = '/home/lshi/Database/UCF-101/'
data_set = {x: dataset.UCFImageFolder(os.path.join(data_dir, x), (x is 'train'), args) for x in ['train', 'val']}
data_set_loaders = {x: DataLoader(data_set[x], batch_size=args.batch_size, shuffle=True,
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
print('Pretrained model load finished: ', args.model_name[:-2])

if args.only_train_classifier is True:
    print('Only train classifier with weight decay: ', args.weight_decay_ratio)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Linear(512, args.class_num)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr, weight_decay=args.weight_decay_ratio)
else:
    print('Train all params with weight decay: ', args.weight_decay_ratio)
    model.fc = torch.nn.Linear(512, args.class_num)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_ratio)

global_step = 0
# The name for model must be **_**-$(step).state
if args.use_last_model is True:
    model.load_state_dict(torch.load(args.last_model, map_location={'cuda:0': 'cuda:7'}))
    global_step = int(args.last_model[:-6].split('-')[1])
    print('Training continue, last model load finished, step is ', global_step)
else:
    print('Training from scratch, step is ', global_step)

use_gpu = torch.cuda.is_available()
print('Use gpu? ', use_gpu)
if use_gpu:
    model = model.cuda(args.default_device)

loss_function = torch.nn.CrossEntropyLoss(size_average=True)
print('Using CrossEntropy loss with average')

lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay_ratio,
                                 patience=args.lr_patience, verbose=True,
                                 threshold=args.lr_threshold, threshold_mode='abs',
                                 cooldown=args.lr_delay)
print('lr scheduler: lr:{} DecayRatio:{} Patience:{} Threshold:{} Before_epoch:{}'
      .format(args.lr, args.lr_decay_ratio, args.lr_patience, args.lr_threshold, args.lr_delay))

print('Train and val begin, total epoch: ', args.max_epoch)
for epoch in range(args.max_epoch):
    time_start = time.time()
    print('Epoch {}/{}'.format(epoch, args.max_epoch - 1))
    print('Train')
    global_step = train_val_model.train(model, data_set_loaders['train'], loss_function, optimizer,
                                        global_step, args.default_device, use_gpu, args.device_id)
    print('Validate')
    loss, acc = train_val_model.validate(model, data_set_loaders['val'], loss_function,
                                         args.default_device, use_gpu, args.device_id)
    log_value('val_loss', loss, global_step)
    log_value('val_acc', acc, global_step)
    lr_scheduler.step(acc)
    time_elapsed = time.time() - time_start
    lr = optimizer.param_groups[0]['lr']
    log_value('lr', lr)
    print('validate loss: {:.4f} acc: {:.4f} lr: {}'.format(loss, acc, lr))

    # save model
    if epoch % args.num_epoch_per_save == 0 and epoch != 0:
        torch.save(model.state_dict(), args.model_name + str(global_step) + '.state')
        print('Save model at step ', global_step)

    print('Epoch {} finished, using time: {:.0f}m {:.0f}s'.
          format(epoch, time_elapsed // 60, time_elapsed % 60))
