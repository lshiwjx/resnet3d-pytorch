from __future__ import print_function, division

import argparse
import os
import shutil
import time

import torch
from tensorboard_logger import configure, log_value
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data_set import dataset
from model import resnet3d_18
from train_res3d import train_val_model

# params
parser = argparse.ArgumentParser()
parser.add_argument('-pre_class_num', default=400)
parser.add_argument('-class_num', default=27)
parser.add_argument('-batch_size', default=160)
parser.add_argument('-weight_decay_ratio', default=5e-4)
parser.add_argument('-momentum', default=0.9)
parser.add_argument('-max_epoch', default=40)

parser.add_argument('-lr', default=0.001)
parser.add_argument('-lr_decay_ratio', default=0.1)
parser.add_argument('-lr_patience', default=3)
parser.add_argument('-lr_threshold', default=0.02)
parser.add_argument('-lr_delay', default=1)

parser.add_argument('-log_dir', default="./runs/deform_jes_l3d3")
parser.add_argument('-num_epoch_per_save', default=4)
parser.add_argument('-model_saved_name', default='deform_jes_l3d3')

parser.add_argument('-use_last_model_cnn', default=False)
parser.add_argument('-last_model_cnn', default='.state')
parser.add_argument('-use_last_model_rnn', default=False)
parser.add_argument('-last_model_rnn', default='.state')

parser.add_argument('-use_pre_trained_model_cnn', default=True)
parser.add_argument('-pre_trained_model_cnn', default='deform-resnet3d-18.state')
parser.add_argument('-use_pre_trained_model_rnn', default=False)
parser.add_argument('-pre_trained_model_rnn', default='deform-resnet3d-18.state')

parser.add_argument('-only_train_rnn', default=True)

parser.add_argument('-clip_length', default=32)
parser.add_argument('-resize_shape', default=[120, 160])
parser.add_argument('-crop_shape', default=[100, 100])  # must be same for rotate
parser.add_argument('-mean', default=[0.45, 0.43, 0.41])  # cha[124,108,115]ego[114,123,125]ucf[101,97,90]k[]
parser.add_argument('-std', default=[0.23, 0.24, 0.23])

parser.add_argument('-hidden_size', default=1024)

parser.add_argument('-device_id', default=[0, 1, 2, 3])
os.environ['CUDA_VISIBLE_DEVICES'] = '7,6,5,4'
args = parser.parse_args()

# for tensorboard --logdir runs
if os.path.isdir(args.log_dir) and not args.use_last_model_cnn and not args.use_last_model_rnn:
    shutil.rmtree(args.log_dir)
    print('Dir removed: ', args.log_dir)
configure(args.log_dir)

# Date reading, setting for batch size, whether shuffle, num_workers
data_set = {x: dataset.JesterImageFolderLstm(x, args) for x in ['train', 'val']}
data_set_loaders = {x: DataLoader(data_set[x],
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=10,
                                  drop_last=True,
                                  pin_memory=True)
                    for x in ['train', 'val']}

cnnmodel = resnet3d_18.ResNet3dFeature(args.clip_length, args.crop_shape)
rnnmodel = resnet3d_18.Lstm(args.class_num, 512, args.hidden_size, args.batch_size)
if args.use_pre_trained_model_cnn:
    model_dict = cnnmodel.state_dict()
    pretrained_dict = torch.load(args.pre_trained_model_cnn)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    cnnmodel.load_state_dict(model_dict)
    print('----Pretrained cnnmodel load finished: ', args.pre_trained_model_cnn)

if args.use_pre_trained_model_rnn:
    rnnmodel.fc = torch.nn.Linear(args.hidden_size, args.pre_class_num)
    model_dict = rnnmodel.state_dict()
    pretrained_dict = torch.load(args.pre_trained_model_rnn)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    rnnmodel.load_state_dict(model_dict)
    print('----Pretrained rnnmodel load finished: ', args.pre_trained_model_rnn)

if args.pre_class_num != args.class_num:
    rnnmodel.fc = torch.nn.Linear(args.hidden_size, args.class_num)

params_dict = dict(cnnmodel.named_parameters())
params = list(rnnmodel.parameters())
for key, value in params_dict.items():
    if key[8:16] == 'conv_off':
        params += [{'params': [value], 'lr': 0.1 * args.lr}]
    else:
        params += [{'params': [value], 'lr': args.lr}]

if args.only_train_rnn is True:
    for param in cnnmodel.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay_ratio)
    # optimizer = torch.optim.SGD(cnnmodel.fc.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay_ratio)
else:
    print('----Train all params with weight decay: ', args.weight_decay_ratio)
    optimizer = torch.optim.Adam(params, weight_decay=args.weight_decay_ratio)
    # optimizer = torch.optim.SGD(cnnmodel.fc.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay_ratio)

global_step = 0
# The name for model must be **_**-$(step).state
if args.use_last_model_rnn or args.use_last_model_cnn is True:
    if args.use_last_model_cnn is True:
        cnnmodel.load_state_dict(torch.load(args.last_model_cnn))
        global_step = int(args.last_model_cnn[:-6].split('-')[1])
        print('----Training continue, last cnnmodel load finished, step is ', global_step)
    elif args.use_last_model_rnn is True:
        rnnmodel.load_state_dict(torch.load(args.last_model_rnn))
        global_step = int(args.last_model_rnn[:-6].split('-')[1])
        print('----Training continue, last rnnmodel load finished, step is ', global_step)
else:
    print('----Training from scratch, step is ', global_step)

log_value('lr', args.lr, global_step)
use_gpu = torch.cuda.is_available()
print('----Use gpu? ', use_gpu)
if use_gpu:
    cnnmodel = cnnmodel.cuda()
    rnnmodel = rnnmodel.cuda()

loss_function = torch.nn.CrossEntropyLoss()
print('----Using CrossEntropy loss with average')

lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay_ratio,
                                 patience=args.lr_patience, verbose=True,
                                 threshold=args.lr_threshold, threshold_mode='abs',
                                 cooldown=args.lr_delay)
print('----lr scheduler: lr:{} DecayRatio:{} Patience:{} Threshold:{} Before_epoch:{}'
      .format(args.lr, args.lr_decay_ratio, args.lr_patience, args.lr_threshold, args.lr_delay))

print('----Train and val begin, total epoch: ', args.max_epoch)
for epoch in range(args.max_epoch):
    log_value('epoch', epoch, global_step)
    time_start = time.time()
    print('Epoch {}/{}'.format(epoch, args.max_epoch - 1))
    print('Train')
    global_step = train_val_model.train_lstm(cnnmodel, data_set_loaders['train'], loss_function, optimizer,
                                             global_step, use_gpu, args.device_id)
    print('Validate')
    loss, acc = train_val_model.validate_lstm(cnnmodel, data_set_loaders['val'], loss_function, args.batch_size,
                                              use_gpu, args.device_id)
    log_value('val_loss', loss, global_step)
    log_value('val_acc', acc, global_step)

    lr_scheduler.step(acc)
    lr = optimizer.param_groups[0]['lr']
    log_value('lr', lr, global_step)

    time_elapsed = time.time() - time_start
    log_value('time', time_elapsed, global_step)
    print('validate loss: {:.4f} acc: {:.4f} lr: {}'.format(loss, acc, lr))

    # save cnnmodel
    if epoch % args.num_epoch_per_save == 0 and epoch != 0 and not args.only_train_rnn:
        torch.save(cnnmodel.state_dict(), args.model_saved_name + '_cnn-' + str(global_step) + '.state')
        print('Save cnnmodel at step ', global_step)
    if epoch % args.num_epoch_per_save == 0 and epoch != 0:
        torch.save(rnnmodel.state_dict(), args.model_saved_name + '_rnn-' + str(global_step) + '.state')
        print('Save rnnmodel at step ', global_step)
    print('Epoch {} finished, using time: {:.0f}m {:.0f}s'.
          format(epoch, time_elapsed // 60, time_elapsed % 60))
