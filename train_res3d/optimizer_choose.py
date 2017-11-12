from __future__ import print_function, division

import os
import shutil
import time

import torch
from tensorboard_logger import configure, log_value
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train_res3d.parser_args import parser_args
from data_set.data_choose import data_choose
from model.model_choose import model_choose
from train_res3d import train_val_model


def optimizer_choose(model, args):
    params = []
    for key, value in model.named_parameters():
        if key[8:16] == 'conv_off':
            params += [{'params': [value], 'lr': args.deform_lr_ratio * args.lr,
                        'weight_decay': args.weight_decay_ratio * args.wdr}]
            print('lr for {}: {}*{}, wd: {}*{}'.format(key, args.lr, args.deform_lr_ratio, args.weight_decay_ratio,
                                                       args.wdr))
        elif key[0:3] == 'stn':
            params += [{'params': [value], 'lr': args.deform_lr_ratio * args.lr,
                        'weight_decay': args.weight_decay_ratio}]
            print('lr for {}: {}*{}, wd: {}*{}'.format(key, args.lr, args.deform_lr_ratio, args.weight_decay_ratio,
                                                       args.wdr))
        elif key[0:6] == 'deform':
            params += [{'params': [value], 'lr': args.deform_lr_ratio * args.lr,
                        'weight_decay': args.weight_decay_ratio}]
            print('lr for {}: {}*{}, wd: {}*{}'.format(key, args.lr, args.deform_lr_ratio, args.weight_decay_ratio,
                                                       args.wdr))
        elif key[0:4] == 'mask':
            params += [{'params': [value], 'lr': args.deform_lr_ratio * args.lr,
                        'weight_decay': args.weight_decay_ratio}]
            print('lr for {}: {}*{}, wd: {}*{}'.format(key, args.lr, args.deform_lr_ratio, args.weight_decay_ratio,
                                                       args.wdr))
        else:
            if value.requires_grad:
                params += [{'params': [value], 'lr': args.lr}]
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params)
        print('----Using Adam optimizer')
    else:
        optimizer = torch.optim.SGD(params, momentum=args.momentum)
        print('----Using SGD with momentum ', args.momentum)
    return optimizer
