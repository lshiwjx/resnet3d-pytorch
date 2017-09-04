"""
This program is for fine_tuning the Alex_net model for the final fc layer with the data set of 'Caltech256'.

Data set: http://www.vision.caltech.edu/Image_Datasets/Caltech256/

Usage: You should firstly download the data and exact to the directory of fine_tuning.py.
       Make sure the name of data dir is '256_ObjectCategories'. Then run the data.py.
       Finally, run the fine_tuning.py
"""
from __future__ import print_function, division
from resnet3d import resnet18
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.serialization import load_lua
import numpy as np
from tensorboard_logger import configure, log_value
import matplotlib.pyplot as plt
import time
import copy
import torchvision
import pickle
import dataset
from torchvision import transforms
import torch
import os

NUM_EPOCHES = 1
LR = 0.001
MOMENTUM = 0.9
LR_DECAY_EPOCH = 5
CLASS_NUM = 101
BATCH_SIZE = 4
MEAN = [0, 0, 0]  # [101, 97, 90]
STD = [1, 1, 1]
SCALE_SIZE = 240
CROP_SIZE = 112
# for tensorboard
configure("runs/run-1234")


def imshow(inp, title=None):
    """Image show for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(MEAN)
    std = np.array(STD)
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.5)  # pause a bit so that plots are updated


data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(SCALE_SIZE),
        transforms.RandomCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # mean std
        transforms.Normalize(MEAN, STD)
    ]),
    'val': transforms.Compose([
        transforms.Scale(SCALE_SIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]),
}

# Date reading, setting for batch size, whether shuffle, num_workers
data_dir = '/home/sl/Resource/UCF/'
data_set = {x: dataset.UCFImageFolder(os.path.join(data_dir, x), (x is 'train')) for x in ['train', 'val']}
data_set_loaders = {x: torch.utils.data.DataLoader(data_set[x], batch_size=BATCH_SIZE, shuffle=True,
                                                   num_workers=0, drop_last=True)
                    for x in ['train', 'val']}
data_set_sizes = {x: len(data_set[x]) for x in ['train', 'val']}
data_set_classes = data_set['train'].classes

# Get a batch of training data and show
# clip, classes = next(iter(data_set_loaders['train']))[0]
# out = torchvision.utils.make_grid(clip[0])
# plt.figure('train data')
# imshow(out, title=[data_set_classes[x] for x in classes])

use_gpu = torch.cuda.is_available()


def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=NUM_EPOCHES):
    since = time.time()
    best_model = model
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            step = 0
            for data in data_set_loaders[phase]:
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if use_gpu:
                    net = torch.nn.DataParallel(model, device_ids=[0])
                    outputs = net(inputs.float())
                else:
                    outputs = model(inputs.float())
                _, preds = torch.max(outputs.data, 1)  # value and index
                loss = criterion(outputs, labels)

                # backward + update only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                ls = loss.data[0]
                acc = torch.sum(preds == labels.data)
                running_loss += ls
                running_corrects += acc

                # statistics
                if phase == 'train':
                    step += 1
                    log_value('train_loss', ls, step)
                    log_value('train_acc', acc, step)
                    print('train step: {}, loss {}, acc {}'.format(step, ls, acc))
                else:
                    log_value('val_loss', ls, step)
                    log_value('val_acc', acc, step)
                    print('val step: {}, loss {}, acc {}'.format(step, ls, acc))

            epoch_loss = running_loss / data_set_sizes[phase]
            epoch_acc = running_corrects / data_set_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model


def exp_lr_scheduler(optimizer, epoch, init_lr=LR, lr_decay_epoch=LR_DECAY_EPOCH):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.5 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


model_conv = resnet18(True)
for param in model_conv.parameters():
    param.requires_grad = False

model_conv.fc = nn.Linear(512, CLASS_NUM)

# continue training
# model_conv.load_state_dict(torch.load('resnet3d_finetuning_18.state'))

if use_gpu:
    model_conv = model_conv.cuda()
entropy_loss = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=LR, momentum=MOMENTUM)
model_conv = train_model(model_conv, entropy_loss, optimizer_conv, exp_lr_scheduler, num_epochs=NUM_EPOCHES)

torch.save(model_conv.state_dict(), 'resnet3d_finetuning_18.state')

# store labels
label_file = open('classes.txt', 'w')
for item in data_set_classes:
    label_file.write("%s\n" % item)
label_file.close()
