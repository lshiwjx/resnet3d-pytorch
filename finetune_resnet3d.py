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
import torchvision
import dataset
import torch
import os
import shutil

NUM_EPOCHES = 20
LR = 0.001
MOMENTUM = 0.9
LR_DECAY_EPOCH = 10
CLASS_NUM = 101
BATCH_SIZE = 256
EPOCH_SAVE = 2
DEVICES = [0, 1, 2, 3, 4, 5, 6, 7]
LOG_DIR = "runs/test"
# for tensorboard --logdir runs
if os.path.isdir(LOG_DIR):
    shutil.rmtree(LOG_DIR)
    print('dir removed: ', LOG_DIR)
configure(LOG_DIR)

# Date reading, setting for batch size, whether shuffle, num_workers
data_dir = '/home/lshi/Database/UCF-101/'
data_set = {x: dataset.UCFImageFolder(os.path.join(data_dir, x), (x is 'train')) for x in ['train', 'val']}
data_set_loaders = {x: torch.utils.data.DataLoader(data_set[x], batch_size=BATCH_SIZE, shuffle=True,
                                                   num_workers=50, drop_last=False)
                    for x in ['train', 'val']}
data_set_sizes = {x: len(data_set[x]) for x in ['train', 'val']}
data_set_classes = data_set['train'].classes

use_gpu = torch.cuda.is_available()


# Get a batch of training data
def imshow(inp, title=None):
    """Image show for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    inp = inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(5)  # pause a bit so that plots are updated


def batch_show(data_set_loader):
    clip, classes = next(iter(data_set_loader['train']))
    clip = clip.numpy()
    img = np.transpose(clip, (2, 0, 1, 3, 4))[0]
    out = torchvision.utils.make_grid(torch.from_numpy(img[0:4]))
    plt.figure('batch data')
    imshow(out, title=[data_set_classes[x] for x in classes])


def clip_show(data_set_loader):
    clip, classes = next(iter(data_set_loader['train']))
    clip = clip.numpy()[0]
    img = np.transpose(clip, (1, 0, 2, 3))
    out = torchvision.utils.make_grid(torch.from_numpy(img))
    plt.figure('clip data')
    imshow(out, title=[data_set_classes[x] for x in classes])


batch_show(data_set_loaders)
clip_show(data_set_loaders)


def train_val_model(model, criterion, optimizer, lr_scheduler, num_epochs=NUM_EPOCHES):
    step = 0
    print('-' * 10)
    for epoch in range(num_epochs):
        since = time.time()
        # Each epoch has a training and validation phase
        val_loss = 0.0
        val_acc = 0.0
        val_step = 0
        for phase in ['train', 'val']:
            print('Epoch {}/{}, phase: {} use_gpu: {}'.
                  format(epoch, num_epochs - 1, phase, use_gpu))

            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

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
                    net = torch.nn.DataParallel(model, device_ids=DEVICES)
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
                acc = torch.sum(preds == labels.data) / BATCH_SIZE

                # statistics
                if phase == 'train':
                    step += 1
                    log_value('train_loss', ls, step)
                    log_value('train_acc', acc, step)
                    print('train step: {}, loss {:.4f}, acc {:.4f}'.format(step, ls, acc))
                else:
                    val_step += 1
                    val_loss += ls
                    val_acc += acc
                    print('val step: {}, loss {:.4f}, acc {:.4f}'.format(val_step, ls, acc))

        val_loss /= val_step
        val_acc /= val_step
        log_value('val_loss', val_loss, step)
        log_value('val_acc', val_acc, step)
        time_elapsed = time.time() - since
        print('\nEpoch {} finished, using time: {:.0f}m {:.0f}s, loss: {:.4f} acc: {:.4f} \n'.
              format(epoch, time_elapsed // 60, time_elapsed % 60, val_loss, val_acc))
        if epoch % EPOCH_SAVE == 0 and epoch != 0:
            torch.save(model_conv.state_dict(), 'resnet3d_finetuning_18_' + str(epoch) + '.state')


def exp_lr_scheduler(optimizer, epoch, init_lr=LR, lr_decay_epoch=LR_DECAY_EPOCH):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))
        log_value('lr', lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


model_conv = resnet18(True)
# for param in model_conv.parameters():
#     param.requires_grad = False

model_conv.fc = nn.Linear(512, CLASS_NUM)

# continue training
model_conv.load_state_dict(torch.load('resnet3d_finetuning_18_6.state'))

if use_gpu:
    model_conv = model_conv.cuda()
entropy_loss = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.parameters(), lr=LR, momentum=MOMENTUM)

train_val_model(model_conv, entropy_loss, optimizer_conv, exp_lr_scheduler, num_epochs=NUM_EPOCHES)

# store labels
label_file = open('classes.txt', 'w')
for item in data_set_classes:
    label_file.write("%s\n" % item)
label_file.close()
