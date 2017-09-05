import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt

MEAN = [0.485, 0.456, 0.406]  # [101, 97, 90]


# Get a batch of training data
def imshow(inp, title=None):
    # chw -> hwc
    inp = inp.permute(1, 2, 0).numpy()
    inp = inp + MEAN
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(5)  # pause a bit so that plots are updated


def batch_show(data_set_loader, data_set_classes):
    clip, classes = next(iter(data_set_loader['train']))
    # nclhw -> lnchw
    clip = clip.permute(2, 0, 1, 3, 4)
    out = torchvision.utils.make_grid(clip[0][0:4])
    plt.figure('batch data')
    imshow(out, title=[data_set_classes[x] for x in classes])


def clip_show(data_set_loader, data_set_classes):
    clip, classes = next(iter(data_set_loader['train']))
    # nclhw -> nlchw
    clip = clip.permute(0, 2, 1, 3, 4)
    out = torchvision.utils.make_grid(clip[0])
    plt.figure('clip data')
    imshow(out, title=[data_set_classes[x] for x in classes])


def write_class_txt(data_set_classes):
    # store labels
    label_file = open('classes.txt', 'w')
    for item in data_set_classes:
        label_file.write("%s\n" % item)
    label_file.close()
