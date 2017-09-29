import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt

def img_show(inp, title=None):
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.5)  # pause a bit so that plots are updated


def in_batch_show(clip, classes, data_set_classes, name):
    # nclhw -> lnchw
    clip = clip.permute(2, 0, 1, 3, 4)[:, :, 0:3, :, :]
    out = torchvision.utils.make_grid(clip[0][0:4], normalize=True)
    plt.figure(name)
    # chw -> hwc
    out = out.permute(1, 2, 0).numpy()
    img_show(out, title=[data_set_classes[x] for x in classes])


def in_clip_show(clip, classes, data_set_classes, name):
    # nclhw -> nlchw
    clip = clip.permute(0, 2, 1, 3, 4)[:, :, 0:3, :, :]
    out = torchvision.utils.make_grid(clip[0], normalize=True)
    plt.figure(name)
    # chw -> hwc
    out = out.permute(1, 2, 0).numpy()
    img_show(out, title=[data_set_classes[x] for x in classes])


def out_channel_show(clip, name):
    # nclhw -> clhw
    clip = clip[0]
    # clhw -> lchw
    clip = clip.permute(1, 0, 2, 3)
    # lchw -> c1hw
    clip = clip[0].unsqueeze(1)
    out = torchvision.utils.make_grid(clip[0:4], normalize=True)
    plt.figure(name)
    # chw -> hwc
    out = out.permute(1, 2, 0).numpy()
    img_show(out)


def out_clip_show(clip, name):
    # nclhw -> lhw
    clip = clip[0][0]
    # lhw -> l1hw
    clip = clip.unsqueeze(1)
    out = torchvision.utils.make_grid(clip, normalize=True)
    plt.figure(name)
    # chw -> hwc
    out = out.permute(1, 2, 0).numpy()
    img_show(out)


def write_class_txt(data_set_classes):
    # store labels
    label_file = open('classes.txt', 'w')
    for item in data_set_classes:
        label_file.write("%s\n" % item)
    label_file.close()
