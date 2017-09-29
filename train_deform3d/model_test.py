import argparse
import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from train_deform3d import deform_resnet3d_18
from data_set import dataset
from train_res3d import util

# params
parser = argparse.ArgumentParser()
parser.add_argument('-pre_class_num', default=101)
parser.add_argument('-class_num', default=101)
parser.add_argument('-batch_size', default=4)
parser.add_argument('-clip_length', default=16)
parser.add_argument('-lr', default=0.001)
parser.add_argument('-weight_decay_ratio', default=1e-4)

parser.add_argument('-pre_trained_model', default='../resnet3d_finetuning_18-399-0.93.state')
parser.add_argument('-use_pre_trained_model', default=True)

parser.add_argument('-mean', default=[114 / 1, 123 / 1, 125 / 1])
parser.add_argument('-std', default=[0.229, 0.224, 0.225])
parser.add_argument('-resize_shape', default=[120, 160])
parser.add_argument('-crop_shape', default=[112, 112])
parser.add_argument('-device_id', default=[0])

args = parser.parse_args()

data_dir = '/home/lshi/Database/UCF-101/val/'
data_set = dataset.UCFImageFolder(data_dir, True, args)
data_set_loaders = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
data_set_classes = data_set.classes

# for data, label in data_set_loaders:
#     util.in_batch_show(data, label, data_set_classes, 'input batch')
#     util.in_clip_show(data, label, data_set_classes, 'input clip')

# show input
clip, labels = next(iter(data_set_loaders))
#
util.in_batch_show(clip, labels, data_set_classes, 'input batch')
util.in_clip_show(clip, labels, data_set_classes, 'input clip')

model = deform_resnet3d_18.ResNet3d(args.class_num, args.clip_length, args.crop_shape)
if args.use_pre_trained_model:
    model.fc = torch.nn.Linear(512, args.pre_class_num)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.pre_trained_model)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('----Pretrained model load finished: ', args.pre_trained_model)

if args.pre_class_num != args.class_num:
    model.fc = torch.nn.Linear(512, args.class_num)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_ratio)
loss_function = torch.nn.CrossEntropyLoss(size_average=True)

# model.cpu()
model.cuda()
model.train()
clip, labels = Variable(clip).float().cuda(), Variable(labels).cuda()
outputs, layers = model(clip)
loss = loss_function(outputs, labels)
loss.backward()
optimizer.step()

for i in range(5):
    l = layers[i].cpu().data
    o = outputs.data
    # util.out_channel_show(l, 'channel layer ' + str(i))
    util.out_clip_show(l, 'clip layer ' + str(i))

print('finish')
