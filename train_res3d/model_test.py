import argparse
import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import resnet3d_test
from data_set import dataset
from test import util

# params
parser = argparse.ArgumentParser()
parser.add_argument('-pre_class_num', default=101)
parser.add_argument('-class_num', default=249)
parser.add_argument('-batch_size', default=4)
parser.add_argument('-device_id', default=[0])
parser.add_argument('-last_model', default='resnet3d_finetuning_18-6142.state')
parser.add_argument('-pre_trained_model', default='resnet3d_finetuning_18-399-0.93.state')
parser.add_argument('-use_pre_trained_model', default=False)
parser.add_argument('-clip_length', default=16)
parser.add_argument('-aug_ratio', default=2)
parser.add_argument('-mean', default=[114 / 1, 123 / 1, 125 / 1])
parser.add_argument('-std', default=[0.229, 0.224, 0.225])
parser.add_argument('-resize_shape', default=[120, 160])
parser.add_argument('-crop_shape', default=[112, 112])
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
data_dir = '/home/lshi/Database/ChaLearn/val/'
data_set = dataset.CHAImageFolderPillow(data_dir, True, args)
data_set_loaders = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
data_set_classes = data_set.classes

# for data, label in data_set_loaders:
#     util.in_batch_show(data, label, data_set_classes, 'input batch')
#     util.in_clip_show(data, label, data_set_classes, 'input clip')

# show input
clip, classes = next(iter(data_set_loaders))

util.in_batch_show(clip, classes, data_set_classes, 'input batch')
util.in_clip_show(clip, classes, data_set_classes, 'input clip')

model = resnet3d_test.resnet18(args.pre_class_num, args.clip_length, args.crop_shape)
model.load_state_dict(torch.load(args.pre_trained_model))
print('Pretrained model load finished: ', args.pre_trained_model)
# model.fc = torch.nn.Linear(512, args.class_num)
# model.load_state_dict(torch.load(args.last_model))
model.cpu()

outputs, layers = model.forward(Variable(clip).float())
for i in range(5):
    l = layers[i].data
    o = outputs.data
    # util.out_channel_show(l, 'channel layer ' + str(i))
    util.out_clip_show(l, 'clip layer ' + str(i))

print('finish')
