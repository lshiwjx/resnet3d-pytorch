import torch
import resnet3d_test
import dataset
import util
import train_val_model
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import argparse

# params
parser = argparse.ArgumentParser()
parser.add_argument('-class_num', default=83)
parser.add_argument('-batch_size', default=4)
parser.add_argument('-device_id', default=[0])
parser.add_argument('-last_model', default='resnet3d_finetuning_18-6142.state')
parser.add_argument('-pre_trained_model', default='resnet3d_finetuning_18-6103.state')
parser.add_argument('-use_pre_trained_model', default=False)
parser.add_argument('-clip_length', default=32)
parser.add_argument('-aug_ratio', default=2)
parser.add_argument('-mean', default=[114 / 1, 123 / 1, 125 / 1])
parser.add_argument('-std', default=[0.229, 0.224, 0.225])
parser.add_argument('-resize_shape', default=[240, 320])
parser.add_argument('-crop_shape', default=[224, 224])
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
data_dir = '/home/lshi/Database/Ego_gesture/val/'
data_set = dataset.EGOImageFolderPillow(data_dir, True, args)
data_set_loaders = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True,
                              pin_memory=False)
data_set_classes = data_set.classes

for data, label in data_set_loaders:
    util.in_batch_show(data, label, data_set_classes, 'input batch')
    util.in_clip_show(data, label, data_set_classes, 'input clip')

# show input
clip, classes = next(iter(data_set_loaders))

util.in_batch_show(clip, classes, data_set_classes, 'input batch')
util.in_clip_show(clip, classes, data_set_classes, 'input clip')

model = resnet3d_test.resnet18(args.class_num, args.clip_length, args.crop_shape)
model.load_state_dict(torch.load('resnet3d_finetuning_18-6142.state'))
print('Pretrained model load finished: resnet3d_finetuning_18-6142.state')
# model = resnet3d_test.resnet34(pretrained=True)
model.fc = torch.nn.Linear(512, args.class_num)
model.load_state_dict(torch.load(args.last_model))
model.cpu()

outputs, layers = model.forward(Variable(clip).float())
for i in range(5):
    l = layers[i].data
    o = outputs.data
    # util.out_channel_show(l, 'channel layer ' + str(i))
    util.out_clip_show(l, 'clip layer ' + str(i))

print('finish')
