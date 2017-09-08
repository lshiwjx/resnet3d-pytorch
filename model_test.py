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
parser.add_argument('-lr', default=0.0001)
parser.add_argument('-class_num', default=101)
parser.add_argument('-batch_size', default=4)
parser.add_argument('-device_id', default=[0, 1, 2, 3])
parser.add_argument('-last_model', default='resnet3d_finetuning_34-972.state')
parser.add_argument('-clip_length', default=16)
parser.add_argument('-mean', default=[0.485, 0.456, 0.406])
parser.add_argument('-resize_shape', default=[120, 160])
parser.add_argument('-crop_shape', default=[112, 112])
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
data_dir = '/home/lshi/Database/UCF-101/val/'
data_set = dataset.UCFImageFolder(data_dir, False, args)
data_set_loaders = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True,
                              pin_memory=False)
data_set_classes = data_set.classes

# show input
clip, classes = next(iter(data_set_loaders))

# util.in_batch_show(clip, classes, data_set_classes, 'input batch')
# util.in_clip_show(clip, classes, data_set_classes, 'input clip')

model = resnet3d_test.resnet34(pretrained=True)
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
