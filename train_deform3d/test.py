from __future__ import print_function, division

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data_set import dataset
from train_res3d import resnet3d_18, train_val_model
import pandas as pd

# params
parser = argparse.ArgumentParser()
parser.add_argument('-class_num', default=27)
parser.add_argument('-batch_size', default=16)
parser.add_argument('-pre_trained_model', default='deform_jes_l3b-15742.state')
parser.add_argument('-clip_length', default=32)
parser.add_argument('-crop_shape', default=[100, 100])  # must be same for rotate
parser.add_argument('-mean', default=[0.45, 0.43, 0.41])  # cha[124,108,115]ego[114,123,125]ucf[101,97,90]k[]
parser.add_argument('-std', default=[0.23, 0.24, 0.23])
parser.add_argument('-device_id', default=[0])
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
args = parser.parse_args()

f = pd.read_csv('/home/lshi/Database/Jester/jester-v1-test.csv', header=None)
data_set = dataset.JesterImageFolderTest(f, args)
data_set_loaders = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, num_workers=10, drop_last=False)

model = resnet3d_18.DeformResNet3d(args.class_num, args.clip_length, args.crop_shape)
model.load_state_dict(torch.load(args.pre_trained_model))
print('Pretrained model load finished: ', args.pre_trained_model)

model = model.cuda()

i = 0
for data in data_set_loaders:
    inputs = data
    inputs = Variable(inputs.cuda(async=True))
    net = torch.nn.DataParallel(model, device_ids=args.device_id)
    outputs, _ = net(inputs.float())
    __, predict_label = torch.max(outputs.data, 1)
    label = list(predict_label)
    for j in label:
        f.loc[i, 1] = int(j)
        # tmp = f.loc[i]
        print('now for ', i, f.loc[i, 0], f.loc[i, 1])
        i += 1

f.to_csv('/home/lshi/Database/Jester/result.csv', header=None, index=None)
