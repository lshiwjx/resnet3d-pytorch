import argparse
import os

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from train_res3d import resnet3d_18
from data_set import dataset
from train_res3d import util
import numpy as np

# params
parser = argparse.ArgumentParser()
parser.add_argument('-pre_class_num', default=27)
parser.add_argument('-class_num', default=27)
parser.add_argument('-batch_size', default=2)
parser.add_argument('-clip_length', default=16)
parser.add_argument('-overlap', default=8)
parser.add_argument('-lr', default=0.001)
parser.add_argument('-weight_decay_ratio', default=5e-4)

parser.add_argument('-pre_trained_model', default='deform_jes_l3d1.state')
parser.add_argument('-use_pre_trained_model', default=False)

parser.add_argument('-mean', default=[0.45, 0.43, 0.41])  # cha[124,108,115]ego[114,123,125]ucf[101,97,90]k[]
parser.add_argument('-std', default=[0.23, 0.24, 0.23])
# parser.add_argument('-mean', default=[114 / 1, 123 / 1, 125 / 1])
# parser.add_argument('-std', default=[0.229, 0.224, 0.225])
parser.add_argument('-crop_shape', default=[100, 100])
parser.add_argument('-device_id', default=[0])

args = parser.parse_args()

# data_dir = '/home/lshi/Database/UCF-101/val/'
data_set = dataset.JesterImageFolderLstm(2, args)
data_set_loaders = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
data_set_classes = data_set.classes

# for data, label in data_set_loaders:
#     util.in_batch_show(data, label, data_set_classes, 'input batch')
#     util.in_clip_show(data, label, data_set_classes, 'input video')

# show input
video, labels = next(iter(data_set_loaders))
#
# util.in_batch_show(video, labels, data_set_classes, 'input batch')
# util.in_clip_show(video, labels, data_set_classes, 'input video')

cnn = resnet3d_18.ResNet3dFeature(args.clip_length, args.crop_shape)
rnn = resnet3d_18.Lstm(args.class_num, 512, 1024, args.batch_size)
# if args.use_pre_trained_model:
#     cnn.fc = torch.nn.Linear(512, args.pre_class_num)
#     model_dict = cnn.state_dict()
#     pretrained_dict = torch.load(args.pre_trained_model)
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     cnn.load_state_dict(model_dict)
#     print('----Pretrained cnn load finished: ', args.pre_trained_model)

if args.pre_class_num != args.class_num:
    cnn.fc = torch.nn.Linear(512, args.class_num)
# optimizer = torch.optim.Adam(cnn.parameters(), lr=args.lr, weight_decay=args.weight_decay_ratio)
optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay_ratio)

loss_function = torch.nn.CrossEntropyLoss(size_average=True)

# cnn.cpu()
cnn.cuda()
cnn.train()
rnn.cuda()
rnn.train()
labels = Variable(labels).cuda()
outputs = []
video = torch.transpose(video, 0, 1)
for clip in video:
    clip = Variable(clip).float().cuda()
    output, layers = cnn(clip)
    outputs.append(torch.unsqueeze(output, 0))
outputs = torch.cat(outputs)
# outputs = torch.transpose(outputs, 0, 1)
final, _ = rnn(outputs)
final = torch.mean(final, 0)
loss = loss_function(final, labels)
value, predict_label = torch.max(final.data, 1)
loss.backward()
optimizer.step()
o = final.cpu().data.numpy()


# for i in range(5):
#     l = layers[i].cpu().data
#     o = outputs.data
#     util.out_channel_show(l, 'channel layer ' + str(i))
# util.out_clip_show(l, 'video layer ' + str(i))

# l1 = layers[0].cpu().data[1][0]
# util.off_show(l1, 'off1')
# l2 = layers[0].cpu().data[2][0]
# util.off_show(l2, 'off2')
# a = np.array(l1)
# print(a)


# torch.save(layers[1], 'off_tensor1')
# d000 = layers[1].cpu().data[0, :, 0, 0, 0]
# a = d000.numpy()
# a = np.reshape(a, (32, 3, 3, 3, 3))
# print(a[0, 0, :, :, :])
# print(a[0, 1, :, :, :])
# print(a[0, 2, :, :, :])
# b = layers[0].cpu().data
# util.out_clip_show(b, 'layer')
# print('none')
