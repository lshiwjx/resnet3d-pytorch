import torch
from train_res3d import util
from train_res3d.parser_args import parser_args, parser_cifar_args
from data_set.data_choose import data_choose
from model.model_choose import model_choose
from torch.autograd import Variable
import numpy as np
import torchvision
# params
args = parser_args()
# args = parser_cifar_args()

data_set_loaders = data_choose(args)
data_set_classes = [x for x in range(args.class_num)]

# for data, label in data_set_loaders:
#     util.in_batch_show(data, label, data_set_classes, 'input batch')
#     util.in_clip_show(data, label, data_set_classes, 'input clip')

# show input
clip, labels = next(iter(data_set_loaders))

util.in_batch_show(clip, labels, data_set_classes, 'input batch')
util.in_clip_show(clip, labels, data_set_classes, 'input clip')
# tmp = torchvision.utils.make_grid(clip, normalize=True)
# tmp = tmp.permute(1,2,0).numpy()
# util.single_show(tmp, 'input')
_, model = model_choose(args)
model.load_state_dict(torch.load(args.last_model))

# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_ratio)

# loss_function = torch.nn.CrossEntropyLoss(size_average=True)

model.cuda()
model.train()
clip, labels = Variable(clip).float().cuda(), Variable(labels).cuda()
outputs, layers = model(clip)
off = layers[5].cpu().data.numpy()
print(np.mean(off, (0, 1, 2)))
# off = layers[2].cpu().data.numpy()
# print(np.mean(off, (0, 1)))
# torch.save(off, 'off_tensor')
# for i in range(4):
#     for j in range(6):
#         tmp = layers[0].cpu().data[i, j].numpy()
#         util.single_show(tmp, '1')
#         tmp = layers[1].cpu().data[i, j].numpy()
#         util.single_show(tmp, '2')
# tmp = layers[2].cpu().data[i, j].numpy()
# util.single_show(tmp, '3')
# tmp = layers[3].cpu().data[i, j].numpy()
# util.single_show(tmp, '4')
# pass
# tmp = layers[2].cpu().data.numpy()
# loss = loss_function(outputs, labels)
# loss.backward()
# optimizer.step()
# while True:
#     pass
for i in range(5):
    l = layers[i].cpu().data
    o = outputs.cpu().data
    #     util.out_channel_show(l, 'channel layer ' + str(i))
    util.out_clip_show(l, 'clip layer ' + str(i))

channel = 64
t = layers[5]
d000 = t.cpu().data[0, :, 0, 0, 0]
f = t.cpu().data.numpy()
fm = np.max(f)
fmi = np.min(f)
a = np.reshape(f, (args.batch_size, channel, 3, 3, 3, 3, 4, 7, 7))
for i in range(4):
    for j in range(7):
        for k in range(7):
            print(i, j, k, '-', np.mean(a[0, :, 0, :, :, :, i, j, k]))
            print(i, j, k, '-', np.mean(a[0, :, 1, :, :, :, i, j, k]))
            print(i, j, k, '-', np.mean(a[0, :, 2, :, :, :, i, j, k]))

for i in range(channel):
    # print(str(i) + ': ')
    # print(np.mean(a[i, 0, :, :, :]))
    # print('')
    # print(np.mean(a[i, 1, :, :, :]))
    # print('')
    # print(np.mean(a[i, 2, :, :, :]))
    # print('')
    print(str(i) + ': ')
    print(a[i, 0, :, :, :])
    print('')
    print(a[i, 1, :, :, :])
    print('')
    print(a[i, 2, :, :, :])
    print('')

print('finish')
