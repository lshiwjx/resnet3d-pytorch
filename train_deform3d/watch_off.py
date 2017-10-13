"""
    visualize tensor saved by model test
"""
import torch
from train_res3d import util
import numpy as np

channel = 128
t = torch.load('off_tensor_b32')
d000 = t.cpu().data[0, :, 0, 0, 0]
f = t.cpu().data.numpy()
fm = np.max(f)
fmi = np.min(f)
a = d000.numpy()
a = np.reshape(a, (channel, 3, 3, 3, 3))
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

# b = d000.contiguous().view(channel, 3, 3, 3, 3)
# util.off_show(b[0][0], 'off1')
# util.off_show(b[0][1], 'off2')
# util.off_show(b[0][2], 'off3')
print('none')
