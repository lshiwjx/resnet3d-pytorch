"""
    visualize tensor saved by model test
"""
import torch
from train_res3d import util
import numpy as np

t = torch.load('off_tensor1')
d000 = t.cpu().data[0, :, 0, 2, 2]
f = t.cpu().data.numpy()
fm = np.max(f)
fmi = np.min(f)
# a = d000.numpy()
# a = np.reshape(a, (64, 3, 3, 3, 3))
# for i in range(64):
#     print(str(i) + ': ')
#     print(a[0, 0, :, :, :])
#     print(a[0, 1, :, :, :])
#     print(a[0, 2, :, :, :])

# b = d000.contiguous().view(64, 3, 3, 3, 3)
# util.off_show(b[0][0], 'off1')
# util.off_show(b[0][1], 'off2')
# util.off_show(b[0][2], 'off3')
print('none')
