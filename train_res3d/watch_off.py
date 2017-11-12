"""
    visualize tensor saved by model test
"""
import torch
from train_res3d import util
import numpy as np

batch_size = 4
# N GH'W'2
channel = 108  # 4 6*3*3*2 12 12
t = torch.load('off_tensor')
a = np.reshape(t, (batch_size, 6, 3, 3, 2, 12, 12))
for i in range(2):
    print(i, '-', np.mean(a[0, :, :, :, i, :, :]))
    print(i, '-', np.mean(a[1, :, :, :, i, :, :]))
    print(i, '-', np.mean(a[2, :, :, :, i, :, :]))
    print(i, '-', np.mean(a[3, :, :, :, i, :, :]))
    for j in range(12):
        # print(i, j, '-', np.mean(a[0, :, :, :, i, j, :]))
        for k in range(12):
            # print(i, j, k, '-', np.mean(a[0, :, :, :, i, j, k]))
            pass

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
