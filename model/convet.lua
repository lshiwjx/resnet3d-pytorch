torch = require 'torch'
require 'nn'
require 'cunn'
cudnn = require 'cudnn'

m = torch.load('resnet-34-kinetics.t7')
print('----------------------------------before-----------------------------------')
print(m)
cudnn.convert(m, nn)
m = m:float()
print('------------------------------------after------------------------------------')
print(m)
torch.save('resnet-34-kinetics-cpu.t7', m)
