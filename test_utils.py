import torch
from train_deform3d.deform_resnet3d_18 import ResNet3d

a = torch.load('resnet3d_finetuning_18-399-0.93.state')
b = a['bn1.bias']

model = ResNet3d(101, 16, (112, 112))
model.load_state_dict(a)
print(" ")
