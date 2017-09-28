import torch
from train_deform3d.deform_resnet3d_18 import ResNet3d

model = ResNet3d(101, 16, (112, 112))
model_dict = model.state_dict()

pretrained_dict = torch.load('resnet3d_finetuning_18-399-0.93.state')
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# 3. load the new state dict
model.load_state_dict(model_dict)


print(" ")

