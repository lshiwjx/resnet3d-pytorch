import torch
from train_deform3d.deform_resnet3d_18 import DeformResNet3d

model = DeformResNet3d(27, 32, (100, 100))
# p = list(model.parameters())
# pp = dict(model.named_parameters())
params_dict = dict(model.named_parameters())
params = []
for key, value in params_dict.items():
    if key[8:16] == 'conv_off':
        params += [{'params': [value], 'lr': 0.01}]
    else:
        params += [{'params': [value], 'lr': 0.1}]
optimizer = torch.optim.Adam(params, weight_decay=0.0001)

model_dict = model.state_dict()
another_dict = {}
pretrained_dict = torch.load('train_deform3d/deform_jes_l3d1-15418.state')
for x in pretrained_dict.keys():
    if x[6] == '.':
        another_dict[x[:6] + x[7:]] = pretrained_dict[x]
    else:
        another_dict[x] = pretrained_dict[x]
# 1. filter out unnecessary keys
another_dict = {k: v for k, v in another_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(another_dict)
# 3. load the new state dict
model.load_state_dict(model_dict)
torch.save(model_dict, 'train_deform3d/deform_jes_l3d1.state')
