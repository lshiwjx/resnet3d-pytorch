import torch
from resnet3d import resnet34
import torchvision
from torch.utils.serialization import load_lua

torchvision.models.resnet34()
model = resnet34()
model_t7 = load_lua('resnet-34-kinetics-cpu.t7')
# dummy inputs for the example
print('load success')


def copy_param(n, m):
    if m.weight is not None: n.weight.data.copy_(m.weight)
    if m.bias is not None: n.bias.data.copy_(m.bias)
    if hasattr(n, 'running_mean'): n.running_mean.copy_(m.running_mean)
    if hasattr(n, 'running_var'): n.running_var.copy_(m.running_var)


copy_param(model.conv1, model_t7.modules[0].modules[0])
copy_param(model.bn1, model_t7.modules[0].modules[1])


# layer1
def layer1(n):
    for i in range(n):
        for j in range(4):
            copy_param(model.layer1[0].conv1, model_t7.modules[0].modules[4]
                       .modules[i].modules[0].modules[0].modules[j])


# 2
def layer2(n):
    for i in range(n):
        for j in range(4):
            copy_param(model.layer2[0].conv1, model_t7.modules[0].modules[5]
                       .modules[i].modules[0].modules[0].modules[j])


# 3
def layer3(n):
    for i in range(n):
        for j in range(4):
            copy_param(model.layer3[0].conv1, model_t7.modules[0].modules[6]
                       .modules[i].modules[0].modules[0].modules[j])


# layer4
def layer4(n):
    for i in range(n):
        for j in range(4):
            copy_param(model.layer1[0].conv1, model_t7.modules[0].modules[7]
                       .modules[i].modules[0].modules[0].modules[j])


layer1(3)
layer2(4)
layer3(6)
layer4(3)

# linear
copy_param(model.fc, model_t7.modules[0].modules[10])

torch.save(model.state_dict(), 'resnet3d-34.state')
