import torch
import torchvision
from torch.utils.serialization import load_lua

from train_res3d.resnet3d import resnet34

torchvision.models.resnet34()
model = resnet34()
model_t7 = load_lua('./model/resnet-34-kinetics-cpu.t7')
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
        copy_param(model.layer1[i].conv1, model_t7.modules[0].modules[4]
                   .modules[i].modules[0].modules[0].modules[0])
        copy_param(model.layer1[i].bn1, model_t7.modules[0].modules[4]
                   .modules[i].modules[0].modules[0].modules[1])
        copy_param(model.layer1[i].conv2, model_t7.modules[0].modules[4]
                   .modules[i].modules[0].modules[0].modules[3])
        copy_param(model.layer1[i].bn2, model_t7.modules[0].modules[4]
                   .modules[i].modules[0].modules[0].modules[4])


# 2
def layer2(n):
    for i in range(n):
        copy_param(model.layer2[i].conv1, model_t7.modules[0].modules[5]
                   .modules[i].modules[0].modules[0].modules[0])
        copy_param(model.layer2[i].bn1, model_t7.modules[0].modules[5]
                   .modules[i].modules[0].modules[0].modules[1])
        copy_param(model.layer2[i].conv2, model_t7.modules[0].modules[5]
                   .modules[i].modules[0].modules[0].modules[3])
        copy_param(model.layer2[i].bn2, model_t7.modules[0].modules[5]
                   .modules[i].modules[0].modules[0].modules[4])


# 3
def layer3(n):
    for i in range(n):
        copy_param(model.layer3[i].conv1, model_t7.modules[0].modules[6]
                   .modules[i].modules[0].modules[0].modules[0])
        copy_param(model.layer3[i].bn1, model_t7.modules[0].modules[6]
                   .modules[i].modules[0].modules[0].modules[1])
        copy_param(model.layer3[i].conv2, model_t7.modules[0].modules[6]
                   .modules[i].modules[0].modules[0].modules[3])
        copy_param(model.layer3[i].bn2, model_t7.modules[0].modules[6]
                   .modules[i].modules[0].modules[0].modules[4])


# layer4
def layer4(n):
    for i in range(n):
        copy_param(model.layer4[i].conv1, model_t7.modules[0].modules[7]
                   .modules[i].modules[0].modules[0].modules[0])
        copy_param(model.layer4[i].bn1, model_t7.modules[0].modules[7]
                   .modules[i].modules[0].modules[0].modules[1])
        copy_param(model.layer4[i].conv2, model_t7.modules[0].modules[7]
                   .modules[i].modules[0].modules[0].modules[3])
        copy_param(model.layer4[i].bn2, model_t7.modules[0].modules[7]
                   .modules[i].modules[0].modules[0].modules[4])


layer1(3)
layer2(4)
layer3(6)
layer4(3)

# linear
copy_param(model.fc, model_t7.modules[0].modules[10])

torch.save(model.state_dict(), './model/resnet3d-34.state')
