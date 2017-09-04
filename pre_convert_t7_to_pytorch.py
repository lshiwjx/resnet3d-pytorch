import torch
from resnet3d import resnet18
import torchvision
from torch.utils.serialization import load_lua

torchvision.models.resnet18()
model = resnet18()
model_t7 = load_lua('resnet-18-kinetics-cpu.t7')
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
copy_param(model.layer1[0].conv1, model_t7.modules[0].modules[4]
           .modules[0].modules[0].modules[0].modules[0])
copy_param(model.layer1[0].bn1, model_t7.modules[0].modules[4]
           .modules[0].modules[0].modules[0].modules[1])
copy_param(model.layer1[0].conv2, model_t7.modules[0].modules[4]
           .modules[0].modules[0].modules[0].modules[3])
copy_param(model.layer1[0].bn2, model_t7.modules[0].modules[4]
           .modules[0].modules[0].modules[0].modules[4])

copy_param(model.layer1[1].conv1, model_t7.modules[0].modules[4]
           .modules[1].modules[0].modules[0].modules[0])
copy_param(model.layer1[1].bn1, model_t7.modules[0].modules[4]
           .modules[1].modules[0].modules[0].modules[1])
copy_param(model.layer1[1].conv2, model_t7.modules[0].modules[4]
           .modules[1].modules[0].modules[0].modules[3])
copy_param(model.layer1[1].bn2, model_t7.modules[0].modules[4]
           .modules[1].modules[0].modules[0].modules[4])
# layer2
copy_param(model.layer2[0].conv1, model_t7.modules[0].modules[5]
           .modules[0].modules[0].modules[0].modules[0])
copy_param(model.layer2[0].bn1, model_t7.modules[0].modules[5]
           .modules[0].modules[0].modules[0].modules[1])
copy_param(model.layer2[0].conv2, model_t7.modules[0].modules[5]
           .modules[0].modules[0].modules[0].modules[3])
copy_param(model.layer2[0].bn2, model_t7.modules[0].modules[5]
           .modules[0].modules[0].modules[0].modules[4])

copy_param(model.layer2[1].conv1, model_t7.modules[0].modules[5]
           .modules[1].modules[0].modules[0].modules[0])
copy_param(model.layer2[1].bn1, model_t7.modules[0].modules[5]
           .modules[1].modules[0].modules[0].modules[1])
copy_param(model.layer2[1].conv2, model_t7.modules[0].modules[5]
           .modules[1].modules[0].modules[0].modules[3])
copy_param(model.layer2[1].bn2, model_t7.modules[0].modules[5]
           .modules[1].modules[0].modules[0].modules[4])

# layer3
copy_param(model.layer3[0].conv1, model_t7.modules[0].modules[6]
           .modules[0].modules[0].modules[0].modules[0])
copy_param(model.layer3[0].bn1, model_t7.modules[0].modules[6]
           .modules[0].modules[0].modules[0].modules[1])
copy_param(model.layer3[0].conv2, model_t7.modules[0].modules[6]
           .modules[0].modules[0].modules[0].modules[3])
copy_param(model.layer3[0].bn2, model_t7.modules[0].modules[6]
           .modules[0].modules[0].modules[0].modules[4])

copy_param(model.layer3[1].conv1, model_t7.modules[0].modules[6]
           .modules[1].modules[0].modules[0].modules[0])
copy_param(model.layer3[1].bn1, model_t7.modules[0].modules[6]
           .modules[1].modules[0].modules[0].modules[1])
copy_param(model.layer3[1].conv2, model_t7.modules[0].modules[6]
           .modules[1].modules[0].modules[0].modules[3])
copy_param(model.layer3[1].bn2, model_t7.modules[0].modules[6]
           .modules[1].modules[0].modules[0].modules[4])

# layer4
copy_param(model.layer4[0].conv1, model_t7.modules[0].modules[7]
           .modules[0].modules[0].modules[0].modules[0])
copy_param(model.layer4[0].bn1, model_t7.modules[0].modules[7]
           .modules[0].modules[0].modules[0].modules[1])
copy_param(model.layer4[0].conv2, model_t7.modules[0].modules[7]
           .modules[0].modules[0].modules[0].modules[3])
copy_param(model.layer4[0].bn2, model_t7.modules[0].modules[7]
           .modules[0].modules[0].modules[0].modules[4])

copy_param(model.layer4[1].conv1, model_t7.modules[0].modules[7]
           .modules[1].modules[0].modules[0].modules[0])
copy_param(model.layer4[1].bn1, model_t7.modules[0].modules[7]
           .modules[1].modules[0].modules[0].modules[1])
copy_param(model.layer4[1].conv2, model_t7.modules[0].modules[7]
           .modules[1].modules[0].modules[0].modules[3])
copy_param(model.layer4[1].bn2, model_t7.modules[0].modules[7]
           .modules[1].modules[0].modules[0].modules[4])

# linear
copy_param(model.fc, model_t7.modules[0].modules[10])

torch.save(model.state_dict(), 'resnet3d-18.state')
