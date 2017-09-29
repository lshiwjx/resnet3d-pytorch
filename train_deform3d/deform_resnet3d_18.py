import torch.nn as nn
import torch
import math
from train_deform3d.modules import ConvOffset3d


class BasicBlock(nn.Module):
    def __init__(self, channel):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(channel)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class DownsampleBlock(nn.Module):
    def __init__(self, channel_in, channel):
        super(DownsampleBlock, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, channel, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(channel)
        self.downsample = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += torch.cat((residual, residual), 1)
        out = self.relu(out)

        return out


class DeformDownsampleBlock(nn.Module):
    def __init__(self, channel_in, channel, channel_per_group):
        super(DeformDownsampleBlock, self).__init__()
        self.conv_off = nn.Conv3d(channel_in, channel_in // channel_per_group * 3 * 27, kernel_size=3, stride=2,
                                  padding=1, bias=True)
        self.conv1 = ConvOffset3d(channel_in, channel, kernel_size=3, stride=2, padding=1,
                                  channel_per_group=channel_per_group)
        self.bn1 = nn.BatchNorm3d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(channel)
        self.downsample = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        residual = self.downsample(x)

        off = self.conv_off(x)
        out = self.conv1(x, off)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += torch.cat((residual, residual), 1)
        out = self.relu(out)

        return out


class ResNet3d(nn.Module):
    def __init__(self, num_classes, clip_length, crop_shape):
        super(ResNet3d, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=True)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(BasicBlock(64), BasicBlock(64))
        self.layer2 = nn.Sequential(DeformDownsampleBlock(64, 128, 8), BasicBlock(128))
        self.layer3 = nn.Sequential(DownsampleBlock(128, 256), BasicBlock(256))

        self.layer4 = nn.Sequential(DownsampleBlock(256, 512), BasicBlock(512))

        self.avgpool = nn.AvgPool3d(
            (math.ceil(clip_length // 16), math.ceil(crop_shape[0] / 32), math.ceil(crop_shape[1] / 32)))
        self.fc = nn.Linear(512, num_classes)
        self.layers = []

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        self.layers.append(x)
        x = self.layer1(x)
        self.layers.append(x)
        x = self.layer2(x)
        self.layers.append(x)
        x = self.layer3(x)
        self.layers.append(x)

        x = self.layer4(x)
        self.layers.append(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, self.layers
