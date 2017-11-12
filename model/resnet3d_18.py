import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from model.deform_conv3dl_modules import ConvOffset3d
from torch.nn import functional as f


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


class DeformBasicBlock(nn.Module):
    def __init__(self, channel, channel_per_group):
        super(DeformBasicBlock, self).__init__()
        self.conv_off = nn.Conv3d(channel, channel // channel_per_group, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv1 = ConvOffset3d(channel, channel, kernel_size=3, stride=1, padding=1,
                                  channel_per_group=channel_per_group)
        self.bn1 = nn.BatchNorm3d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(channel)
        self.layers = []

    def forward(self, x):
        residual = x

        off = self.conv_off(x)
        out = self.conv1(x, off)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out, off


class DeformBasicBlock0(nn.Module):
    def __init__(self, channel, channel_per_group):
        super(DeformBasicBlock0, self).__init__()
        self.conv_off1 = nn.Conv3d(channel, channel, kernel_size=5, stride=1,
                                   padding=2, bias=True)
        self.conv_off2 = nn.Conv3d(channel, channel // channel_per_group, kernel_size=1, stride=1,
                                   padding=0, bias=True)
        self.conv1 = ConvOffset3d(channel, channel, kernel_size=3, stride=1, padding=1,
                                  channel_per_group=channel_per_group)
        self.bn1 = nn.BatchNorm3d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(channel)
        self.layers = []

    def forward(self, x):
        residual = x

        off = self.conv_off1(x)
        off = self.conv_off2(off)

        out = self.conv1(x, off)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out, off


class DeformBasicBlock1(nn.Module):
    def __init__(self, channel, channel_per_group):
        super(DeformBasicBlock1, self).__init__()
        self.conv_off1 = nn.Conv3d(channel, channel // channel_per_group * 3 * 27, kernel_size=3, stride=1,
                                   padding=1, bias=True)
        self.conv1 = ConvOffset3d(channel, channel, kernel_size=3, stride=1, padding=1,
                                  channel_per_group=channel_per_group)
        self.bn1 = nn.BatchNorm3d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv_off2 = nn.Conv3d(channel, channel // channel_per_group * 3 * 27, kernel_size=3, stride=1,
                                   padding=1, bias=True)
        self.conv2 = ConvOffset3d(channel, channel, kernel_size=3, stride=1, padding=1,
                                  channel_per_group=channel_per_group)
        self.bn2 = nn.BatchNorm3d(channel)
        self.layers = []

    def forward(self, x):
        self.layers = []
        residual = x

        off = self.conv_off1(x)
        x = self.conv1(x, off)
        # off = self.conv_off1(x)
        # x = self.conv1(x, off)
        x = self.bn1(x)
        x = self.relu(x)

        off = self.conv_off2(x)
        x = self.conv2(x, off)
        # off = self.conv_off2(x)
        # x = self.conv2(x, off)
        x = self.bn2(x)

        x += residual
        x = self.relu(x)

        return x, self.layers


class DeformBasicBlock2(nn.Module):
    def __init__(self, channel, channel_per_group):
        super(DeformBasicBlock2, self).__init__()
        self.conv_off1 = nn.Conv3d(channel, channel // (channel_per_group * 2) * 3 * 27, kernel_size=3, stride=1,
                                   padding=1, bias=True)
        self.conv1 = ConvOffset3d(channel, channel, kernel_size=3, stride=1, padding=1,
                                  channel_per_group=channel_per_group * 2)
        self.bn1 = nn.BatchNorm3d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv_off2 = nn.Conv3d(channel, channel // channel_per_group * 3 * 27, kernel_size=3, stride=1,
                                   padding=1, bias=True)
        self.conv2 = ConvOffset3d(channel, channel, kernel_size=3, stride=1, padding=1,
                                  channel_per_group=channel_per_group)
        self.bn2 = nn.BatchNorm3d(channel)
        self.layers = []

    def forward(self, x):
        self.layers = []
        residual = x

        # off = self.conv_off1(x)
        off = self.conv_off1(x)
        x = self.conv1(x, off)
        x = self.bn1(x)
        x = self.relu(x)

        # off = self.conv_off2(x)
        off = self.conv_off2(x)
        x = self.conv2(x, off)
        x = self.bn2(x)

        x += residual
        x = self.relu(x)

        return x, self.layers


class DeformBasicBlockd(nn.Module):
    def __init__(self, channel, channel_per_group):
        super(DeformBasicBlockd, self).__init__()
        self.conv_off1 = nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_off11 = nn.Conv3d(channel, channel // channel_per_group * 3 * 27, kernel_size=3, stride=1,
                                    padding=1, bias=True)
        self.conv1 = ConvOffset3d(channel, channel, kernel_size=3, stride=1, padding=1,
                                  channel_per_group=channel_per_group)
        self.bn1 = nn.BatchNorm3d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv_off2 = nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_off22 = nn.Conv3d(channel, channel // channel_per_group * 3 * 27, kernel_size=3, stride=1,
                                    padding=1, bias=True)
        self.conv2 = ConvOffset3d(channel, channel, kernel_size=3, stride=1, padding=1,
                                  channel_per_group=channel_per_group)
        self.bn2 = nn.BatchNorm3d(channel)
        self.layers = []

    def forward(self, x):
        self.layers = []
        residual = x

        off = self.conv_off1(x)
        off = self.conv_off11(off)
        x = self.conv1(x, off)
        x = self.bn1(x)
        x = self.relu(x)

        off = self.conv_off2(x)
        off = self.conv_off22(off)
        x = self.conv2(x, off)
        x = self.bn2(x)

        x += residual
        x = self.relu(x)

        return x, self.layers


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


class DownsampleBlockSaveL(nn.Module):
    def __init__(self, channel_in, channel):
        super(DownsampleBlockSaveL, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, channel, kernel_size=3, stride=(1, 2, 2), padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(channel)
        self.downsample = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)

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
        self.layers = []

    def forward(self, x):
        residual = self.downsample(x)

        off = self.conv_off(x)
        self.layers.append(off)
        out = self.conv1(x, off)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += torch.cat((residual, residual), 1)
        out = self.relu(out)

        return out, self.layers


class DeformDownsampleBlock1(nn.Module):
    def __init__(self, channel_in, channel, channel_per_group):
        super(DeformDownsampleBlock1, self).__init__()
        self.conv_off1 = nn.Conv3d(channel_in, channel_in // channel_per_group * 3 * 27, kernel_size=3, stride=2,
                                   padding=1, bias=True)
        self.conv1 = ConvOffset3d(channel_in, channel, kernel_size=3, stride=2, padding=1,
                                  channel_per_group=channel_per_group)
        self.bn1 = nn.BatchNorm3d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv_off2 = nn.Conv3d(channel, channel // channel_per_group * 3 * 27, kernel_size=3, stride=1,
                                   padding=1, bias=True)
        self.conv2 = ConvOffset3d(channel, channel, kernel_size=3, stride=1, padding=1,
                                  channel_per_group=channel_per_group)
        self.bn2 = nn.BatchNorm3d(channel)
        self.downsample = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layers = []

    def forward(self, x):
        self.layers = []
        residual = self.downsample(x)
        # s = x.data.numpy()
        off1 = self.conv_off1(x)
        # self.layers.append(off1)
        out = self.conv1(x, off1)
        out = self.bn1(out)
        out = self.relu(out)

        off2 = self.conv_off2(out)
        # self.layers.append(off2)
        out = self.conv2(out, off2)
        out = self.bn2(out)

        out += torch.cat((residual, residual), 1)
        out = self.relu(out)

        return out


class DeformDownsampleBlock2(nn.Module):
    def __init__(self, channel_in, channel, channel_per_group):
        super(DeformDownsampleBlock2, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, channel, kernel_size=3, stride=2, padding=1,
                               bias=True)
        self.bn1 = nn.BatchNorm3d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv_off2 = nn.Conv3d(channel, channel // channel_per_group * 3 * 27, kernel_size=3, stride=1,
                                   padding=1, bias=True)
        self.conv2 = ConvOffset3d(channel, channel, kernel_size=3, stride=1, padding=1,
                                  channel_per_group=channel_per_group)
        self.bn2 = nn.BatchNorm3d(channel)
        self.downsample = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layers = []

    def forward(self, x):
        self.layers = []
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        off2 = self.conv_off2(out)
        # self.layers.append(off2)
        out = self.conv2(out, off2)
        out = self.bn2(out)

        out += torch.cat((residual, residual), 1)
        out = self.relu(out)

        return out


class ResNet3d(nn.Module):
    def __init__(self, num_classes, clip_length, crop_shape, mode):
        super(ResNet3d, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=True)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
        self.layer10 = BasicBlock(64)
        self.layer11 = BasicBlock(64)
        self.layer20 = DownsampleBlock(64, 128)
        self.layer21 = BasicBlock(128)
        self.layer30 = DownsampleBlock(128, 256)
        self.layer31 = BasicBlock(256)
        self.layer40 = DownsampleBlock(256, 512)
        self.layer41 = BasicBlock(512)

        self.avgpool = nn.AvgPool3d(
            (math.ceil(clip_length // 8), math.ceil(crop_shape[1] / 32), math.ceil(crop_shape[0] / 32)))
        self.fc = nn.Linear(512, num_classes)
        self.layers = []

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_out')
                nn.init.constant(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant(m.weight.data, 1)
                nn.init.constant(m.bias.data, 0)

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # self.layers.append(x)
        x = self.layer10(x)
        x = self.layer11(x)
        # self.layers.append(x)
        x = self.layer20(x)
        x = self.layer21(x)
        # self.layers.append(x)
        x = self.layer30(x)
        x = self.layer31(x)
        # self.layers = y

        x = self.layer40(x)
        x = self.layer41(x)
        # self.layers.append(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, self.layers


class DeformResNet3d(nn.Module):
    def __init__(self, num_classes, clip_length, crop_shape, mode='train'):
        super(DeformResNet3d, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=True)  # 32, 56
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=(1, 2, 2), padding=1)  # 16, 28
        self.layer10 = BasicBlock(64)
        self.layer11 = BasicBlock(64)
        self.layer20 = DownsampleBlock(64, 128)  # 8, 14
        self.layer21 = BasicBlock(128)
        self.layer30 = DownsampleBlock(128, 256)  # 4, 7
        self.layer31 = DeformBasicBlock0(256, 256)
        self.layer40 = DownsampleBlock(256, 512)  # 2, 4
        self.layer41 = BasicBlock(512)

        self.avgpool = nn.AvgPool3d(
            (math.ceil(clip_length // 8), math.ceil(crop_shape[1] / 32), math.ceil(crop_shape[0] / 32)))
        self.fc = nn.Linear(512, num_classes)
        self.layers = []
        self.mode = mode
        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_out')
                nn.init.uniform(m.bias.data, -1e-5, 1e-5)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant(m.weight.data, 1)
                nn.init.constant(m.bias.data, 0)

        for m in self.modules():
            if isinstance(m, DeformBasicBlock0):
                nn.init.uniform(m.conv_off1.weight.data, -1e-5, 1e-5)
                nn.init.uniform(m.conv_off2.weight.data, -1e-5, 1e-5)
                nn.init.uniform(m.conv_off1.bias.data, -1e-5, 1e-5)
                nn.init.uniform(m.conv_off2.bias.data, -1e-5, 1e-5)
                # nn.init.constant(m.conv_off1.weight.data,0)
                # nn.init.constant(m.conv_off2.weight.data, 0)
                # nn.init.constant(m.conv_off1.bias.data, 0)
                # nn.init.constant(m.conv_off2.bias.data, 0)
            elif isinstance(m, DeformBasicBlock):
                nn.init.uniform(m.conv_off1.weight.data, -1e-5, 1e-5)
                nn.init.uniform(m.conv_off1.bias.data, -1e-5, 1e-5)
                nn.init.uniform(m.conv_off2.weight.data, -1e-5, 1e-5)
                nn.init.uniform(m.conv_off2.bias.data, -1e-5, 1e-5)

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.mode == 'test':
            self.layers.append(x)
        x = self.layer10(x)
        x = self.layer11(x)
        if self.mode == 'test':
            self.layers.append(x)
        x = self.layer20(x)
        x = self.layer21(x)

        x = self.layer30(x)
        if self.mode == 'test':
            self.layers.append(x)
        x, y = self.layer31(x)
        if self.mode == 'test':
            self.layers.append(x)

        x = self.layer40(x)
        x = self.layer41(x)
        if self.mode == 'test':
            self.layers.append(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.mode == 'test':
            self.layers.append(y)

        return x, self.layers


class ResNet3dFeature(nn.Module):
    def __init__(self, clip_length, crop_shape):
        super(ResNet3dFeature, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=True)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer10 = BasicBlock(64)
        self.layer11 = BasicBlock(64)
        self.layer20 = DownsampleBlock(64, 128)
        self.layer21 = BasicBlock(128)
        self.layer30 = DownsampleBlock(128, 256)
        self.layer31 = BasicBlock(256)
        self.layer40 = DownsampleBlock(256, 512)
        self.layer41 = BasicBlock(512)
        self.avgpool = nn.AvgPool3d(
            (math.ceil(clip_length // 16), math.ceil(crop_shape[0] / 32), math.ceil(crop_shape[1] / 32)))
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
            elif isinstance(m, ConvOffset3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # self.layers.append(x)
        x = self.layer10(x)
        x = self.layer11(x)
        # self.layers.append(x)
        x = self.layer20(x)
        x = self.layer21(x)
        # self.layers.append(x)
        x = self.layer30(x)
        x = self.layer31(x)
        # self.layers.append(x)

        x = self.layer40(x)
        x = self.layer41(x)
        # self.layers.append(x)
        x = self.avgpool(x)

        # batch_size input_size 4,512,144
        x = x.view(x.size(0), -1)
        return x, self.layers


class Lstm(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, batch_size):
        super(Lstm, self).__init__()
        self.LSTM = nn.LSTM(input_size, hidden_size, 1)
        self.h0 = Variable(torch.zeros(1, batch_size, hidden_size)).cuda()
        self.c0 = Variable(torch.zeros(1, batch_size, hidden_size)).cuda()
        self.fc = nn.Linear(hidden_size, num_classes)
        self.layers = []

    def forward(self, x):
        self.layers = []
        x, _ = self.LSTM(x, (self.h0, self.c0))
        x = self.fc(x)

        return x, self.layers


class Mask(nn.Module):
    def __init__(self, channel, kernel, pad):
        super(Mask, self).__init__()
        self.conv1 = nn.Conv3d(channel, channel, kernel, padding=pad)
        self.conv2 = nn.Conv3d(channel, channel, kernel, padding=pad)

    def forward(self, x):
        mask = self.conv1(x)
        mask = f.relu(mask)
        mask = self.conv2(mask)
        mask = f.relu(mask)
        x = (1 + mask) * x
        return x, mask


class EDMask(nn.Module):
    def __init__(self, channel, kernel, pad):
        super(EDMask, self).__init__()
        self.conv1 = nn.Conv3d(channel, channel, kernel, padding=pad)
        self.conv2 = nn.Conv3d(channel, channel, kernel, padding=pad)
        # self.conv3 = nn.Conv3d(channel, 1, kernel_size=(3,1,1), padding=(1,0,0),stride=(2,1,1))

    def forward(self, x):
        mask = self.conv1(x)
        mask = f.relu(mask)
        mask = f.max_pool3d(mask, 2, stride=2)
        mask = self.conv2(mask)
        mask = f.relu(mask)
        mask = f.upsample(mask, scale_factor=2)
        x = (1 + mask) * x
        mask = f.max_pool3d(mask, (2, 1, 1), stride=(2, 1, 1))
        return x, mask


class ResNet3dMask(nn.Module):
    def __init__(self, num_classes, clip_length, crop_shape, mode):
        super(ResNet3dMask, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=True)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.mask1 = EDMask(64, 3, 1)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer10 = BasicBlock(64)
        self.layer11 = BasicBlock(64)
        self.mask2 = EDMask(64, 3, 1)
        self.layer20 = DownsampleBlock(64, 128)
        self.layer21 = BasicBlock(128)
        self.layer30 = DownsampleBlock(128, 256)
        self.layer31 = BasicBlock(256)
        self.layer40 = DownsampleBlock(256, 512)
        self.layer41 = BasicBlock(512)
        self.mode = mode

        self.avgpool = nn.AvgPool3d(
            (math.ceil(clip_length // 16), math.ceil(crop_shape[1] / 32), math.ceil(crop_shape[0] / 32)))
        self.fc = nn.Linear(512, num_classes)
        self.layers = []

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_out')
                nn.init.constant(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant(m.weight.data, 1)
                nn.init.constant(m.bias.data, 0)

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x, mask1 = self.mask1(x)
        mask1 = f.upsample(mask1, scale_factor=2)
        x = self.maxpool(x)

        # self.layers.append(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x, mask2 = self.mask2(x)
        mask2 = f.upsample(mask2, scale_factor=2)
        mask2 = f.upsample(mask2, scale_factor=2)
        # self.layers.append(x)
        x = self.layer20(x)
        x = self.layer21(x)
        # self.layers.append(x)
        x = self.layer30(x)
        x = self.layer31(x)
        # self.layers = y

        x = self.layer40(x)
        x = self.layer41(x)
        # self.layers.append(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        mask = (mask1 + mask2) / 2
        mask = torch.mean(mask, 1)
        return mask, x, self.layers
