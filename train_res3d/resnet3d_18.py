import torch.nn as nn
import torch
import math
from train_deform3d.modules import ConvOffset3d
from torch.autograd import Variable


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
        self.conv_off = nn.Conv3d(channel, channel // channel_per_group * 3 * 27, kernel_size=3, stride=1,
                                  padding=1, bias=True)
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

        return out


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

        off1 = self.conv_off1(x)
        self.layers.append(off1)
        out = self.conv1(x, off1)
        out = self.bn1(out)
        out = self.relu(out)

        off2 = self.conv_off2(x)
        self.layers.append(off2)
        out = self.conv2(out, off2)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out, self.layers


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
    def __init__(self, num_classes, clip_length, crop_shape):
        super(ResNet3d, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=True)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(BasicBlock(64), BasicBlock(64))
        self.layer2 = nn.Sequential(DownsampleBlock(64, 128), BasicBlock(128))
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
        # self.layers.append(x)
        x = self.layer1(x)
        # self.layers.append(x)
        x = self.layer2(x)
        # self.layers.append(x)
        x = self.layer3(x)
        # self.layers.append(x)

        x = self.layer4(x)
        # self.layers.append(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, self.layers


class DeformResNet3d(nn.Module):
    def __init__(self, num_classes, clip_length, crop_shape):
        super(DeformResNet3d, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=3, bias=True)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer10 = BasicBlock(64)
        self.layer11 = BasicBlock(64)
        self.layer20 = DownsampleBlock(64, 128)
        self.layer21 = BasicBlock(128)
        self.layer30 = DeformDownsampleBlock2(128, 256, 2)
        self.layer31 = DeformBasicBlock1(256, 2)
        self.layer40 = DownsampleBlock(256, 512)
        self.layer41 = BasicBlock(512)

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
        # self.layers.append(x)
        x = self.layer10(x)
        x = self.layer11(x)
        # self.layers.append(x)
        x = self.layer20(x)
        x = self.layer21(x)
        # self.layers.append(x)
        x = self.layer30(x)
        x, y = self.layer31(x)
        self.layers = y

        x = self.layer40(x)
        x = self.layer41(x)
        # self.layers.append(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

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
