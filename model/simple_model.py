from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Conv2d
import torch.nn.functional as F
import torch
# from model.modules import ConvOffset2d
from model.deform_conv2d_modules import ConvOffset2d
from torchvision.models import resnet


class LeNet(nn.Module):
    def __init__(self, mode):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)  # 32->30
        self.conv2 = nn.Conv2d(6, 6, 3)  # 30->28
        self.pool = nn.MaxPool2d(2, 2)  # 28-14 10->5
        self.conv3 = nn.Conv2d(6, 16, 3)  # 14->12
        self.conv4 = nn.Conv2d(16, 16, 3)  # 12->10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.layers = []
        self.mode = mode

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)
        x = F.relu(x)
        if self.mode == 'test':
            self.layers.append(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        if self.mode == 'test':
            self.layers.append(x)

        x = self.conv3(x)
        x = F.relu(x)
        if self.mode == 'test':
            self.layers.append(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool(x)
        if self.mode == 'test':
            self.layers.append(x)

        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x, self.layers


class PLeNet(nn.Module):
    def __init__(self, mode):
        super(PLeNet, self).__init__()
        self.conv1 = nn.Conv2d(5, 6, 3)  # 32->30
        self.conv2 = nn.Conv2d(6, 6, 3)  # 30->28
        self.pool = nn.MaxPool2d(2, 2)  # 28-14 10->5
        self.conv3 = nn.Conv2d(6, 16, 3)  # 14->12
        self.conv4 = nn.Conv2d(16, 16, 3)  # 12->10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.layers = []

    def p_1d(self, x):
        area = x.size(2) * x.size(3)
        v = [(i - area / 2) / (area / 2) for i in range(area)] * x.size(0)
        p = Variable(torch.FloatTensor(v)).cuda()
        p = p.view((x.size(0), 1, x.size(2), x.size(3)))
        x = torch.cat((x, p, p), 1)

        return x

    def p_2d(self, x):
        h = x.size(2)
        w = x.size(3)
        v1 = [(i - w / 2) / (w / 2) for i in range(w)] * h * x.size(0)
        v2 = [(i - h / 2) / (h / 2) for i in range(h)] * w
        p1 = Variable(torch.FloatTensor(v1)).cuda()
        p2 = Variable(torch.FloatTensor(v2).view((w, h)).t()).cuda()
        p2 = torch.stack([p2 for _ in range(x.size(0))])
        p1 = p1.view((x.size(0), 1, x.size(2), x.size(3)))
        p2 = p2.view((x.size(0), 1, x.size(2), x.size(3)))
        x = torch.cat((x, p1, p2), 1)

        return x

    def forward(self, x):
        self.layers = []
        x = self.p_2d(x)

        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.pool(x)

        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x, self.layers


class SStn(nn.Module):
    def __init__(self, channel, size):
        super(SStn, self).__init__()
        # self.max_pool = nn.MaxPool2d((2,2))
        # self.conv1 = nn.Conv2d(channel, channel, 5, padding=2)
        # self.conv2 = nn.Conv2d(channel, channel, 5, padding=2)
        # self.conv3 = nn.Conv2d(channel, 1, 1)
        self.conv = nn.Conv2d(channel, 1, 5, padding=2)
        self.fc = nn.Linear(size * size, 3)

    def forward(self, x):
        # s = F.relu(self.conv1(x))
        # s = F.relu(self.conv2(s))
        # s = self.conv3(s)
        s = self.conv(x)
        s = s.view(x.size(0), -1)
        s = self.fc(s)
        scale = s[:, 0].contiguous().view(-1, 1, 1)
        vec = s[:, 1:3].contiguous().view(-1, 2, 1)
        i = Variable(torch.stack([torch.eye(2) for x in range(x.size(0))])).cuda()
        i = i * (1 + scale)
        s = torch.cat((i, 10 * vec), 2)
        g = F.affine_grid(s, x.size())
        x = F.grid_sample(x, g)

        return x


class ICSStn(nn.Module):
    def __init__(self, channel, size):
        super(ICSStn, self).__init__()
        self.conv = nn.Conv2d(channel, 1, 5, padding=2)
        self.fc = nn.Linear(size * size, 3)
        self.conv1 = nn.Conv2d(channel, 1, 5, padding=2)
        self.fc1 = nn.Linear(size * size, 3)

    def icsstn(self, x_pre, x_ori, s0, v0, conv, fc):
        s = conv(x_pre)
        s = s.view(x_pre.size(0), -1)
        s = fc(s)
        scale = s[:, 0].contiguous().view(-1, 1, 1)
        vec = s[:, 1:3].contiguous().view(-1, 2, 1)
        # s1 = scale * s0
        s1 = scale * s0 + s0 + scale
        v1 = vec + v0
        i = Variable(torch.stack([torch.eye(2) for _ in range(x_pre.size(0))])).cuda()
        # i = i * s1
        i = i * (1 + s1)
        s = torch.cat((i, v1), 2)
        g = F.affine_grid(s, x_ori.size())
        x_aft = F.grid_sample(x_ori, g)

        return x_aft, s1, v1

    def forward(self, x):
        # s0 = Variable(torch.ones((x.size(0), 1, 1))).cuda()
        s0 = Variable(torch.zeros((x.size(0), 1, 1))).cuda()
        v0 = Variable(torch.zeros((x.size(0), 2, 1))).cuda()
        x_aft, s0, v0 = self.icsstn(x, x, s0, v0, self.conv, self.fc)
        x_aft, s0, v0 = self.icsstn(x_aft, x, s0, v0, self.conv, self.fc)
        x_aft, s0, v0 = self.icsstn(x_aft, x, s0, v0, self.conv1, self.fc1)
        x_aft, s0, v0 = self.icsstn(x_aft, x, s0, v0, self.conv1, self.fc1)

        return x_aft


class CStn(nn.Module):
    def __init__(self, channel, size):
        super(CStn, self).__init__()
        self.conv1 = nn.Conv2d(channel, 1, 5, padding=2)
        self.conv2 = nn.Conv2d(channel, 1, 5, padding=2)
        self.conv3 = nn.Conv2d(channel, 1, 5, padding=2)
        self.conv4 = nn.Conv2d(channel, 1, 5, padding=2)
        self.fc1 = nn.Linear(size * size, 6)
        self.fc2 = nn.Linear(size * size, 6)
        self.fc3 = nn.Linear(size * size, 6)
        self.fc4 = nn.Linear(size * size, 6)

    def cstn(self, x_pre, x_ori, s0, conv, fc):
        vec = Variable(torch.FloatTensor(x_pre.size(0) * [[[0, 0, 1]]])).cuda()

        s1 = conv(x_pre)
        s1 = s1.view(x_pre.size(0), -1)
        s1 = fc(s1)
        s1 = s1.view(-1, 2, 3)

        s1 = torch.cat((s1, vec), 1)
        s1 = torch.bmm(s1, s0)
        s = s1[:, 0:2, :].contiguous()

        g = F.affine_grid(s, x_ori.size())
        x_aft = F.grid_sample(x_ori, g)

        return x_aft, s1

    def forward(self, x):
        s = Variable(torch.stack([torch.eye(3) for x in range(x.size(0))])).cuda()
        x_aft = x

        x_aft, s = self.cstn(x_aft, x, s, self.conv1, self.fc1)
        # x_aft, s = self.cstn(x_aft, x, s, self.conv2, self.fc2)
        # x_aft, s = self.cstn(x_aft, x, s, self.conv3, self.fc3)
        # x_aft, s = self.cstn(x_aft, x, s, self.conv4, self.fc4)

        return x_aft


class ICStn(nn.Module):
    def __init__(self, channel, size):
        super(ICStn, self).__init__()
        self.conv = nn.Conv2d(channel, 1, 5, padding=2)
        self.fc = nn.Linear(size * size, 6)
        # self.conv1 = nn.Conv2d(channel, 1, 5, padding=2)
        # self.fc1 = nn.Linear(size * size, 6)

    def icstn(self, x_pre, x_ori, s0, conv, fc):
        vec = Variable(torch.FloatTensor(x_pre.size(0) * [[[0, 0, 1]]])).cuda()

        s1 = conv(x_pre)
        s1 = s1.view(x_pre.size(0), -1)
        s1 = fc(s1)
        s1 = s1.view(-1, 2, 3)

        s1 = torch.cat((s1, vec), 1)
        s1 = torch.bmm(s1, s0)
        s = s1[:, 0:2, :].contiguous()

        g = F.affine_grid(s, x_ori.size())
        x_aft = F.grid_sample(x_ori, g)

        return x_aft, s1

    def forward(self, x):
        s = Variable(torch.stack([torch.eye(3) for x in range(x.size(0))])).cuda()
        x_aft = x

        x_aft, s = self.icstn(x_aft, x, s, self.conv, self.fc)
        x_aft, s = self.icstn(x_aft, x, s, self.conv, self.fc)
        x_aft, s = self.icstn(x_aft, x, s, self.conv, self.fc)
        x_aft, s = self.icstn(x_aft, x, s, self.conv, self.fc)

        return x_aft


class DStn(nn.Module):
    def __init__(self, channel, size):
        super(DStn, self).__init__()
        self.conv1 = nn.Conv2d(channel, 1, 5, padding=2)
        self.conv2 = nn.Conv2d(channel, 1, 5, padding=2)
        self.fc1 = nn.Linear(size * size, 6)
        self.fc2 = nn.Linear(size * size, 6)

    def forward(self, x):
        s1 = self.conv1(x)
        s1 = s1.view(x.size(0), -1)
        s1 = self.fc1(s1)
        s1 = s1.view(-1, 2, 3)

        g = F.affine_grid(s1, x.size())
        x = F.grid_sample(x, g)

        s2 = self.conv2(x)
        s2 = s2.view(x.size(0), -1)
        s2 = self.fc2(s2)
        s2 = s2.view(-1, 2, 3)

        g = F.affine_grid(s2, x.size())
        x = F.grid_sample(x, g)

        return x


class Stn(nn.Module):
    def __init__(self, channel, size):
        super(Stn, self).__init__()
        self.conv = nn.Conv2d(channel, 1, 5, padding=2)
        self.fc = nn.Linear(size * size, 6)

    def forward(self, x):
        s = self.conv(x)
        s = s.view(x.size(0), -1)
        s = self.fc(s)
        s = s.view(-1, 2, 3)
        g = F.affine_grid(s, x.size())
        x = F.grid_sample(x, g)

        return x


class LeNetStn(nn.Module):
    def __init__(self, mode):
        super(LeNetStn, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)  # 32->30
        self.conv2 = nn.Conv2d(6, 6, 3)  # 30->28
        self.pool = nn.MaxPool2d(2, 2)  # 28-14 10->5
        self.conv3 = nn.Conv2d(6, 16, 3)  # 14->12
        self.conv4 = nn.Conv2d(16, 16, 3)  # 12->10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.layers = []
        self.mode = mode

        self.stn = SStn(6, 30)

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)
        x = F.relu(x)
        if self.mode == 'test':
            self.layers.append(x)
        x = self.stn(x)
        if self.mode == 'test':
            self.layers.append(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.pool(x)

        # self.layers.append(x)
        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.pool(x)
        # self.layers.append(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x, self.layers


class SimpleNet(nn.Module):
    def __init__(self, class_num, mode):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 3, padding=1, stride=2)  # 32->16
        self.conv2 = nn.Conv2d(10, 10, 3, padding=1, stride=1)  # 16->16
        self.conv3 = nn.Conv2d(10, 20, 3, padding=1, stride=2)  # 16->8
        self.conv4 = nn.Conv2d(20, 20, 3, padding=1, stride=1)  # 8->8
        # self.conv5 = nn.Conv2d(20, 40, 3, padding=1, stride=2)  # 8->4
        # self.conv6 = nn.Conv2d(40, 40, 3, padding=1, stride=1)  # 4->4
        self.avgpool = nn.AvgPool2d((8, 8))
        self.fc = nn.Linear(20, class_num)
        self.layers = []

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv5(x)
        # x = self.conv6(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, self.layers


class Mask(nn.Module):
    def __init__(self, channel, kernel, pad):
        super(Mask, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel, padding=pad)

    def forward(self, x):
        mask = self.conv1(x)
        x = (1 + mask) * x
        return x


class ICMask(nn.Module):
    def __init__(self, channel, kernel, pad):
        super(ICMask, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel, padding=pad)

    def icmask(self, x_pre, x_ori, mask_pre, conv):
        mask_aft = conv(x_pre)
        mask_aft = mask_pre + mask_aft
        x_aft = x_ori * (1 + mask_aft)

        return x_aft, mask_aft

    def forward(self, x):
        mask_aft = Variable(torch.zeros(x.size())).cuda()
        x_aft, mask_aft = self.icmask(x, x, mask_aft, self.conv1)
        x_aft, mask_aft = self.icmask(x_aft, x, mask_aft, self.conv1)
        x_aft, mask_aft = self.icmask(x_aft, x, mask_aft, self.conv1)
        x_aft, mask_aft = self.icmask(x_aft, x, mask_aft, self.conv1)
        return x_aft


class EDMask(nn.Module):
    def __init__(self, channel, kernel, pad):
        super(EDMask, self).__init__()
        self.conv2 = nn.Conv2d(channel // 2, channel // 2, kernel, padding=pad)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.conv1 = nn.Conv2d(channel, channel // 2, 1)
        self.conv3 = nn.Conv2d(channel // 2, channel, 1)

    def forward(self, x):
        mask = self.max_pool(x)
        mask = self.conv1(mask)
        mask = self.conv2(mask)
        mask = self.conv3(mask)
        mask = F.upsample(mask, scale_factor=2)
        mask = F.sigmoid(mask)
        x = (1 + mask) * x
        return x


class LeNetMask(nn.Module):
    def __init__(self, mode):
        super(LeNetMask, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)  # 32->30
        self.conv2 = nn.Conv2d(6, 6, 3)  # 30->28
        self.pool = nn.MaxPool2d(2, 2)  # 28-14 10->5
        self.conv3 = nn.Conv2d(6, 16, 3)  # 14->12
        self.conv4 = nn.Conv2d(16, 16, 3)  # 12->10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.layers = []
        self.mode = mode

        self.mask = ICMask(6, 5, 2)

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)
        x = F.relu(x)
        if self.mode == 'test':
            self.layers.append(x)
        x = self.mask(x)
        if self.mode == 'test':
            self.layers.append(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.pool(x)

        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x, self.layers


class EDDeform(nn.Module):
    def __init__(self, cin, cout, ker, c_p_g=1):
        super(EDDeform, self).__init__()
        self.deform = ConvOffset2d(cin, cout, ker)

        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.conv0 = Conv2d(cin, cin, 3)
        self.conv1 = Conv2d(cin, cin // 2, 1)
        self.conv2 = Conv2d(cin // 2, cin // 2, 3, padding=1)
        self.conv3 = Conv2d(cin // 2, cin // c_p_g * 2 * ker * ker, 1)

    def forward(self, x):
        t = self.conv0(x)
        t = self.max_pool(t)
        t = self.conv1(t)
        t = self.conv2(t)
        t = self.conv3(t)
        t = F.upsample(t, scale_factor=2, mode='nearest')

        x = self.deform(x, t)
        return x


class Deform(nn.Module):
    def __init__(self, cin, cout, ker, c_p_g=1):
        super(Deform, self).__init__()
        self.deform = ConvOffset2d(cin, cout, ker, channel_per_group=c_p_g)

        self.conv1 = Conv2d(cin, cin, 3)
        self.conv2 = Conv2d(cin, cin // c_p_g * 2 * ker * ker, 1)

    def forward(self, x):
        off = self.conv1(x)
        off = self.conv2(off)

        x = self.deform(x, off)
        return x, off


class LeNetDeform(nn.Module):
    def __init__(self, mode):
        super(LeNetDeform, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)  # 32->30
        self.conv2 = Conv2d(6, 6, 3)  # 30->28
        self.pool = nn.MaxPool2d(2, 2)  # 28-14 10->5
        self.conv3 = Conv2d(6, 16, 3)  # 14->12
        self.conv4 = nn.Conv2d(16, 16, 3)  # 12->10
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.layers = []
        self.mode = mode

        self.deform1 = Deform(6, 6, 3)
        # self.deform2 = Deform(16, 16, 3)
        # self.deform_0 = nn.Conv2d(6, 6, 3, padding=1)
        # self.deform_1 = nn.Conv2d(6, 6, 3)
        # self.deform_2 = nn.Conv2d(6, 6 // 1 * 2 * 3 * 3, 1)
        nn.init.uniform(self.deform1.conv1.weight.data, -1e-5, 1e-5)
        nn.init.uniform(self.deform1.conv1.bias.data, -1e-5, 1e-5)
        nn.init.uniform(self.deform1.conv2.weight.data, -1e-5, 1e-5)
        nn.init.uniform(self.deform1.conv2.bias.data, -1e-5, 1e-5)
        # nn.init.uniform(self.deform2.weight.data, -1e-5, 1e-5)
        # nn.init.uniform(self.deform2.bias.data, -1e-5, 1e-5)

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)
        x = F.relu(x)
        if self.mode == 'test':
            self.layers.append(x)
        x, off = self.deform1(x)
        # x = self.conv2(x)
        if self.mode == 'test':
            self.layers.append(x)
        x = F.relu(x)

        x = self.pool(x)

        # x, off = self.deform1(x)
        x = self.conv3(x)

        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)

        x = self.pool(x)

        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.mode == 'test':
            self.layers.append(off)
        return x, self.layers


class ICDeform(nn.Module):
    def __init__(self, cin, cout, ker, c_p_g=1):
        super(ICDeform, self).__init__()
        self.deform = ConvOffset2d(cin, cout, ker, padding=1)

        self.conv1 = Conv2d(cin, cin // c_p_g * 2 * ker * ker, 3, padding=1)

    def forward(self, x):
        off = self.conv1(x)
        t = self.deform(x, off)

        off = self.conv1(t)
        x = self.deform(x, off)
        return x


class LeNetP(nn.Module):
    def __init__(self, mode):
        super(LeNetP, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)  # 32->32
        self.conv2 = Conv2d(6, 6, 3, padding=1)  # 32->32
        self.pool = nn.MaxPool2d(2, 2)  # 32-16 16->8
        self.conv3 = Conv2d(6, 16, 3, padding=1)  # 16->16
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)  # 16->16
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.layers = []
        self.mode = mode
        self.deform = Deform(6, 6, 3)

    def forward(self, x):
        self.layers = []
        x = self.conv1(x)
        x = F.relu(x)

        x = self.deform(x)
        x = F.relu(x)

        x = self.pool(x)
        if self.mode == 'test':
            self.layers.append(x)
        x = self.conv3(x)
        if self.mode == 'test':
            self.layers.append(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)

        x = self.pool(x)

        x = x.view(-1, 16 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x, self.layers
