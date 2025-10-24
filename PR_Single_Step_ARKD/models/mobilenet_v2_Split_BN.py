import torch
import torch.nn as nn
import torch.nn.functional as F

# __all__ = ['mobilenet_v2']

class Block(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn1_auto = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn2_auto = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.bn3_auto = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        self.shortcut_bn = nn.Sequential()
        self.shortcut_bn_auto = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            )
            self.shortcut_bn = nn.Sequential(
                nn.BatchNorm2d(out_planes)
            )
            self.shortcut_bn_auto = nn.Sequential(
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = x[0]
        batch_norm = x[1]

        out1 = self.conv1(out)
        if batch_norm=='base':
            out1 = F.relu(self.bn1(out1))
        elif batch_norm=='auto':
            out1 = F.relu(self.bn1_auto(out1))

        out1 = self.conv2(out1)
        if batch_norm=='base':
            out1 = F.relu(self.bn2(out1))
        elif batch_norm=='auto':
            out1 = F.relu(self.bn2_auto(out1))

        out1 = self.conv3(out1)
        if batch_norm=='base':
            out1 = self.bn3(out1)
        elif batch_norm=='auto':
            out1 = self.bn3_auto(out1)

        if self.stride == 1:
            feat = self.shortcut(out)
            if batch_norm == 'base':
                feat = self.shortcut_bn(feat)
            elif batch_norm=='auto':
                feat = self.shortcut_bn_auto(feat)
            out1 = out1 + feat
        return [out1, batch_norm]

class MobileNetV2(nn.Module):
    #(expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn1_auto = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.bn2_auto = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x, batch_norm='base', is_feat=False):
        out = self.conv1(x)
        if batch_norm=='base':
            out = F.relu(self.bn1(out))
        elif batch_norm=='auto':
            out = F.relu(self.bn1_auto(out))
        out = self.layers([out, batch_norm])
        out = self.conv2(out[0])
        if batch_norm=='base':
            out = F.relu(self.bn2(out))
        elif batch_norm=='auto':
            out = F.relu(self.bn2_auto(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if is_feat:
            return self.linear(out), out
        out = self.linear(out)
        return out

def mobilenet_v2_SplitBN(num_classes=10):
    net = MobileNetV2(num_classes=num_classes)
    return net
