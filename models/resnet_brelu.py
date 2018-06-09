'''ResNet in PyTorch, without Batch Normalization.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def _make_bipolar(fn):
    def _fn(x, *args, **kwargs):
        dim = 0 if x.dim() == 1 else 1
        x0, x1 = torch.chunk(x, chunks=2, dim=dim)
        y0 = fn(x0, *args, **kwargs)
        y1 = -fn(-x1, *args, **kwargs)
        return torch.cat((y0, y1), dim=dim)

    return _fn

brelu = _make_bipolar(F.relu)

class BasicBlock_br(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_br, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = brelu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = brelu(out)
        return out


class Bottleneck_br(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_br, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = brelu(self.conv1(x))
        out = brelu(self.conv2(out))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = brelu(out)
        return out


class ResNet_BR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_BR, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = brelu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_BR():
    return ResNet_BR(BasicBlock_br, [2,2,2,2])

def ResNet34_BR():
    return ResNet_BR(BasicBlock_br, [3,4,6,3])

def ResNet50_BR():
    return ResNet_BR(Bottleneck_br, [3,4,6,3])

def ResNet101_BR():
    return ResNet_BR(Bottleneck_br, [3,4,23,3])

def ResNet152_BR():
    return ResNet_BR(Bottleneck_br, [3,8,36,3])


def test():
    net = ResNet18_BR()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()
