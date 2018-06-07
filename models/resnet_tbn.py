'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

# inspired by https://zhuanlan.zhihu.com/p/31310177
class TrailingBatchNorm(torch.nn.Module):
    def __init__(self, gamma=1., beta=0.):
        super(TrailingBatchNorm, self).__init__()

        self.running_mean = None

        self.gamma = gamma
        self.beta = beta
        self.eps = 1e-5
        self.momentum = 0.9 # uses the same momentum definition as pytorch batchnorm

        self.register_backward_hook(self.backward_hook)
        self.first = True

    def forward(self, x):
        if x.dim() == 4:
            return self.spatial_batchnorm_forward(x, self.gamma, self.beta)
        else:
            return self.batchnorm_forward(x, self.gamma, self.beta)

    def backward_hook(self, module, grad_input, grad_output):
        # self.running_mean = self.temp_running_mean
        # self.running_variance = self.temp_running_variance
        return None

    def batchnorm_forward(self, x, gamma, beta):
        momentum = self.momentum
        eps = self.eps
        if self.first:
            self.first = False
            out = x

            sample_mean = torch.mean(x, dim=0)  # we do not need to calculate gradients over mean/var
            sample_var = torch.var(x, dim=0)

            self.running_mean = sample_mean.new_tensor(sample_mean.data)
            self.running_variance = sample_var.new_tensor(sample_var.data)
        else:

            sample_mean = torch.mean(x, dim=0)
            sample_var = torch.var(x, dim=0)

            x_normalized = (x - sample_mean) / torch.sqrt(sample_var + eps)
            out = gamma * x_normalized + beta

        # sample_mean = torch.mean(x, dim=0, keepdim=True).detach()  # we do not need to calculate gradients over mean/var
        # sample_var = torch.var(x, dim=0, keepdim=True).detach()

        self.running_mean = (1 - momentum) * self.running_mean + momentum * sample_mean.data
        self.running_variance = (1 - momentum) * self.running_variance + momentum * sample_var.data

        return out

    def spatial_batchnorm_forward(self, x, gamma, beta):
        N, C, H, W = x.shape
        x_new = x.permute(0, 2, 3, 1).reshape(N*H*W, C)
        out = self.batchnorm_forward(x_new, gamma, beta)
        out = out.reshape(N, H, W, C).permute(0, 3, 1, 2)

        return out

class BasicBlock_TBN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_TBN, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = TrailingBatchNorm()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = TrailingBatchNorm()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                TrailingBatchNorm()
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_TBN(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_TBN, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = TrailingBatchNorm()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = TrailingBatchNorm()
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = TrailingBatchNorm()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                TrailingBatchNorm()
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_TBN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_TBN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = TrailingBatchNorm()
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_TBN():
    return ResNet_TBN(BasicBlock_TBN, [2,2,2,2])

def ResNet34_TBN():
    return ResNet_TBN(BasicBlock_TBN, [3,4,6,3])

def ResNet50_TBN():
    return ResNet_TBN(Bottleneck_TBN, [3,4,6,3])

def ResNet101_TBN():
    return ResNet_TBN(Bottleneck_TBN, [3,4,23,3])

def ResNet152_TBN():
    return ResNet_TBN(Bottleneck_TBN, [3,8,36,3])


def test():
    net = ResNet18_TBN()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()
