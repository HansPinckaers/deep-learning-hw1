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

class AsyncBatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, beta, gamma, r_mu, r_sigma2):
        hath = (input - r_mu) * (r_sigma2 + 1e-5)**(-1. / 2.)
        ctx.save_for_backward(input, beta, gamma)
        return gamma * hath + beta

    @staticmethod
    def backward(ctx, grad_output):
        input, beta, gamma = ctx.saved_tensors

        dy = grad_output
        N = input.shape[0]
        eps = 1e-5

        mu = 1 / N * torch.sum(input, dim=0)  # Size (H,) maybe torch.mean is faster
        sigma2 = 1 / N * torch.sum((input - mu)**2, dim=0)  # Size (H,) maybe torch variance is faster

        dx = (1. / N) * gamma * (sigma2 + eps)**(-1. / 2.) * \
            (N * dy - torch.sum(dy, dim=0) - (input - mu) * (sigma2 + eps)**(-1.0) * torch.sum(dy * (input - mu), dim=0))

        dbeta = torch.sum(dy, dim=0)
        dgamma = torch.sum((input - mu) * (sigma2 + eps)**(-1. / 2.) * dy, dim=0)

        return dx, dbeta, dgamma, None, None


# inspired by https://zhuanlan.zhihu.com/p/31310177
class TrailingBatchNorm(torch.nn.Module):
    def __init__(self, gamma=1., beta=0.):
        super(TrailingBatchNorm, self).__init__()

        self.running_mean = None

        self.temp_running_mean = 0
        self.temp_running_variance = 0

        self.gamma = torch.Tensor([1.])
        self.beta = torch.Tensor([0.])
        self.eps = 1e-5
        self.momentum = 0.9  # uses the same momentum definition as pytorch batchnorm

        self.first = 0

    def forward(self, x):
        if x.dim() == 4:
            return self.spatial_batchnorm_forward(x, self.gamma, self.beta)
        else:
            return self.batchnorm_forward(x, self.gamma, self.beta)

    def batchnorm_forward(self, x, gamma, beta):
        if x.requires_grad:
            self.running_mean = self.temp_running_mean
            self.running_variance = self.temp_running_variance

        async_batchnorm = AsyncBatchNorm.apply
        momentum = self.momentum

        if self.first == 0:
            self.first += 1
            out = x
            if self.first == 1:
                self.running_mean = torch.mean(x, dim=0).detach()
                self.running_variance = torch.var(x, dim=0).detach()
        else:
            out = async_batchnorm(x, self.beta, self.gamma, self.running_mean, self.running_variance)

        if x.requires_grad:
            with torch.no_grad():
                sample_mean = torch.mean(x, dim=0)
                sample_var = torch.var(x, dim=0)

                self.temp_running_mean = (1 - momentum) * self.running_mean + momentum * sample_mean
                self.temp_running_variance = (1 - momentum) * self.running_variance + momentum * sample_var

        return out

    def spatial_batchnorm_forward(self, x, gamma, beta):
        N, C, H, W = x.shape
        x_new = x.permute(0, 2, 3, 1).reshape(N * H * W, C)
        out = self.batchnorm_forward(x_new, gamma, beta)
        out = out.reshape(N, H, W, C).permute(0, 3, 1, 2)

        return x


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
