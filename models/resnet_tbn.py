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

def _make_bipolar(fn):
    def _fn(x, *args, **kwargs):
        dim = 0 if x.dim() == 1 else 1
        x0, x1 = torch.chunk(x, chunks=2, dim=dim)
        y0 = fn(x0, *args, **kwargs)
        y1 = -fn(-x1, *args, **kwargs)
        return torch.cat((y0, y1), dim=dim)

    return _fn

def max_relu():
    def _fn(x, *args, **kwargs):
        return torch.clamp(x, min=-2, max=2)
    return _fn

brelu = _make_bipolar(F.relu)
brelu = max_relu()
brelu = F.elu


class AsyncBatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, beta, gamma, r_mu, r_sigma2):
        hath = (input - 0.25*r_mu) / (0.1*torch.sqrt(r_sigma2) + 0.9)
        ctx.save_for_backward(input, beta, gamma, 0.25*r_mu, 0.1*torch.sqrt(r_sigma2) + 0.9)
        return gamma * hath + beta

    @staticmethod
    def backward(ctx, grad_output):
        input, beta, gamma, mu, sigma2 = ctx.saved_tensors

        dy = grad_output
        N = input.shape[0]
        eps = 1e-5

        mu = 1/N * torch.sum(input, dim=0)  # Size (H,) maybe torch.mean is faster
        sigma2 = 1/N * torch.sum((input - mu)**2, dim=0)  # Size (H,) maybe torch variance is faster

        dx = (1. / N) * gamma * (sigma2 + eps)**(-1. / 2.) * (N * dy - torch.sum(dy, dim=0)
                - (input - mu) * (sigma2 + eps)**(-1.0) * torch.sum(dy * (input - mu), dim=0))

        dbeta = torch.sum(dy, dim=0)
        dgamma = torch.sum((input - mu) * (sigma2 + eps)**(-1. / 2.) * dy, dim=0)

        return dx, dbeta, dgamma, None, None

# inspired by https://zhuanlan.zhihu.com/p/31310177
class TileBatchNorm(torch.nn.Module):
    def __init__(self, gamma=1., beta=0.):
        super(TileBatchNorm, self).__init__()

        self.running_mean = None

        self.temp_running_mean = 0
        self.temp_running_variance = 0

        self.gamma = None
        self.beta = None
        self.eps = 1e-5
        self.momentum = 0.1 # uses the same momentum definition as pytorch batchnorm

        self.first = True

    def forward(self, x):
        if x.dim() == 4:
            return self.batchnorm_forward(x, self.gamma, self.beta)
        else:
            return x

    def batchnorm_forward(self, x, gamma, beta):
        momentum = self.momentum

        N, C, H, W = x.shape
        tiles = 2
        x_new = x.contiguous().view(x.size(0), x.size(1), tiles, -1).transpose(1, 2).contiguous().view(N, tiles, -1)

        mean = torch.mean(x_new, dim=2)[:, :, None]
        variance = torch.var(x_new, dim=2)[:, :, None]
        out = (x_new - mean) / torch.sqrt(variance)
        out = x_new.contiguous().view(N, tiles, x.size(1), x.size(2)*x.size(3) // tiles).transpose(2, 1).contiguous().view(N, C, H, W)
        if self.gamma is None:
            self.gamma = torch.ones((C), requires_grad=True).cuda()
            self.beta = torch.zeros((C), requires_grad=True).cuda()
        out *= self.gamma[None, :, None, None]
        out += self.beta[None, :, None, None]
        return out

        G = 16

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+1e-5).sqrt()
        if self.gamma is None:
            self.gamma = torch.ones((C), requires_grad=True).cuda()
            self.beta = torch.zeros((C), requires_grad=True).cuda()
        x = x.view(N,C,H,W)
        return x * self.gamma[None, :, None, None] + self.beta[None, :, None, None]


# inspired by https://zhuanlan.zhihu.com/p/31310177
class TrailingBatchNorm(torch.nn.Module):
    def __init__(self, gamma=1., beta=0.):
        super(TrailingBatchNorm, self).__init__()

        self.running_mean = None

        self.temp_running_mean = 0
        self.temp_running_variance = 0

        self.gamma = torch.tensor([1.], requires_grad=True).cuda()
        self.beta = torch.tensor([0.], requires_grad=True).cuda()
        self.eps = 1e-5
        self.momentum = 0.001 # uses the same momentum definition as pytorch batchnorm

        self.first = 0

    def forward(self, x):
        return x
        if x.dim() != 4:
            return self.batchnorm_forward(x, self.gamma, self.beta)
        else:
            N, C, H, W = x.shape
            x_new = x.transpose(0, 1).contiguous().view(x.size(1), -1).transpose(0, 1)
            out = self.batchnorm_forward(x_new, self.gamma, self.beta)
            out = out.transpose(0, 1).view(C, N, H, W).transpose(0, 1)
            return out

    def batchnorm_forward(self, x, gamma, beta):
        if x.requires_grad:
            self.running_mean = self.temp_running_mean
            self.running_variance = self.temp_running_variance

        async_batchnorm = AsyncBatchNorm.apply
        momentum = self.momentum

        # if self.first < 2:
        #     self.first += 1
        #     out = x

        if self.first == 0:
            self.first = 1
            self.running_mean = torch.zeros((x.shape[1]), requires_grad=False).cuda()
            self.running_variance = torch.ones((x.shape[1]), requires_grad=False).cuda()

        # else:
        # if x.requires_grad:
            # mean = torch.mean(x, dim=0).detach()
            # variance = torch.var(x, dim=0).detach()

        mean = self.running_mean
        variance = self.running_variance

        # out = (x - 0.2*mean) / (0.2*torch.sqrt(variance) + 0.8)
        out = (x - mean) / (torch.sqrt(variance))
        # out = self.gamma * out + self.beta

            # out = async_batchnorm(x, self.beta, self.gamma, mean, variance)
        # else:
        #     out = async_batchnorm(x, self.beta, self.gamma, self.running_mean, self.running_variance)

        # sample_mean = torch.mean(x, dim=0)  # we do not need to calculate gradients over mean/var
        #    from pdb import set_trace; set_trace()

        if x.requires_grad:
            with torch.no_grad():
                sample_mean = torch.mean(x, dim=0)  # we do not need to calculate gradients over mean/var
                sample_var = torch.var(x, dim=0)

                if torch.sum(sample_mean == float("Inf")) > 0:
                    return out

        #         N, C = x.shape
        #         mu = 1/N * torch.sum(x, dim=0)  # Size (H,) maybe torch.mean is faster
        #         sigma2 = 1/N * torch.sum((x - mu)**2, dim=0)  # Size (H,) maybe torch variance is faster

        #         if torch.sum(mu != sample_mean) > 0:
        #             print(x, torch.max(x, dim=0))

        #         # if self.first > 45:
        #         #     print(torch.mean(x))
        #         #     print()
        #         #     print()
        #         summed = torch.sum(torch.abs((self.running_mean[0:3] - sample_mean[0:3]) / sample_mean[0:3]))
        #         if summed > 0.1:
        #             print(summed)

                self.temp_running_mean = (1 - momentum) * self.running_mean + momentum * sample_mean
                self.temp_running_variance = (1 - momentum) * self.running_variance + momentum * sample_var

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
        out = brelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = brelu(out)
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
        out = brelu(self.bn1(self.conv1(x)))
        out = brelu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = brelu(out)
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
        out = brelu(self.bn1(self.conv1(x)))
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
