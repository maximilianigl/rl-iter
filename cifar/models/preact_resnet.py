'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

"""
Code inspiration from https://github.com/1Konny/VIB-pytorch/blob/master/model.py
"""
def xavier_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        print("Initiating bottleneck")
        nn.init.xavier_uniform(m.weight,gain=nn.init.calculate_gain('relu'))
        m.bias.data.zero_()

class Bottleneck(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.output_size = output_size

        self.encode = nn.Linear(input_size, 2 * output_size)
        self.weight_init()

    def forward(self, x):
        device = x.device
        stats = self.encode(x)

        mu = stats[:,:self.output_size]
        std = F.softplus(stats[:,self.output_size:])

        prior = Normal(torch.zeros(self.output_size).to(device), torch.ones(self.output_size).to(device))

        dist = Normal(mu, std)
        kl = kl_divergence(dist, prior)

        return mu, dist.rsample(), kl

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_bn=True):
        super(PreActBlock, self).__init__()

        if use_bn:
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            # Replace with identity
            self.bn1 = nn.Sequential()
            self.bn2 = nn.Sequential()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_bn=True, use_vib=False):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.use_vib = use_vib
        self.frozen_encoder = False

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use_bn=use_bn)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use_bn=use_bn)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use_bn=use_bn)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, use_bn=use_bn)
        # if use_vib:
        #     raise NotImplementedError("Make bn layer optional")
        #     self.bottleneck_layer = Bottleneck(512*block.expansion, 512)
        # else:
        #     self.bottleneck_layer = nn.Linear(512*block.expansion, 512)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, use_bn):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_bn))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def reset_last_layer(self):
        self.linear.reset_parameters()

    def freeze(self):
        self.frozen_encoder = True

    def unfreeze(self):
        self.frozen_encoder = False

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # if self.use_vib:
        #     _, out, kl = self.bottleneck_layer(out)
        # else:
        #     out = self.bottleneck_layer(out)  # Just a nn.Linear here
        #     kl = torch.Tensor([0.])
        kl = torch.Tensor([0.])
        if self.frozen_encoder:
            out = out.detach()
        self.embedding = out
        out = self.linear(out)
        return out, kl


def PreActResNet18(use_bn=True, use_vib=False):
    return PreActResNet(PreActBlock, [2,2,2,2], use_bn=use_bn, use_vib=use_vib)


def test():
    net = PreActResNet18()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

# test()
