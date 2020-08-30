# ==============================================================================
#
# Decoder (reversed ResNet)
#
# ==============================================================================

import torch
import torch.nn as nn
from torch.nn import functional as F


__all__ = ['decoder11', 'decoder17', 'decoder31', 'decoder46']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def normal_init(m, mean, std):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class DecoderBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, scale=1, upsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(DecoderBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if scale != 1:
            self.up2d = nn.UpsamplingNearest2d(scale_factor=scale)

        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(inplanes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.scale = scale

    def forward(self, z):

        if self.scale != 1:
            z = self.up2d(z)

        identity = z

        out = self.conv1(z)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(z)

        out += identity
        out = self.relu(out)

        return out


class DecoderBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, scale=1, upsample=None, groups=1,
                 base_width=64, norm_layer=None):
        super(DecoderBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if scale != 1:
            self.up2d = nn.UpsamplingNearest2d(scale_factor=scale)
        width = int(inplanes / 4 * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.scale = scale

    def forward(self, z):

        out = self.conv1(z)
        out = self.bn1(out)
        out = self.relu(out)

        if self.scale != 1:
            z = self.up2d(z)
            out = self.up2d(out)

        identity = z

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(z)

        out += identity
        out = self.relu(out)

        return out


class Decoder(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, groups=1,
                 width_per_group=64, norm_layer=None):
        super(Decoder, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 512 * block.expansion
        if len(layers) != 3:
            raise ValueError("layers should be None or a 3-element tuple, "
                             "got {}".format(layers))
        self.groups = groups
        self.base_width = width_per_group
        self.layer1 = self._make_layer(block, 256, layers[0], scale=2)
        self.layer2 = self._make_layer(block, 128, layers[1], scale=2)
        self.layer3 = self._make_layer(block,  64, layers[2], scale=2)
        self.layer4 = self._make_layer(block,  64, 1, scale=2, exp_planes=0)
        self.layer5 = self._make_layer(block,  32, 1, scale=2, exp_planes=0)
        self.conv1 = nn.Conv2d(32, 3, kernel_size=7, padding=3, bias=False)
        # self.conv1 = conv3x3(32, 3)
        self.bn1 = norm_layer(3)

        self.expansion = block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DecoderBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, DecoderBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, scale=1, exp_planes=1):
        norm_layer = self._norm_layer
        upsample = None

        if exp_planes:
            planes *= block.expansion

        if scale != 1 or self.inplanes != planes:
            upsample = nn.Sequential(
                conv1x1(self.inplanes, planes),
                norm_layer(planes),
            )
        layers = [block(self.inplanes, planes, scale, upsample, self.groups,
                        self.base_width, norm_layer)]

        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
            continue

        return nn.Sequential(*layers)

    def forward(self, z):

        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        z = self.layer5(z)
        z = self.bn1(self.conv1(z))

        return torch.sigmoid(z)


def decoder11(**kwargs):
    r"""Decoder-11 model from
    "Deep Reversed Residual Learning for Image Recognition"
    """
    return Decoder(DecoderBasicBlock, [1, 1, 1], **kwargs)


def decoder17(**kwargs):
    r"""Decoder-17 model from
    "Deep Reversed Residual Learning for Image Recognition"
    """
    return Decoder(DecoderBasicBlock, [2, 2, 2], **kwargs)


def decoder31(**kwargs):
    r"""Decoder-31 model from
    "Deep Reversed Residual Learning for Image Recognition"
    """
    return Decoder(DecoderBasicBlock, [6, 4, 3], **kwargs)


def decoder46(**kwargs):
    r"""Decoder-46 model from
    "Deep Reversed Residual Learning for Image Recognition"
    """
    return Decoder(DecoderBottleneck, [6, 4, 3], **kwargs)