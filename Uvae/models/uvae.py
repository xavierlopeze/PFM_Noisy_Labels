# ==============================================================================
#
# (encoder: ResNet) + (latent_variable: N(0,1)) + (decoder)
#                   + (classifier: Softmax)
#
# ==============================================================================

import torch
import torch.nn as nn
from torch.nn import functional as F
from models import resnet34


__all__ = ['DecoderBasicBlock', 'Decoder', 'UVae']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)


def normal_init(m, mean, std):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class DecoderBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, scale=2, base_width=64, norm_layer=None):
        super(DecoderBasicBlock, self).__init__()

        if base_width != 64:
            raise ValueError('DecoderBasicBlock only supports base_width=64')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.up2d = nn.UpsamplingNearest2d(scale_factor=scale)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

    def forward(self, z):

        z = self.up2d(z)
        z = self.relu(self.bn1(self.conv1(z)))
        z = self.relu(self.bn2(self.conv2(z)))

        return z


class Decoder(nn.Module):
    inplanes = 512

    def __init__(self, block, layers=None, base_width=64, norm_layer=None):
        super(Decoder, self).__init__()
        if layers is None:
            layers = [1, 1, 1, 1]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.base_width = base_width

        self.layer1 = self._make_layer(block, 256, layers[0], scale=2)
        self.layer2 = self._make_layer(block, 128, layers[1], scale=2)
        self.layer3 = self._make_layer(block, 64, layers[2], scale=2)
        self.layer4 = block(self.inplanes, 64, scale=2)
        self.layer5 = block(64, 32, scale=2)

        self.conv1 = conv3x3(32, 3)
        self.bn1 = norm_layer(3)

    def _make_layer(self, block, planes, blocks, scale=1):
        norm_layer = self._norm_layer

        layers = [block(self.inplanes, planes * block.expansion, scale,
                        base_width=self.base_width, norm_layer=norm_layer)]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
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

    def weight_init(self, mean, std):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean, std)


class UVae(nn.Module):
    out_planes = 512

    def __init__(self, encoder=None, decoder=None, num_classes=1000,
                 latent_variable_size=500, **kwargs):
        super(UVae, self).__init__()

        # Encoder
        if encoder is None:
            encoder = resnet34(**kwargs)
        self._encoder = encoder

        # Latent variable
        # self.fc1 = nn.Linear(self.out_planes * encoder.expansion * 8 * 8,
        #                      latent_variable_size)
        # self.fc2 = nn.Linear(self.out_planes * encoder.expansion * 8 * 8,
        #                      latent_variable_size)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc3 = nn.Linear(self.out_planes * encoder.expansion,
                             num_classes)

        # Decoder
        # self.fc4 = nn.Linear(latent_variable_size, self.out_planes * 8 * 8)
        # if decoder is None:
        #    decoder = Decoder(DecoderBasicBlock)
        self._decoder = decoder

    def encode(self, x):
        return self._encoder(x)

    def classify(self, x):
        out = self.avgpool(x)
        out = torch.flatten(out, 1)

        return self.fc3(out)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = F.relu(self.fc4(z))
        z = z.view(z.size(0), -1, 8, 8)
        z = self._decoder(z)

        return z

    def forward(self, x, inference=True):
        # Encoder without latent var.
        x = self.encode(x)
        # Classifier
        out = self.classify(x)

        if inference:
            return out
        # Latent variable
        x = torch.flatten(x, 1)
        mu, logvar = self.fc1(x), self.fc2(x)
        z = self.reparameterize(mu, logvar)

        return out, self.decode(z), mu, logvar
