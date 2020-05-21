import torch
from torch import nn
from torch.nn import functional as F


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class AdaptiveVAE(nn.Module):
    def __init__(self, nc=3, img=32, hid_size=100, ker=3, strides=(2, 2, 2),
                 leaky=(1, 1)):
        super(AdaptiveVAE, self).__init__()

        self.strides = strides
        self.act1 = nn.LeakyReLU(0.2) if leaky[0] else nn.ReLU()
        self.act2 = nn.LeakyReLU(0.2) if leaky[1] else nn.ReLU()

        # repl. padding, in and out_channels
        out_ch = None
        pad = 1 if ker == 3 else 2 if ker == 5 else 3
        stride_channels = [(s,
                            img * 2 ** (i - 1) if i != 0 else nc,
                            img * 2 ** i) for i, s in enumerate(strides)]

        # encoder
        i = 1
        for (s, in_ch, out_ch) in stride_channels:
            setattr(self, "conv%d" % i, nn.Conv2d(in_ch, out_ch, ker, s, pad))
            setattr(self, "bn2d%d" % i, nn.BatchNorm2d(out_ch))
            i += 1

        self.mid_ch = img // 2 ** sum([1 for s in strides if s == 2])

        # hidden layer
        self.fc1 = nn.Linear(out_ch * self.mid_ch ** 2, hid_size)
        self.fc2 = nn.Linear(out_ch * self.mid_ch ** 2, hid_size)

        # decoder
        self.fc3 = nn.Linear(hid_size, out_ch * self.mid_ch ** 2)

        for (s, out_ch, in_ch) in reversed(stride_channels):
            setattr(self, "up2d%d" % i, nn.UpsamplingNearest2d(scale_factor=s))
            setattr(self, "pd2d%d" % i, nn.ReplicationPad2d(pad))
            setattr(self, "conv%d" % i, nn.Conv2d(in_ch, out_ch, ker, 1))
            if out_ch != nc:
                setattr(self, "bn2d%d" % i, nn.BatchNorm2d(out_ch, 1.e-3))
            i += 1

    def encode(self, x):

        i = 1
        for _ in self.strides:

            x = getattr(self, "conv%d" % i)(x)
            x = getattr(self, "bn2d%d" % i)(x)
            x = self.act1(x)
            print('encoder_output: %d' % i, x.shape)
            i += 1
        x = x.view(x.size(0), -1)

        return self.fc1(x), self.fc2(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):

        z = F.relu(self.fc3(z))
        z = z.view(z.size(0), -1, self.mid_ch, self.mid_ch)

        i = len(self.strides) + 1
        for _ in self.strides:
            print('decoder_input: %d' % i, z.shape)
            z = getattr(self, "up2d%d" % i)(z)
            z = getattr(self, "pd2d%d" % i)(z)
            z = getattr(self, "conv%d" % i)(z)
            if i != 2 * len(self.strides):
                z = self.act2(getattr(self, "bn2d%d" % i)(z))

            i += 1

        return torch.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)



