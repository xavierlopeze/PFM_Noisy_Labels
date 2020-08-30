# ==============================================================================
#
# VAE --> (encoder: ResNet) + (latent_variable: N(0,1)) + (decoder: reversed ResNet)
#
# ==============================================================================

import torch
import torch.nn as nn
from torch.nn import functional as F
from models import resnet34, decoder31


__all__ = ['VAE']


class VAE(nn.Module):

    def __init__(self, encoder=None, decoder=None, latent_size=500):
        super(VAE, self).__init__()

        # Encoder
        if encoder is None:
            encoder = resnet34()
        self.encoder = encoder

        # Latent variable
        intermediate_size = 512 * encoder.expansion * 7 * 7
        self.fc1 = nn.Linear(intermediate_size, latent_size)
        self.fc2 = nn.Linear(intermediate_size, latent_size)

        # Decoder
        self.fc3 = nn.Linear(latent_size, intermediate_size)
        if decoder is None:
            decoder = decoder31()
        self.decoder = decoder

        if encoder.expansion != decoder.expansion:
            raise ValueError('Encoder expansion, {} != Decoder expansion, {}'
                             ''.format(encoder.expansion, decoder.expansion))

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)

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
        z = z.view(z.size(0), -1, 7, 7)
        z = self.decoder(z)

        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

