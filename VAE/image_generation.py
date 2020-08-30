# ==============================================================================
#
# (encoder: ResNet) + (latent_variable: N(0,1)) + (decoder: reversed ResNet)
#
# ============================================================================== 

import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
from models import *

weights_path = './checkpoint/weights_clothing/h500_e191.pth.tar'
latent_size = 500
nb_samples = 20

encoder = resnet34()
decoder = decoder31()
net = VAE(encoder, decoder, latent_size)

checkpoint = torch.load(weights_path, map_location='cpu')
net.load_state_dict(checkpoint['state_dict'])

# Weights directory                            
os.makedirs('samples', exist_ok=True)
    
for i in range(nb_samples):
    
    sample = torch.randn(16, latent_size)
    with torch.no_grad():
        sample = net.decode(sample).cpu()

    save_image(sample.data.view(16, 3, 224, 224), './samples/sample' + str(i+1) + '.png')