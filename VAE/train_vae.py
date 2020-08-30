# ==============================================================================
#
# (encoder: ResNet) + (latent_variable: N(0,1)) + (decoder)
#
# ==============================================================================

from __future__ import print_function
import os
import torch
from torch.nn import functional as F
from torch import optim
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
import time
from dataloader import KeyDataLoader
from models import *


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


# Reconstruction + KL divergence losses summed over all elements and batch
def variational_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(recon_x.size(0), - 1),
                                 x.view(x.size(0), - 1),
                                 reduction='sum')


    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return 0.1 * BCE, KLD


# gpu info.
use_cuda = torch.cuda.is_available()


def main():

    main_dir = '.'
    data_dir = './data'

    # Hyper-parameter settings
    train_epoch = 100
    image_size = 224
    crop = 32
    batch_size = 128
    num_workers = 1
    latent_size = 512
    lr = 0.0001

    # Network setup
    print('\n| Building network: ResNet34 + Decoder31')
    encoder = resnet34(pretrained=True)
    decoder = decoder31()
    net = VAE(encoder, decoder, latent_size)
    if use_cuda:
        net.cuda()

    # Data setup
    loader = KeyDataLoader(image_size, crop)
    train_loader, valid_loader, test_loader = loader.run(batch_size, num_workers,
                                                         data_dir=data_dir)

    # Instantiate an optimizer to train the model.
    optimizer = optim.Adam(net.parameters(), lr=lr) 
                                
    # Weights directory                            
    os.makedirs('checkpoint', exist_ok=True)

    # Reconstruction and latent variable sampling dir.
    os.makedirs('results_rec', exist_ok=True)
    os.makedirs('results_gen', exist_ok=True)
    os.makedirs('epoch_rec', exist_ok=True)

    best_loss = None
    for epoch in range(train_epoch):

        # TRAINING # ---------------------------------------------------------

        net.train()
        epoch_start_time = time.time()

        if (epoch + 1) == 12:
            optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")

        for step, (x, labels) in enumerate(train_loader):

            step_start_time = time.time()
            n_samples = float(labels.size(0))

            if use_cuda:
                x = x.cuda()

            net.zero_grad()

            rec, mu, logvar = net(x)

            loss_re, loss_kl = variational_loss(rec, x, mu, logvar)
            (loss_re + loss_kl).backward()
            optimizer.step()

            # Grab training results
            print(

                "| Epoch: {}/{}, ".format(epoch, train_epoch),
                "step: [{}/{}], ".format(step * len(x), len(train_loader.dataset)),
                "loss_re: {:.3f}, ".format(loss_re.item()/n_samples),
                "loss_kl: {:.3f}, ".format(loss_kl.item()/n_samples),
                "time: {:.3f}".format(time.time() - step_start_time),
                end="\r")

        # VALIDATION # -------------------------------------------------------

        net.eval()

        total = 0
        rec_loss = 0
        kld_loss = 0

        for step, (x, labels) in enumerate(valid_loader):

            n_samples = float(labels.size(0))

            if use_cuda:
                x = x.cuda()

            with torch.no_grad():

                rec, mu, logvar = net(x)

                loss_re, loss_kl = variational_loss(rec, x, mu, logvar)
                total += n_samples
                rec_loss += loss_re.item()
                kld_loss += loss_kl.item()

                n = min(x.size(0), 8)
                comparison = torch.cat([x[:n], rec[:n]])
                save_image(comparison.data.cpu(),
                           main_dir + '/epoch_rec/rec_' + str(step) + '.png',
                           nrow=n)

        # Grab valid. results
        print(

            "| Epoch: {}/{}, ".format(epoch, train_epoch),
            "val_loss_re: {:.3f}, ".format(rec_loss/total),
            "val_loss_kl: {:.3f}, ".format(kld_loss/total),
            "time: {:.3f}, ".format(time.time() - epoch_start_time),
            "lr: {:.6f}".format(optimizer.param_groups[0]['lr']))

        loss = (rec_loss + kld_loss) / total
        if best_loss is None or loss < best_loss:
            best_loss = loss
            print('| Saving Best Model ...')
            save_checkpoint(
                state={'state_dict': net.state_dict(), },
                filename=main_dir + '/checkpoint/vae_net_weights_%d.pth.tar' % (epoch + 1))

            n = min(x.size(0), 8)
            comparison = torch.cat([x[:n], rec[:n]])
            save_image(comparison.data.cpu(),
                        main_dir + '/results_rec/rec_' + str(epoch + 1) + '.png',
                        nrow=n)

            sample = torch.randn(16, latent_size)
            if use_cuda:
                sample = sample.cuda()
            with torch.no_grad():
                sample = net.decode(sample).cpu()
                save_image(sample.data.view(16, 3, image_size, image_size),
                           main_dir + '/results_gen/sample_' + str(epoch + 1) + '.png')


if __name__ == '__main__':
    main()
