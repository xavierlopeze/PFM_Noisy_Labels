# ==============================================================================
#
# (encoder: ResNet) + (latent_variable: N(0,1)) + (decoder)
#                   + (classifier: Softmax)
#
# ==============================================================================

from __future__ import print_function
import numpy as np
import pickle
import time
import random
import os
import torch
from torch.nn import functional as F
from torch import optim
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
import time
from dataloader import KeyDataLoader
from models import Decoder, DecoderBasicBlock, UVae, resnet34, resnet50
# import wandb


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

    return BCE, KLD


# Instantiate a classification loss function.
criterion = torch.nn.CrossEntropyLoss()

# gpu info.
use_cuda = torch.cuda.is_available()


def main():

    # wandb.init(project="Food101_Noise_Server_uvae")
    main_dir = '.'
    data_dir = '../FOOD101/data'

    # Hyper-parameter settings
    train_epoch = 15
    image_size = 224
    crop = 32
    batch_size = 64 
    num_workers = 1
    latent_variable_size = 500
    lr = 0.0001 

    # Network setup
    print('\n| Building network: ResNet34 + Decoder')
    encoder = resnet34(pretrained=True)
    # decoder = Decoder(DecoderBasicBlock)
    # decoder.weight_init(0, 0.02)
    decoder = None
    net = UVae(encoder=encoder, decoder=decoder, num_classes=101,
               latent_variable_size=latent_variable_size)
    if torch.cuda.is_available():
        net.cuda()
    net.train()

    # Data setup
    loader = KeyDataLoader(image_size, crop)
    train_loader, valid_loader, test_loader = loader.run(
        batch_size, num_workers, data_dir=data_dir)

    # Instantiate an optimizer to train the model.
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999),
                           weight_decay=1e-5)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
    #                             weight_decay=0.001)
                                
    # Weights directory                            
    os.makedirs('checkpoint', exist_ok=True)

    # Reconstruction and latent variable sampling dir.
    os.makedirs('results_rec', exist_ok=True)
    os.makedirs('results_gen', exist_ok=True)

    best_acc = 0
    best_loss = None
    for epoch in range(train_epoch):

        net.train()
        epoch_start_time = time.time()

        if (epoch + 1) % 8 == 0:
            optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")

        for step, (x, labels) in enumerate(train_loader):

            step_start_time = time.time()

            if use_cuda:
                x = x.cuda()
                labels = labels.cuda()

            net.train()
            net.zero_grad()
            cl, rec, mu, logvar = net(x)

            loss_cl = criterion(cl, labels)
            # loss_re, loss_kl = variational_loss(rec, x, mu, logvar)
            # (loss_re + loss_kl + loss_cl).backward()
            (loss_cl).backward()
            optimizer.step()

            _, predicted = torch.max(cl, 1)
            correct = predicted.eq(labels).cpu().sum()

            # Grab training results
            train_acc = float(correct) / float(labels.size(0))
            print(

                "| Epoch: {}/{}, ".format(epoch, train_epoch),
                "step: [{}/{}], ".format(step * len(x), len(train_loader.dataset)),
                "loss_cl: {:.3f}, ".format(loss_cl.item()/labels.size(0)),
                # "loss_re: {:.3f}, ".format(loss_re.item()/labels.size(0)),
                # "loss_kl: {:.3f}, ".format(loss_kl.item()/labels.size(0)),
                "acc: {:.3f}, ".format(train_acc),
                "time: {:.3f}".format(time.time() - step_start_time),
                end="\r")

        correct = 0
        total = 0
        cla_loss = 0
        rec_loss = 0
        kld_loss = 0

        for step, (x, labels) in enumerate(valid_loader):
            with torch.no_grad():
                net.eval()
                if use_cuda:
                    x = x.cuda()
                    labels = labels.cuda()

                cl, rec, mu, logvar = net(x)

                loss_cl = criterion(cl, labels)
                # loss_re, loss_kl = variational_loss(rec, x, mu, logvar)

                _, predicted = torch.max(cl, 1)
                total += labels.size(0)
                correct += predicted.eq(labels).cpu().sum()
                cla_loss += loss_cl.item()
                # rec_loss += loss_re.item()
                # kld_loss += loss_kl.item()

                # n = min(x.size(0), 8)
                # comparison = torch.cat([x[:n], rec[:n]])
                # save_image(comparison.data.cpu(),
                #            main_dir + '/results_rec/rec_' + str(step) + '.png',
                #            nrow=n)

        # Grab valid. results
        valid_acc = float(correct) / float(total)
        print(

            "| Epoch: {}/{}, ".format(epoch, train_epoch),
            "val_loss_cl: {:.3f}, ".format(cla_loss/float(total)),
            # "val_loss_re: {:.3f}, ".format(rec_loss/float(total)),
            # "val_loss_kl: {:.3f}, ".format(kld_loss/float(total)),
            "val_acc: {:.3f}, ".format(valid_acc),
            "time: {:.3f}, ".format(time.time() - epoch_start_time),
            "lr: {:.6f}".format(optimizer.param_groups[0]['lr']))
            
        # wandb.log({'epoch': epoch, 'accy_val' : valid_acc })
        # Save checkpoint when best model
        if valid_acc > best_acc:
            best_acc = valid_acc
        # loss = (rec_loss + kld_loss) / float(total)
        # if best_loss is None or loss < best_loss:
        #     best_loss = loss
            print('| Saving Best Model ...', end="\r")
            save_checkpoint(
                state={'state_dict': net.state_dict(), },
                filename=main_dir + '/checkpoint/vae_net_weights_%d.pth.tar' % epoch)

            # sample = torch.randn(16, latent_variable_size)
            # if use_cuda:
            #     sample = sample.cuda()
            # with torch.no_grad():
            #     sample = net.decode(sample).cpu()
            #     save_image(sample.data.view(16, 3, image_size, image_size),
            #                main_dir + '/results_gen/sample_' + str(epoch + 1) + '.png')


if __name__ == '__main__':
    main()
