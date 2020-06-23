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

    return BCE, KLD + 0.1


# Instantiate a classification loss function.
criterion = torch.nn.CrossEntropyLoss()

# gpu info.
use_cuda = torch.cuda.is_available()


def main():

    main_dir = '.'
    data_dir = './data/'

    # Hyper Parameter settings
    train_epoch = 100
    image_size = 256
    crop = 32
    batch_size = 2
    num_workers = 1
    latent_variable_size = 500
    lr = 1e-4

    # Network setup
    encoder = resnet50(pretrained=True)
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
    # optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999),
    #                        weight_decay=1e-5)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                                weight_decay=1e-4)

    # Reconstruction and latent variable sampling dir.
    os.makedirs('results_rec', exist_ok=True)
    os.makedirs('results_gen', exist_ok=True)

    best_acc = 0
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

            net.train()
            net.zero_grad()
            # cl, rec, mu, logvar = net(x)
            cl = net(x)

            loss_cl = criterion(cl, labels)
            # loss_re, loss_kl = variational_loss(rec, x, mu, logvar)
            # (loss_re + loss_kl + loss_cl).backward()
            loss_cl.backward()
            optimizer.step()

            _, predicted = torch.max(cl, 1)
            correct = predicted.eq(labels).cpu().sum()

            # Grab training results
            train_acc = float(correct) / float(labels.size(0))
            print(

                "| Epoch: {}/{}, ".format(epoch, train_epoch),
                "step: [{}/{}], ".format(step * len(x), len(train_loader.dataset)),
                "loss_cl: {:.3f}, ".format(loss_cl.item()),
                # "loss_re: {:.3f}, ".format(loss_re.item()),
                # "loss_kl: {:.3f}, ".format(loss_kl.item()),
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

                # cl, rec, mu, logvar = net(x)
                cl = net(x)

                loss_cl = criterion(cl, labels)
                # loss_re, loss_kl = variational_loss(rec, x, mu, logvar)

                _, predicted = torch.max(cl, 1)
                total += labels.size(0)
                correct += predicted.eq(labels).cpu().sum()
                cla_loss += loss_cl.item()
                # rec_loss += loss_re.item()
                # kld_loss += loss_kl.item()

                # n = x.size(0)
                # comparison = torch.cat([x[:n], rec[:n]])
                # save_image(comparison.data.cpu(),
                #            main_dir + '/results_rec/rec_' + str(step) + '.png',
                #            nrow=n)

        # Grab valid. results
        valid_acc = float(correct) / float(total)
        print(

            "| Epoch: {}/{}, ".format(epoch, train_epoch),
            "val_loss_cl: {:.3f}, ".format(cla_loss),
            # "val_loss_re: {:.3f}, ".format(rec_loss),
            # "val_loss_kl: {:.3f}, ".format(kld_loss),
            "val_acc: {:.3f}, ".format(valid_acc),
            "time: {:.3f}, ".format(time.time() - epoch_start_time),
            "lr: {:.6f}".format(optimizer.param_groups[0]['lr']))

        # Save checkpoint when best model
        if valid_acc > best_acc:
            best_acc = valid_acc
            print('| Saving Best Model ...', end="\r")
            save_checkpoint(
                state={'state_dict': net.state_dict(), },
                filename=main_dir + '/checkpoint/net_weights.pth.tar')

            # sample = torch.randn(16, latent_variable_size)
            # if use_cuda:
            #     sample = sample.cuda()
            # with torch.no_grad():
            #     sample = net.decode(sample).cpu()
            #     save_image(sample.data.view(16, 3, image_size, image_size),
            #                main_dir + '/results_gen/sample_' + str(epoch + 1) + '.png')


if __name__ == '__main__':
    main()
