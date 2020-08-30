# ==============================================================================
#
# (encoder: ResNet) + (latent_variable: N(0,1)) + (decoder: reversed ResNet)
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
from models import decoder11, decoder31, decoder46, resnet34, resnet50, SoftVAE


__all__ = ['save_checkpoint', 'bce_loss', 'variational_loss']


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def bce_loss(recon_x, x):
    loss = F.binary_cross_entropy(recon_x.view(recon_x.size(0), - 1),
                                  x.view(x.size(0), - 1),
                                  reduction='none')

    return loss.sum(dim=1)


def variational_loss(recon_x, x, mu, logvar):
    BCE = bce_loss(recon_x, x)
    
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return 0.1 * BCE, KLD


# Instantiate a classification loss function.
criterion = torch.nn.CrossEntropyLoss(reduction='none')


# gpu info.
use_cuda = torch.cuda.is_available()


def main():

    # wandb.init(project="Food101_Noise_Server_uvae")
    main_dir = '.'
    data_dir = './data'  # '../FOOD101/data'

    # Hyper-parameter settings
    nb_classes = 101
    train_epoch = 30
    image_size = 224
    crop = 32
    batch_size = 128
    num_workers = 1
    latent_size = 512
    lr = 0.0001
    tau = 0.0001

    # Network setup
    print('\n| Building network: ResNet34 + Decoder31')
    encoder = resnet34(pretrained=True)
    decoder = decoder31()
    net = SoftVAE(nb_classes, encoder, decoder, latent_size)
    if torch.cuda.is_available():
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

    # best_acc = 0
    best_loss = None
    for epoch in range(train_epoch):

        # TRAINING # ---------------------------------------------------------

        net.train()
        epoch_start_time = time.time()

        # if (epoch + 1) % 8 == 0:
        #     optimizer.param_groups[0]['lr'] /= 4
        #     print("learning rate change!")

        for step, (x, targets) in enumerate(train_loader):

            step_start_time = time.time()
            n_samples = targets.size(0)
            zero_vect = torch.zeros((n_samples, nb_classes))

            if use_cuda:
                x = x.cuda()
                targets = targets.cuda()
                zero_vect = zero_vect.cuda()

            net.zero_grad()

            rec, mu, logvar, z_rec, z_c = net(x)

            # Variational auto-encoder loss
            loss_re, loss_kl = variational_loss(rec, x, mu, logvar)

            # Reconstruction ratio loss
            with torch.no_grad():
                rec = net.decode(z_rec, zero_vect)
                r = loss_re / bce_loss(rec, x)

            # Classification loss
            loss_cl = criterion(z_c, targets)

            # Global loss
            loss1 = (r * loss_re).sum() + loss_kl
            loss2 = (r * loss_cl).sum()
            (tau * loss1 + (1. - tau) * loss2).backward()
            optimizer.step()

            _, predicted = torch.max(z_c.data, 1)
            correct = predicted.eq(targets).cpu().sum()

            # Grab training results
            print(

                "| Epoch: {}/{}, ".format(epoch, train_epoch),
                "step: [{}/{}], ".format(step * len(x), len(train_loader.dataset)),
                "loss_cl: {:.3f}, ".format(loss_cl.sum() / n_samples),
                "acc: {:.3f}, ".format(100. * correct / n_samples),
                "loss_re: {:.3f}, ".format(loss_re.sum() / n_samples),
                "loss_kl: {:.3f}, ".format(loss_kl.sum() / n_samples),
                "r: {:.3f}, ".format(r.sum() / n_samples),
                "time: {:.3f}".format(time.time() - step_start_time),
                end="\r")

        # VALIDATION # -------------------------------------------------------

        net.eval()

        total = 0
        correct = 0
        cla_loss = 0
        rec_loss = 0
        kld_loss = 0

        for step, (x, targets) in enumerate(valid_loader):

            n_samples = targets.size(0)

            if use_cuda:
                x, targets = x.cuda(), targets.cuda()

            with torch.no_grad():

                rec, mu, logvar, _, z_c = net(x)

                # Classification loss
                loss_cl = criterion(z_c, targets)

                # Variational auto-encoder loss
                loss_re, loss_kl = variational_loss(rec, x, mu, logvar)

                _, predicted = torch.max(z_c.data, 1)
                total += n_samples
                correct += predicted.eq(targets).cpu().sum()
                cla_loss += loss_cl.sum().item()
                rec_loss += loss_re.sum().item()
                kld_loss += loss_kl.item()

                n = min(x.size(0), 8)
                comparison = torch.cat([x[:n], rec[:n]])
                save_image(comparison.data.cpu(),
                           main_dir + '/results_rec/rec_' + str(step) + '.png',
                           nrow=n)

        # Grab valid. results
        acc = 100. * correct / total
        print(

            "| Epoch: {}/{}, ".format(epoch, train_epoch),
            "val_loss_cl: {:.3f}, ".format(cla_loss / total),
            "val_acc: {:.3f}, ".format(acc),
            "val_loss_re: {:.3f}, ".format(rec_loss / total),
            "val_loss_kl: {:.3f}, ".format(kld_loss / total),
            "time: {:.3f}, ".format(time.time() - epoch_start_time),
            "lr: {:.6f}".format(optimizer.param_groups[0]['lr']))

        # if acc > best_acc:
        #     best_acc = acc
        loss = (rec_loss + kld_loss) / total
        if best_loss is None or loss < best_loss:
            best_loss = loss
            print('| Saving Best Model ...')
            save_checkpoint(
                state={'state_dict': net.state_dict(), },
                filename=main_dir + '/checkpoint/vae_net_weights_%d.pth.tar' % epoch)

            # sample = torch.randn(16, latent_size)
            # if use_cuda:
            #     sample = sample.cuda()
            # with torch.no_grad():
            #     sample = net.decode(sample).cpu()
            #     save_image(sample.data.view(16, 3, image_size, image_size),
            #                main_dir + '/results_gen/sample_' + str(epoch + 1) + '.png')


if __name__ == '__main__':
    main()
