from __future__ import print_function
from models import AdaptiveVAE
from dataloader import KeyDataset
import torch
import torch.utils.data
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader


def get_data(img=128,
             crop=0,
             batch_size=128,
             shuffle_first=True,
             num_workers=1):

    transform = transforms.Compose([
        transforms.Resize(img + crop),
        transforms.CenterCrop(img),
        transforms.ToTensor(),
    ])

    l1 = DataLoader(
        dataset=KeyDataset(mode='train', transform=transform),
        batch_size=batch_size,
        shuffle=shuffle_first,
        num_workers=num_workers)

    l2 = DataLoader(
        dataset=KeyDataset(mode='valid', transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

    return l1, l2


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(recon_x.size(0), - 1),
                                 x.view(x.size(0), - 1),
                                 reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def main():

    drive_dir = '.'

    # Hyper Parameter settings
    img = 32
    crop = 0
    num_epochs = 100
    batch_size = 128
    hid_size = 100
    ker = 3
    strides = (1, 2, 1, 2, 1, 2)
    leaky = (0, 1)
    lr = 1e-4  # 0.0005
    log_interval = 10

    # Networks setup
    print('\nModel setup')
    print('| Building network: AdaptiveVAE(hid_size=%s)' % hid_size)
    print('| Input image size: %s' % img)
    vae = AdaptiveVAE(img=img, hid_size=hid_size, ker=ker,
                      strides=strides, leaky=leaky)
    if torch.cuda.is_available():
        vae.cuda()
    vae.train()
    # vae.weight_init(mean=0, std=0.02)

    # Get the original_dataset
    train_loader, test_loader = get_data(img, crop, batch_size)

    # Instantiate an optimizer to train the model.
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999),
                                 weight_decay=1e-5)

    print('\nTraining model')
    print('| Training Epochs: %s' % num_epochs)
    print('| Initial Learning Rate: %s' % lr)

    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    use_cuda = torch.cuda.is_available()

    best_loss = None
    for epoch in range(num_epochs):
        vae.train()
        train_loss = 0
        for step, (data, _) in enumerate(train_loader):
            if use_cuda:
                data = data.cuda()
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.data.item()
            optimizer.step()
            if step % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, step * len(data), len(train_loader.dataset),
                    100. * step / len(train_loader),
                    loss.data.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))

        vae.eval()
        test_loss = 0
        for step, (data, _) in enumerate(test_loader):
            if use_cuda:
                data = data.cuda()
            with torch.no_grad():
                recon_batch, mu, logvar = vae(data)
                test_loss += loss_function(recon_batch, data, mu, logvar).data.item()
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n], recon_batch[:n]])
            save_image(comparison.data.cpu(),
                       drive_dir + '/conv_vae/reconstruction_' + str(step) + '.png',
                       nrow=n)

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

        # Save checkpoint when best model
        if epoch == 0 or test_loss < best_loss:
            best_loss = test_loss
            print('| Saving Best Model ...')  # , end="\r")
            save_point = drive_dir + '/checkpoint/vae_h%s.pth.tar' % hid_size
            save_checkpoint({'state_dict': vae.state_dict(), }, save_point)

        sample = torch.randn(64, hid_size)
        if use_cuda:
            sample = sample.cuda()
        sample = vae.decode(sample).cpu()
        save_image(sample.data.view(64, 3, img, img),
                   drive_dir + '/conv_vae/vae_sample_' + str(epoch+1) + '.png')


if __name__ == '__main__':
    main()
