import argparse
import multiprocessing
import torch
import torch.utils.data
import torch.nn as nn
import torch.autograd
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import models.dcgan as dcgan
from utils import visualize_data

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# For reproducibility
torch.manual_seed(0)

# Constants
NOISE_LENGTH = 100
NOISE_BASELINE = torch.randn(5 * 5, NOISE_LENGTH, device=device)


def load_dataset(root, batch_size):
    dataset = dset.ImageFolder(
        root,
        transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size,
                                             True,
                                             num_workers=2,
                                             pin_memory=True)
    return dataloader


def compute_gradient_penalty(real, fake, critic):
    batch_size = real.shape[0]
    coeff = torch.rand(batch_size, 1, 1, 1, device=device)
    x_hat = coeff * real + (1 - coeff) * fake
    x_hat.detach_().requires_grad_()
    d_x_hat = critic(x_hat)
    grads = torch.autograd.grad(d_x_hat,
                                x_hat,
                                torch.ones_like(d_x_hat, device=device),
                                create_graph=True)[0]
    penalty = torch.mean((grads.flatten(1).norm(dim=1) - 1)**2)
    return penalty


def train(dataloader,
          gradient_penalty,
          n_epochs,
          lr,
          n_critic=5,
          c=0.01,
          penalty_factor=10):
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    netG = dcgan.Generator().to(device)
    netG.apply(weights_init)
    optG = optim.Adam(netG.parameters(), lr,
                      (0, 0.9)) if gradient_penalty else optim.RMSprop(
                          netG.parameters(), lr)

    netD = dcgan.Critic(gradient_penalty).to(device)
    netD.apply(weights_init)
    optD = optim.Adam(netD.parameters(), lr,
                      (0, 0.9)) if gradient_penalty else optim.RMSprop(
                          netD.parameters(), lr)

    noise = torch.randn(dataloader.batch_size, NOISE_LENGTH, device=device)

    batches_done = 0

    for epoch in range(n_epochs):
        # Train critic
        for i, (data, _) in enumerate(dataloader):
            data = data.to(device)
            real_batch_size = len(data)

            optD.zero_grad()
            noise.normal_()
            fake = netG(noise[:real_batch_size])
            # Gradients from G are not used, so detach to avoid computing them
            # Maximize (3) -> minimize its inverse
            lossD = -(netD(data).mean() - netD(fake.detach()).mean())
            if gradient_penalty:
                lossD += penalty_factor * compute_gradient_penalty(
                    data, fake, netD)

            lossD.backward()
            optD.step()
            # Clamp the weights
            if not gradient_penalty:
                for p in netD.parameters():
                    p.data.clamp_(-c, c)

            # Train generator
            if i % n_critic == 0:
                optG.zero_grad()
                noise.normal_()
                # Minimize EM distance
                lossG = -netD(netG(noise[:real_batch_size])).mean()
                lossG.backward()
                optG.step()
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                      (epoch, n_epochs, batches_done % len(dataloader),
                       len(dataloader), lossD.item(), lossG.item()))

            if batches_done % 100 == 0:
                fake = netG(NOISE_BASELINE)
                vutils.save_image(fake,
                                  "samples/%d.png" % batches_done,
                                  5,
                                  normalize=True,
                                  range=(-1, 1))

            batches_done += 1

        # Save the model
        torch.save(netG.state_dict(), 'pretrain/netG_epoch_%d.pth' % epoch)
        torch.save(netD.state_dict(), 'pretrain/netD_epoch_%d.pth' % epoch)


def main():
    # BUG: https://github.com/microsoft/ptvsd/blob/master/TROUBLESHOOTING.md#1-multiprocessing-on-linuxmac
    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='path to dataset')
    parser.add_argument('--wgan_gp', action='store_true')
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--learning_rate')
    parser.add_argument('--n_epoches', default=10)
    opts = parser.parse_args()

    dataloader = load_dataset(opts.dataset, opts.batch_size)
    visualize_data(dataloader)

    # Default settings in corresponding papers
    lr = opts.learning_rate or (0.0001 if opts.wgan_gp else 0.00005)
    train(dataloader, opts.wgan_gp, opts.n_epoches, lr)


if __name__ == "__main__":
    main()
