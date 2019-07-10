import argparse
import multiprocessing
import torch
import torch.utils.data
import torch.nn as nn
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


def train(dataloader, n_epochs, lr, n_critic=5, c=0.01):
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    netG = dcgan.Generator().to(device)
    netG.apply(weights_init)
    optG = optim.RMSprop(netG.parameters(), lr)

    netD = dcgan.Critic().to(device)
    netD.apply(weights_init)
    optD = optim.RMSprop(netD.parameters(), lr)

    noise = torch.randn(dataloader.batch_size, NOISE_LENGTH, device=device)

    batches_done = 0

    for epoch in range(n_epochs):
        # Train critic
        for i, (data, _) in enumerate(dataloader):
            data = data.to(device)

            optD.zero_grad()
            noise.normal_()
            # Gradients from G are not used, so detach to avoid computing them
            # Maximize (3) -> minimize its inverse
            lossD = -(netD(data).mean() - netD(netG(noise).detach()).mean())
            lossD.backward()
            optD.step()
            # Clamp the weights
            for p in netD.parameters():
                p.data.clamp_(-c, c)

            # Train generator
            if i % n_critic == 0:
                optG.zero_grad()
                noise.normal_()
                # Minimize EM distance
                lossG = -netD(netG(noise)).mean()
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
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--learning_rate', default=0.00005)
    parser.add_argument('--n_epoches', default=10)
    opts = parser.parse_args()

    dataloader = load_dataset(opts.dataset, opts.batch_size)
    visualize_data(dataloader)

    train(dataloader, opts.n_epoches, opts.learning_rate)


if __name__ == "__main__":
    main()
