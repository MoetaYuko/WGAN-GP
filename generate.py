import argparse
import torch
import torch.utils.data
import torchvision.utils as vutils
import models.dcgan as dcgan

# Constants
NOISE_LENGTH = 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_images', default=25)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model', required=True)
    opts = parser.parse_args()

    # GPU
    device = 'cpu'
    if opts.cuda:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            print('CUDA not available!')

    # Load pretrained model
    netG = dcgan.Generator()
    netG.load_state_dict(torch.load(opts.model, map_location=device))
    netG = netG.to(device)
    netG.eval()

    # Start generation
    noise = torch.randn(opts.num_images, NOISE_LENGTH, device=device)
    fake = netG(noise)
    vutils.save_image(fake,
                      "samples/generate.png",
                      5,
                      normalize=True,
                      range=(-1, 1))


if __name__ == "__main__":
    main()
