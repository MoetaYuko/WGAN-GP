import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils


def visualize_data(dataloader):
    batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training images")
    images = vutils.make_grid(batch[0][:64], normalize=True, range=(-1, 1))
    images = images.permute(1, 2, 0)
    plt.imshow(images)
    plt.show()


def visualize_batch(data, batches_done):
    plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.title("Batches done %d" % batches_done)
    images = vutils.make_grid(data.cpu().detach()[:25],
                              5,
                              normalize=True,
                              range=(-1, 1))
    images = images.permute(1, 2, 0)
    plt.imshow(images)
    plt.show()


def save_imgs(generator):
    r, c = 5, 5
    noise = torch.randn(r * c, 100)
    # gen_imgs should be shape (25, 64, 64, 3)
    gen_imgs = generator(noise)
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, :])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("samples/output.png")
    plt.close()
