import sys
sys.path.append(".")

import torch  # noqa: E402
from models import dcgan as dcgan  # noqa: E402


def test_output_shape():
    netG = dcgan.Generator()
    noise = torch.randn(2, 100)
    fake = netG(noise)
    assert fake.shape == torch.Size([2, 3, 64, 64])

    netD = dcgan.Critic()
    fake = torch.randn(2, 3, 64, 64)
    output = netD(fake)
    assert output.shape == torch.Size((2, ))
