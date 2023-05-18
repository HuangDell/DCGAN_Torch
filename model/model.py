import torch.nn as nn

import config
from model.base import BaseModel


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.backbone = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(config.generator_input, config.generator_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(config.generator_features * 8),
            nn.ReLU(True),
            # state size. (config.generator_features*8) x 4 x 4
            nn.ConvTranspose2d(config.generator_features * 8, config.generator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.generator_features * 4),
            nn.ReLU(True),
            # state size. (config.generator_features*4) x 8 x 8
            nn.ConvTranspose2d(config.generator_features * 4, config.generator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.generator_features * 2),
            nn.ReLU(True),
            # state size. (config.generator_features*2) x 16 x 16
            nn.ConvTranspose2d(config.generator_features * 2, config.generator_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.generator_features),
            nn.ReLU(True),
            # state size. (config.generator_features) x 32 x 32
            nn.ConvTranspose2d(config.generator_features, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, image):
        return self.backbone(image)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.backbone = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, config.discriminator_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (config.discriminator_features) x 32 x 32
            nn.Conv2d(config.discriminator_features, config.discriminator_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.discriminator_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (config.discriminator_features*2) x 16 x 16
            nn.Conv2d(config.discriminator_features * 2, config.discriminator_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.discriminator_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (config.discriminator_features*4) x 8 x 8
            nn.Conv2d(config.discriminator_features * 4, config.discriminator_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(config.discriminator_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (config.discriminator_features*8) x 4 x 4
            nn.Conv2d(config.discriminator_features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image):
        return self.backbone(image)

