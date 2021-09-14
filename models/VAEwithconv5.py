import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

#8, 256, 7, 7
prev_flatten_shape = (-1, 256, 7, 13)

class Flatten(nn.Module):
    def forward(self, input):
        global prev_flatten_shape
        prev_flatten_shape = input.shape
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(prev_flatten_shape[0], prev_flatten_shape[1], prev_flatten_shape[2], prev_flatten_shape[3])


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=512, z_dim=64):
        super(VAE, self).__init__()


        self.encoder = nn.Sequential(
            nn.BatchNorm2d(image_channels),
            nn.Conv2d(image_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Flatten(),
            nn.Linear( prev_flatten_shape[1]*prev_flatten_shape[2]*prev_flatten_shape[3], h_dim),
            nn.ReLU(),
        )
        #global prev_flatten_shape
        #self.fc0 = nn.Linear(23296, h_dim)
        #h_dim = prev_flatten_shape[1]*prev_flatten_shape[2]*prev_flatten_shape[3]
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        #self.fc3 = nn.Linear(z_dim, h_dim)



        self.decoder = nn.Sequential(
            nn.Linear(z_dim, prev_flatten_shape[1]*prev_flatten_shape[2]*prev_flatten_shape[3]),
            nn.ReLU(),
            UnFlatten(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, image_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return self.decoder(z), mu, logvar


    def weight_init(self):
        print('fc1 weight = ', self.fc1.weight)
        print('fc2 weight = ', self.fc2.weight)
        print('fc2 bias = ', self.fc2.bias)
        #nn.init.uniform_(self.fc2.weight, -8e-4, 8e-4)

    def loss_function(self, recon_x, x, mu, logvar):
        #x = x.reshape(x.shape[0], -1)
        #recon_x = x.reshape(recon_x.shape[0], -1)
        kld = -0.5 * torch.sum(1 + logvar - mu*mu - torch.exp(logvar), axis=-1)

        recon = torch.sum(torch.sum(torch.sum(F.binary_cross_entropy(recon_x.permute(0, 2, 3, 1), x.permute(0, 2, 3, 1), reduce=False, reduction='none'), axis=-1), axis=-1), axis=-1)


        loss = torch.mean(recon + kld)
        if torch.isnan(loss):
            print('weight = ', self.fc2.weight)
            print('input = ', x)
            print('input have nan ?  = ', torch.isnan(x))
            print('bias = ', self.fc2.bias)
            print('logvar = ', logvar)


        return loss


        """
        recon = torch.sum(F.binary_cross_entropy(x, recon_x, reduce=False, reduction='sum'), axis=-1)
        #print(F.binary_cross_entropy(x, recon_x, size_average = False, reduce=True, reduction='mean').shape)
        #print('bce = ', any({torch.isnan(x * torch.log(recon_x) + (1 - x) * torch.log(1 - recon_x))}))
        #recon = torch.sum(x * torch.log(recon_x) + (1 - x) * torch.log(1 - recon_x), axis=-1)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=-1)
        """


