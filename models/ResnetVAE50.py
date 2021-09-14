from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms



class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()
        planes = int(in_planes/stride)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class
        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out



class ResNet50_Encoder(nn.Module):
    def __init__(self, fc_hidden1=1024, drop_p=0.3, CNN_embed_dim=256, pre_train=True):
        super(ResNet50_Encoder, self).__init__()

        self.fc_hidden1, self.CNN_embed_dim = fc_hidden1, CNN_embed_dim
        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        resnet = models.resnet50(pretrained=pre_train)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc2_mu = nn.Linear(self.fc_hidden1, self.CNN_embed_dim)      # output = CNN embedding latent variables
        self.fc2_logvar = nn.Linear(self.fc_hidden1, self.CNN_embed_dim)  # output = CNN embedding latent variables
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv
        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        mu, logvar = self.fc2_mu(x), self.fc2_logvar(x)
        return mu, logvar



class ResNet50_Decoder(nn.Module):
    def __init__(self, fc_hidden1=1024, CNN_embed_dim=256, num_Blocks=[1,2,2,2]):
        super(ResNet50_Decoder, self).__init__()
        self.in_planes = 512
        self.fc_hidden1, self.CNN_embed_dim = fc_hidden1, CNN_embed_dim
        self.resnet18_dim = 512
        if True:
            self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
            self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
            self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
            self.layer1 = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=2, padding=0, stride=2, bias=False),
                nn.BatchNorm2d(64)
            )
            self.layer0 = nn.Sequential(
                nn.ConvTranspose2d(64, 3, kernel_size=8, padding=3, stride=2, bias=False),
                nn.BatchNorm2d(3),
                nn.Sigmoid()
            )
        else:
            self.layer4 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(256, momentum=0.01),
                nn.ReLU(inplace=True),
            )
            self.layer3 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(128, momentum=0.01),
                nn.ReLU(inplace=True),
            )
            self.layer2 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(64, momentum=0.01),
                nn.ReLU(inplace=True),
            )
            self.layer1 = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=2, padding=0, stride=2, bias=False),
                nn.BatchNorm2d(64)
            )
            self.layer0 = nn.Sequential(
                nn.ConvTranspose2d(64, 3, kernel_size=8, padding=3, stride=2, bias=False),
                nn.BatchNorm2d(3),
                nn.Sigmoid()
            )


        # Sampling vector
        self.fc3 = nn.Linear(self.CNN_embed_dim, self.resnet18_dim)
        self.fc_bn3 = nn.BatchNorm1d(self.resnet18_dim)
        self.relu = nn.ReLU(inplace=True)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.fc_bn3(self.fc3(x)))
        #x = self.convTrans6(x)
        x = x.view(x.size(0), 512, 1, 1)
        #x = F.interpolate(x, scale_factor=4)
        x = F.adaptive_avg_pool2d(x, (7, 7))
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.layer0(x)
        return x




class ResNet50_VAE(nn.Module):
    def __init__(self, z_dim=64, h_dim=1024, drop_p = 0.3, pre_train=True):
        super().__init__()
        self.encoder = ResNet50_Encoder(fc_hidden1=h_dim, drop_p=drop_p, CNN_embed_dim=z_dim, pre_train=pre_train)
        self.decoder = ResNet50_Decoder(fc_hidden1=h_dim, CNN_embed_dim=z_dim)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x = self.decoder(z)
        return x, mean, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

   





