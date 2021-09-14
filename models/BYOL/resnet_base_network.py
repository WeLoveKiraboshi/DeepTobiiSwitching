import torchvision.models as models
import torch
from models.BYOL.mlp_head import MLPHead


class ResNet18(torch.nn.Module):
    def __init__(self, network='resnet18', is_pretrained=False, hidden_size=512, projection_size=64):
        super(ResNet18, self).__init__()
        if network == 'resnet18':
            resnet = models.resnet18(pretrained=is_pretrained)
        elif network == 'resnet50':
            resnet = models.resnet50(pretrained=is_pretrained)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead(in_channels=resnet.fc.in_features, mlp_hidden_size=hidden_size, projection_size=projection_size)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)
