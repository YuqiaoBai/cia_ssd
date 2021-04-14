import torch
import torch.nn as nn
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x1):
        x1 = self.up(x1)
        return x1


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.base_model = torchvision.models.resnet18(True)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(21, 64, kernel_size=(3, 3), stride=2, padding=(3, 3), bias=False))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=(3, 3), bias=False))
        self.layer3 =nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2, padding=(3, 3), bias=False))
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=2, padding=(3, 3), bias=False))
        self.decode4 = Decoder(512, 256)
        self.decode3 = Decoder(256, 128)
        self.decode2 = Decoder(128, 64)
        self.decode1 = Decoder(64, 20)

    def forward(self, input):
        e1 = self.layer1(input)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        f = self.layer4(e3)
        d3 = self.decode4(f, e3)
        d2 = self.decode3(d3, e2)
        d1 = self.decode2(d2, e1)
        out = self.decode1(d1)
        return out
