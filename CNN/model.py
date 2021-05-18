import torch
import torch.nn as nn
from torch.nn import Sequential


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = Sequential(
            nn.Conv2d(in_channels=21, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False, ),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.ReLU(),
        )
        self.layer2 = Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False, ),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.ReLU(),
        )
        self.layer3 = Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False, ),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.ReLU(),
        )
        self.layer4 = Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=False, ),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.ReLU(),
        )
        self.decode4 = Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.decode3 = Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.decode2 = Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.decode1 = Sequential(
            nn.ConvTranspose2d(64, 20, kernel_size=2, stride=2),
            nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

    def forward(self, input):
        # print('in forward:', input.dtype) #(2,21,256,256)
        e1 = self.layer1(input) #(2,64,128,128)
        e2 = self.layer2(e1) #(2,128,64,64)
        e3 = self.layer3(e2) #(2,256,32,32)
        f = self.layer4(e3) #(2,512,16,16)

        d3 = self.decode4(f)
        d2 = self.decode3(d3)
        d1 = self.decode2(d2)
        out = self.decode1(d1)

        # x = torch.ones(out.shape).cuda()
        # y = torch.zeros(out.shape).cuda()
        # out = torch.where(out > 0, x, y)

        out[torch.where(out >= 0)] = 1
        out[torch.where(out < 0)] = 0
        return out


