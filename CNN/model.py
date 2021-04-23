import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Conv2d(in_channels=21, out_channels=64, kernel_size=2, stride=2)
        self.layer2 = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        self.layer3 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.layer4 = nn.Conv2d(256, 512, kernel_size=2, stride=2)
        self.decode4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decode3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decode2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decode1 = nn.ConvTranspose2d(64, 20, kernel_size=2, stride=2)

    def forward(self, input):
        # print('in forward:', input.dtype) #(2,21,256,256)
        e1 = self.layer1(input) #(2,64,130,130)
        e2 = self.layer2(e1) #(2,128,67,67)
        e3 = self.layer3(e2) #(2,256,36,36)
        f = self.layer4(e3) #(2,512,20,20)

        d3 = self.decode4(f)
        d2 = self.decode3(d3)
        d1 = self.decode2(d2)
        out = self.decode1(d1)

        x = torch.ones(out.shape).cuda()
        y = torch.zeros(out.shape).cuda()
        out = torch.where(out > 0, x, y)
        # print('greater than 0:', torch.sum(out > 0))
        # print('smaller than 0:', torch.sum(out < 0))
        return out


class PointsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, added_points, original_point):
        added_points = torch.sum(added_points, axis=1)
        convert_points = (added_points == 0)
        ego_points = original_point[:, 1, :, :]
        a = torch.sum(convert_points.cuda() != ego_points.cuda())
        b = torch.sum(ego_points)
        loss = a/b
        return loss
