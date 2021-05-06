from model import ConvNet
from loss import PointsLoss
import torch
from dataloader import coopDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt


def data_mask(mask, points):
    # remove map layer
    points = points[:, 1:, :, :]
    fused_points = mask * points
    return fused_points


epoches = 5
batch_size = 2
learning_rate = 0.001

# Load data
train_data = coopDataset()
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=6)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Model
Coop = ConvNet()
Coop = Coop.float().cuda()

# Define optimizer and loss function
optimizer = optim.SGD(Coop.parameters(), lr=learning_rate)

# Train
for epoch in range(epoches):
    print('epoch:{}'.format(epoch))
    for step, cnn_input in enumerate(train_loader):
        output = Coop(cnn_input["points"].float().cuda())
        # select points
        points_fused = data_mask(output, cnn_input["points"].float().cuda())
        loss_function = PointsLoss()
        loss = loss_function(points_fused.cuda(), cnn_input["points"].cuda(), cnn_input["gt_boxes"].cuda(), cnn_input["tf_ego"].cuda())
        loss.requires_grad = True
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print('Epoch:', epoch, '|loss:', loss)



