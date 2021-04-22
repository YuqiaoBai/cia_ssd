from model import ConvNet
import torch
from dataloader import coopDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from ops.roiaware_pool3d import roiaware_pool3d_cuda

def points_in_boxes_gpu(points, boxes):
    """
    :param points: (B, M, 3)
    :param boxes: (B, T, 7), num_valid_boxes <= T
    :return box_idxs_of_pts: (B, M), default background = -1
    """
    assert boxes.shape[0] == points.shape[0]
    assert boxes.shape[2] == 7 and points.shape[2] == 3
    batch_size, num_points, _ = points.shape

    box_idxs_of_pts = points.new_zeros((batch_size, num_points), dtype=torch.int).fill_(-1)
    roiaware_pool3d_cuda.points_in_boxes_gpu(boxes.contiguous(), points.contiguous(), box_idxs_of_pts)

    return box_idxs_of_pts

def loss_function(output, gt):

    return

def data_mask():
    pass


epoches = 2
batch_size = 2
learning_rate = 0.001

# Load data
train_data = coopDataset()
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Model
Coop = ConvNet()
Coop = Coop.float().to(device)
# Define optimizer and loss function
optimizer = optim.Adam(Coop.parameters(), lr=learning_rate)


# Train
for epoch in range(epoches):
    print('epoch:{}'.format(epoch))
    for step, cnn_input in enumerate(train_loader):
        output = Coop(cnn_input["points"].float().to(device))
        loss = loss_function(output, cnn_input["gt_boxes"])
        optimizer.zero_grad()
        optimizer.step()

        if step % 10 == 0:
            test_output = Coop(cnn_input)
            gt = 'to do'
            print('Epoch:', epoch, '|loss:', loss)



