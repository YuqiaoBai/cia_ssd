import torch
import torch.nn as nn
from ops.roiaware_pool3d import roiaware_pool3d_cuda
from utils import common_utils
from torch.nn import Sequential
from models.utils import xavier_init, build_norm_layer
from torch.nn import functional as F

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

        x = torch.ones(out.shape).cuda()
        y = torch.zeros(out.shape).cuda()
        out = torch.where(out > 0, x, y)
        return out


class PointsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, added_points, original_points, boxes):
        """
        :param added_points: prdicted points
        :param original_point:
        :param boxes:
        :return:
        """
        original_points = original_points[:, 1:, :, :]
        # matrix to 3d points
        # sum every coop

        predicted_points = torch.sum(added_points, dim=1)
        original_points = torch.sum(original_points, dim=1)
        # for every frame
        batch_size = original_points.shape[0]
        iou = 0
        for i in range(batch_size):
            predicted_points_idx = torch.nonzero(predicted_points[i, :, :], as_tuple=False)
            original_points_idx = torch.nonzero(original_points[i, :, :], as_tuple=False)
            boxes_frame = boxes[i, :, :]
            # set z = 0
            y1 = torch.zeros(predicted_points_idx.shape[0], 1).cuda()
            predicted_points_idx = torch.cat((predicted_points_idx, y1), dim=1)*0.8
            y2 = torch.zeros(original_points_idx.shape[0], 1).cuda()
            original_points_idx = torch.cat((original_points_idx, y2), dim=1)*0.8
            # all zeros? boxes coordinate problem
            idx_object = self.points_in_boxes_gpu(original_points_idx, boxes_frame)
            idx_predict = self.points_in_boxes_gpu(predicted_points_idx, boxes_frame)
            # get points
            for i in range(20):
                p_idx = idx_object[i, :]
                nz = torch.nonzero(p_idx)
                n_object = original_points_idx[nz]
            print(torch.max(idx_predict))
            idx_p = torch.nonzero(idx_predict)
            n_predict = predicted_points_idx[idx_p[:, 0], idx_p[:, 1]]
            # in grid
            n_object_grid = torch.zeros(256, 256)
            n_predict_grid = torch.zeros(256, 256)
            x = n_object[:, 0]
            y = n_object[:, 1]
            inds_x = (x / 0.8 + 256 / 2)
            inds_y = (y / 0.8 + 256 / 2)
            n_object_grid[inds_x, inds_y] = 1

            x1 = n_predict[:, 0]
            y1 = n_predict[:, 1]
            inds_x = (x1 / 0.8 + 256 / 2)
            inds_y = (y1 / 0.8 + 256 / 2)
            n_predict_grid[inds_x, inds_y] = 1
            for j in range(boxes_frame.shape[0]):
                intersection = (n_object_grid & n_predict_grid).float()
                union = (n_object_grid | n_predict_grid).float()
                iou = iou + intersection/union
                print(i, 'iou:', iou)
        iou = iou/batch_size
        return iou

    def points_in_boxes_gpu(self, points, boxes):
        """
        Args:
            points: (num_points, 3)
            boxes: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps
        Returns:
            point_indices: (N, num_points)
        """
        assert boxes.shape[-1] == 7
        assert points.shape[-1] == 3
        points, is_numpy = common_utils.check_numpy_to_torch(points)
        boxes, is_numpy = common_utils.check_numpy_to_torch(boxes)
        point_indices = points.new_zeros((boxes.shape[0], points.shape[0]), dtype=torch.int)
        roiaware_pool3d_cuda.points_in_boxes_gpu(boxes.float().contiguous(), points.float().contiguous(), point_indices)
        return point_indices.numpy() if is_numpy else point_indices