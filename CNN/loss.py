import torch
import torch.nn as nn
from ops.roiaware_pool3d import roiaware_pool3d_cuda
from utils import common_utils
import matplotlib.pyplot as plt
from torch.nn import Sequential
from models.utils import xavier_init, build_norm_layer
from torch.nn import functional as F

class PointsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, added_points, original_points, boxes, tf_ego):
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
        # predicted_points = torch.dot(predicted_points, torch.linalg.inv(tf_ego).float())
        original_points = torch.sum(original_points, dim=1)
        # original_points= torch.dot(original_points, torch.linalg.inv(tf_ego).float())

        # =============================vis============================================
        plt.subplot(1,2,1)
        plt.imshow(predicted_points.cpu().data.numpy()[0, :, :])
        plt.subplot(1,2,2)
        plt.imshow(original_points.cpu().data.numpy()[0, :, :])
        plt.savefig("temp")
        # ==============================================================================
        # for every frame
        batch_size = original_points.shape[0]
        iou = 0

        predicted_points_idx = torch.nonzero(predicted_points, as_tuple=False)
        original_points_idx = torch.nonzero(original_points, as_tuple=False)
        boxes_frame = boxes
        # set z = 0
        y1 = torch.ones(predicted_points_idx.shape[0], 1).cuda()
        predicted_points_idx = torch.cat((predicted_points_idx, y1), dim=1)*0.8
        y2 = torch.ones(original_points_idx.shape[0], 1).cuda()
        original_points_idx = torch.cat((original_points_idx, y2), dim=1)*0.8
        # all zeros? boxes coordinate problem
        idx_object = self.points_in_boxes_gpu(original_points_idx, boxes_frame)
        idx_predict = self.points_in_boxes_gpu(predicted_points_idx, boxes_frame)
        # get points
        n_object = []
        n_predict = []
        for i in range(20):
            p_idx = idx_object[i, :]
            nz = torch.nonzero(p_idx)
            n_object.append(original_points_idx[nz])
            p_idx = idx_predict[i, :]
            nz = torch.nonzero(p_idx)
            n_predict.append(predicted_points_idx[nz])
        # in grid
        n_object_grid = torch.zeros(256, 256)
        n_predict_grid = torch.zeros(256, 256)
        x = torch.Tensor(n_object)
        y = torch.Tensor(n_object)
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

    # def points_in_boxes_gpu(self, points, boxes):
    #     """
    #     Args:
    #         points: (num_points, 3)
    #         boxes: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps
    #     Returns:
    #         point_indices: (N, num_points)
    #     """
    #     assert boxes.shape[-1] == 7
    #     assert points.shape[-1] == 3
    #     points, is_numpy = common_utils.check_numpy_to_torch(points)
    #     boxes, is_numpy = common_utils.check_numpy_to_torch(boxes)
    #     point_indices = points.new_zeros((boxes.shape[0], points.shape[0]), dtype=torch.int)
    #     roiaware_pool3d_cuda.points_in_boxes_gpu(boxes.float().contiguous(), points.float().contiguous(), point_indices)
    #     return point_indices.numpy() if is_numpy else point_indices

    def points_in_boxes_gpu(self, points, boxes):
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