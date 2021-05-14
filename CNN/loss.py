import torch
import torch.nn as nn
from ops.roiaware_pool3d import roiaware_pool3d_cuda
from utils import common_utils
import matplotlib.pyplot as plt
from torch.nn import Sequential
from models.utils import xavier_init, build_norm_layer
from torch.nn import functional as F
import numpy as np
from cnn_utils import draw_box_plt

class PointsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, added_points, original_points, boxes, ego_loc):
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
        # ==========================================vis============================================
        plt.subplot(1,2,1)
        plt.imshow(predicted_points[0,:,:].cpu().detach().numpy())
        plt.subplot(1,2,2)
        plt.imshow(original_points[0,:,:].cpu().detach().numpy())
        plt.savefig('diff.png')
        plt.close()

        # =========================================================================================

        # for every frame
        batch_size = original_points.shape[0]
        p = []
        o = []
        for i in range(batch_size):
            p.append(torch.nonzero(predicted_points[i,:,:], as_tuple=False))
            o.append(torch.nonzero(original_points[i,:,:], as_tuple=False))
        boxes_frame = boxes
        # fill with zeros
        if len(p[0]) > len(p[1]):
            p[1]=torch.cat((p[1],torch.zeros(len(p[0]) - len(p[1]),2).cuda()),dim=0)
        else:
            p[0] = torch.cat((p[0], torch.zeros(len(p[1]) - len(p[0]),2).cuda()),dim=0)
        if len(o[0]) > len(o[1]):
            o[1] = torch.cat((o[1], torch.zeros(len(o[0]) - len(o[1]), 2).cuda()), dim=0)
        else:
            o[0] = torch.cat((o[0], torch.zeros(len(o[1]) - len(o[0]), 2).cuda()), dim=0)
        predicted_points_idx = torch.stack(p, 0)-128
        original_points_idx = torch.stack(o, 0)-128
        # set z = 1
        y1 = torch.ones(2, predicted_points_idx.shape[1], 1).cuda()
        predicted_points_idx = torch.cat((predicted_points_idx, y1), dim=2)*0.8
        y2 = torch.ones(2, original_points_idx.shape[1], 1).cuda()
        original_points_idx = torch.cat((original_points_idx, y2), dim=2)*0.8

        # =============================vis============================================
        ax = plt.figure(figsize=(8, 8)).add_subplot(1, 1, 1)
        points = predicted_points_idx.cpu()
        ax.plot(points[0,:, 0], points[0,:, 1], 'b.', markersize=0.5)
        boxes[:,:,0:2] = boxes[:,:,0:2] - ego_loc[:,None,:]
        ax = draw_box_plt(boxes[0,:,:], ax, color='green')
        # ax = draw_box_plt(pred_boxes[0], ax, color='red')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('temp.png')
        plt.close()
        # ==============================================================================

        idx_original = self.points_in_boxes_gpu(original_points_idx.float(), boxes_frame.float())
        idx_predict = self.points_in_boxes_gpu(predicted_points_idx.float(), boxes_frame.float())
        # get points
        n_object = []
        n_predict = []
        for i in range(batch_size):
            o_idx = idx_original[i, :]
            nz1 = torch.nonzero(o_idx)
            n_object.append(original_points_idx[nz1])
            p_idx = idx_predict[i, :]
            nz2 = torch.nonzero(p_idx)
            n_predict.append(predicted_points_idx[nz2])
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