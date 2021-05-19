import torch
import torch.nn as nn
from ops.roiaware_pool3d import roiaware_pool3d_cuda
import matplotlib.pyplot as plt
from cnn_utils import draw_box_plt
from utils import common_utils

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
        y1 = torch.zeros(2, predicted_points_idx.shape[1], 1).cuda()
        predicted_points_idx = torch.cat((predicted_points_idx, y1), dim=2)*0.8
        y2 = torch.zeros(2, original_points_idx.shape[1], 1).cuda()
        original_points_idx = torch.cat((original_points_idx, y2), dim=2)*0.8

        # =============================vis============================================
        ax = plt.figure(figsize=(8, 8)).add_subplot(1, 1, 1)
        points = predicted_points_idx.cpu()
        ax.plot(points[0,:, 0], points[0,:, 1], 'b.', markersize=0.5)
        boxes_frame[:,:,0:2] = boxes_frame[:,:,0:2] - ego_loc[:,None,:]
        ax = draw_box_plt(boxes_frame[0,:,:], ax, color='green')
        # ax = draw_box_plt(pred_boxes[0], ax, color='red')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('temp.png')
        plt.close()
        # ==============================================================================
        for i in range(batch_size):
            idx_original = self.points_in_boxes_gpu(original_points_idx[i,:,:].float().unsqueeze(0), boxes_frame[i,:,:].float().unsqueeze(0))
            idx_predict = self.points_in_boxes_gpu(predicted_points_idx[i,:,:].float().unsqueeze(0), boxes_frame[i,:,:].float().unsqueeze(0))

            o_idx = torch.where(idx_original != -1)
            p_idx = torch.where(idx_predict != -1)
            n_object = original_points_idx[o_idx]
            n_predict = predicted_points_idx[p_idx]

            # in grid
            n_object_grid = torch.zeros(256, 256)
            n_predict_grid = torch.zeros(256, 256)
            x =n_object[:, 0]
            y =n_object[:, 1]
            inds_x = (x / 0.8 + 256 / 2).long()
            inds_y = (y / 0.8 + 256 / 2).long()
            n_object_grid[inds_x, inds_y] = 1
            n_object_grid = n_object_grid.bool()

            x1 = n_predict[:, 0]
            y1 = n_predict[:, 1]
            inds_x = (x1 / 0.8 + 256 / 2).long()
            inds_y = (y1 / 0.8 + 256 / 2).long()
            n_predict_grid[inds_x, inds_y] = 1
            n_predict_grid = n_predict_grid.bool()
            intersection = ((n_object_grid & n_predict_grid)==True).sum().float()
            union = ((n_object_grid | n_predict_grid)==True).sum().float()
            iou = intersection/union
        return iou

    def points_in_boxes_gpu(self, points, boxes):
        """Find points that are in boxes (CUDA)
        Args:
        points (torch.Tensor): [B, M, 3], [x, y, z] in LiDAR coordinate
        boxes (torch.Tensor): [B, T, 7],
            num_valid_boxes <= T, [x, y, z, w, l, h, ry] in LiDAR coordinate,
            (x, y, z) is the bottom center
        Returns:
        box_idxs_of_pts (torch.Tensor): (B, M), default background = -1
        """
        assert boxes.shape[0] == points.shape[0]
        assert boxes.shape[2] == 7 and points.shape[2] == 3
        batch_size, num_points, _ = points.shape

        box_idxs_of_pts = points.new_zeros((batch_size, num_points), dtype=torch.int).fill_(-1)
        # If manually put the tensor 'points' or 'boxes' on a device
        # which is not the current device, some temporary variables
        # will be created on the current device in the cuda op,
        # and the output will be incorrect.
        # Therefore, we force the current device to be the same
        # as the device of the tensors if it was not.
        # Please refer to https://github.com/open-mmlab/mmdetection3d/issues/305
        # for the incorrect output before the fix.
        points_device = points.get_device()
        assert points_device == boxes.get_device(), \
            'Points and boxes should be put on the same device'
        if torch.cuda.current_device() != points_device:
            torch.cuda.set_device(points_device)
        roiaware_pool3d_cuda.points_in_boxes_gpu(boxes.contiguous(), points.contiguous(), box_idxs_of_pts)

        return box_idxs_of_pts

    def points_in_boxes_cpu(self, points, boxes):
        """
        Args:
            points: (num_points, 3)
            boxes: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps
        Returns:
            point_indices: (N, num_points)
        """
        assert boxes.shape[1] == 7
        assert points.shape[1] == 3
        points, is_numpy = common_utils.check_numpy_to_torch(points)
        boxes, is_numpy = common_utils.check_numpy_to_torch(boxes)

        point_indices = points.new_zeros((boxes.shape[0], points.shape[0]), dtype=torch.int)
        roiaware_pool3d_cuda.points_in_boxes_gpu(boxes.float().contiguous(), points.float().contiguous(), point_indices)

        return point_indices.numpy() if is_numpy else point_indices