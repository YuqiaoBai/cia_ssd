from torch import nn
import torch
import torch.nn.functional as F
from ops.roiaware_pool3d import roiaware_pool3d_utils
from utils import box_utils
from losses.cross_entropy import cross_entropy2d_binary

class KeypointWeighting(nn.Module):

    def __init__(self, cfg, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.cfg = cfg
        self.mlp = MLP(input_dim, hidden_dim, 1, 2)
        self.out_layer = nn.Linear(640, output_dim)
        self.loss_fn = cross_entropy2d_binary
        self.step = 0

    # @torch.no_grad()
    def get_keypoints_label(self, points, gt_boxes):
        extended_gt_boxes = box_utils.enlarge_box3d(gt_boxes.view(-1, gt_boxes.shape[-1]),
                                                    extra_width=[0.1, 0.1, 0.1]
                                                    ).view(*gt_boxes.shape)
        labels = roiaware_pool3d_utils.points_in_boxes_gpu(points[..., :3],
                                                           extended_gt_boxes[..., :7]) # B, N
        labels[torch.where(labels>=0)] = 1
        labels[torch.where(labels<0)] = 0

        return labels.float().flatten().unsqueeze(-1)


    def forward(self, data_dict):
        self.step += 1
        if self.cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = data_dict['point_features_before_fusion']
        else:
            point_features = data_dict['point_features']
        batch_size = data_dict['point_features'].shape[0]
        points_preds_cls = self.mlp(point_features)
        points_preds_scores = torch.sigmoid(points_preds_cls)
        point_features = point_features * points_preds_scores
        point_features = self.out_layer(point_features)
        point_features = point_features.view(batch_size, -1, point_features.shape[-1])
        selected = points_preds_scores.view(batch_size, -1) > 0.4
        if self.step % 1 == 0:
            wandb.log({'keypoints_scores': wandb.Histogram(points_preds_scores.cpu().data.numpy())})
            wandb.log({'selected_keypoints': selected.sum() // batch_size})
        sizes = [m.sum() for m in selected]

        selected_keypoints = torch.zeros((batch_size, max(sizes), point_features.shape[-1]),
                                         dtype=point_features.dtype, device=point_features.device)
        point_coords = data_dict['point_coords']
        selected_coords = torch.zeros((batch_size, max(sizes), 3),
                                      dtype=point_coords.dtype, device=point_coords.device)
        for i in range(batch_size):
            selected_keypoints[i, :sizes[i]] = point_features[i, selected[i]]
            selected_coords[i, :sizes[i]] = point_coords[i, selected[i]]
        data_dict['point_features'] = selected_keypoints

        if self.training:
            labels = self.get_keypoints_label(point_coords, data_dict['gt_boxes'][..., :7])
            tp = torch.mul(points_preds_scores > 0.5, labels==1).sum().float()
            recall = tp / (labels==1).sum().float()
            if self.step % 1 == 0:
                wandb.log({'keypoints_in_boxes': labels.sum() / batch_size,
                           'kpw_recall': recall.item()})
            loss = self.loss_fn(points_preds_cls, labels, weight=50.0)
            data_dict['kpw_loss'] = loss
            point_gt_idx = labels.reshape(2, -1).bool()
            data_dict['kpoints_obj'] = [point_coords[i, point_gt_idx[i]] for i in range(batch_size)]
        data_dict['selected_kpoints'] = selected_coords
        # mask to track non-valid values in the batch of less chosen points
        data_dict['point_mask'] = ((selected_coords==0).sum(dim=-1) != 0)
        return data_dict