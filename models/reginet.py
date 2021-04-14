from torch import nn
from models import *
import torch
from losses import build_loss


class RegiNet(nn.Module):
    def __init__(self, mcfg, cfg, dcfg):
        super(RegiNet, self).__init__()
        self.vfe = MeanVFE(dcfg.n_point_features)
        self.spconv_block = VoxelBackBone8x(mcfg.SPCONV,
                                            input_channels=dcfg.n_point_features,
                                            grid_size=dcfg.grid_size)
        self.map_to_bev = HeightCompression(mcfg.MAP2BEV)
        self.conv = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 128, 22, stride=1, padding=0)
        )
        self.head = nn.Linear(128, 3)
        self.loss_fn = build_loss(mcfg.LOSS['loss_reg'])
        self.noise_std = torch.tensor(dcfg.gps_noise_std)[[0, 1, 5]]

    def forward(self, batch_dict):
        src_input = {'voxels': batch_dict['src_voxels'],
                     'voxel_num_points': batch_dict['src_voxel_num_points'],
                     'voxel_coords': batch_dict['src_voxel_coords'],
                     'batch_size': batch_dict['batch_size']}
        features_src = self.features(src_input)
        tgt_input = {'voxels': batch_dict['tgt_voxels'],
                     'voxel_num_points': batch_dict['tgt_voxel_num_points'],
                     'voxel_coords': batch_dict['tgt_voxel_coords'],
                     'batch_size': batch_dict['batch_size']}
        features_tgt = self.features(tgt_input)
        # features = torch.cat([features_src, features_tgt], dim=1)
        out = self.head(features_src - features_tgt)
        batch_dict['preds'] = out.squeeze()

        return batch_dict

    def features(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.spconv_block(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)
        out = self.conv(batch_dict['spatial_features'])
        return out.view(batch_dict['batch_size'], 128)

    def loss(self, batch_dict):
        reg_preds = batch_dict['preds']
        reg_tgts = batch_dict['reg_target']
        loss = self.loss_fn._loss_weight * self.loss_fn(reg_preds, reg_tgts).sum() / reg_tgts.shape[0]
        return {'loss': loss}

    def post_processing(self, batch_dict):
        src_points = batch_dict['src_points']
        # tgt_points = batch_dict['tgt_points']
        preds = batch_dict['preds'] * self.noise_std[None, :].to(src_points.device)
        batch_size = preds.shape[0]
        src_points = [src_points[src_points[:, 0]==i] for i in range(batch_size)]
        shifts = preds[:, :2].view([batch_size, 1, 2])
        points_rot = []
        for b, points in enumerate(src_points):
            cur_points = points[(points[:, 1:]!=0).all(dim=1), 1:]
            cur_points[:, :2] = cur_points[:, :2] - shifts[b]
            angle = - preds[b, 2] / 180 * torch.acos(torch.zeros(1)).to(preds.device)
            cosa = torch.cos(angle)
            sina = torch.sin(angle)
            rot_mat = torch.zeros((3, 3), device=preds.device)
            rot_mat[0, 0] = cosa
            rot_mat[0, 1] = sina
            rot_mat[1, 0] = -sina
            rot_mat[1, 1] = cosa
            rot_mat[2, 2] = 1.0
            points_rot.append(torch.matmul(cur_points, rot_mat))

        return preds, points_rot
