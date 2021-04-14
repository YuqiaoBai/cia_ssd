from torch import nn
import torch
from losses import prepare_loss_weights, build_loss
from models.box_coders import GroundBoxBevGridCoder

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=1, bias=bias),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution without padding"""
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


class PIXOR(nn.Module):
    def __init__(self, mcfg, cfg, dcfg):
        super(PIXOR, self).__init__()
        self.cfg = mcfg
        self.dcfg = dcfg
        self.n_cls = len(dcfg.classes)
        self.box_coder = GroundBoxBevGridCoder(**dcfg.BOX_CODER)
        self.box_dim = self.box_coder.code_size
        self.loss_norm = mcfg.LOSS['loss_norm']
        self.loss_cls = build_loss(mcfg.LOSS['loss_cls'])
        self.loss_reg = build_loss(mcfg.LOSS['loss_bbox'])

        self.nms_type = mcfg.NMS['name']
        if self.nms_type=='normal':
            from ops.iou3d_nms.iou3d_nms_utils import nms_gpu
            self.nms = nms_gpu
        elif self.nms_type=='iou_weighted':
            from utils.box_torch_ops import rotate_weighted_nms
            self.nms = rotate_weighted_nms

        in_channel = int((dcfg.pc_range[-1] - dcfg.pc_range[2]) / dcfg.voxel_size[2])
        expansion = mcfg.expansion

        self.bottom = nn.Sequential(
            conv3x3(in_channel, 32),
            conv3x3(32, 32)
        )

        self.res_block2 = ResBlock(32, mcfg.c_blocks[0], mcfg.n_blocks[0])
        self.res_block3 = ResBlock(mcfg.c_blocks[0]*expansion, mcfg.c_blocks[1], mcfg.n_blocks[1])
        self.res_block4 = ResBlock(mcfg.c_blocks[1]*expansion, mcfg.c_blocks[2], mcfg.n_blocks[2])
        self.res_block5 = ResBlock(mcfg.c_blocks[2]*expansion, mcfg.c_blocks[3], mcfg.n_blocks[3])

        self.up_layer1 = conv1x1(mcfg.c_blocks[-1]*expansion, mcfg.up_channels[0])
        self.upsample1 = UpSample(mcfg.c_blocks[-2]*expansion, mcfg.up_channels[0], mcfg.up_channels[1])
        self.upsample2 = UpSample(mcfg.c_blocks[-3]*expansion, mcfg.up_channels[1], mcfg.up_channels[2])

        self.header = Header(mcfg.up_channels[-1], mcfg.header_channels, mcfg.head_layers, self.n_cls, self.box_dim)
        self.apply(weights_init)

    def forward(self, batch_data):
        x = batch_data.pop('bev_input')
        x = self.bottom(x)
        x = self.res_block2(x)
        c3 = self.res_block3(x)
        c4 = self.res_block4(c3)
        c5 = self.res_block5(c4)

        up = self.up_layer1(c5)
        up = self.upsample1(c4, up)
        up = self.upsample2(c3, up)

        cls_logits, reg_logits = self.header(up)
        batch_data.update({
            'cls_logits': cls_logits,
            'reg_logits': reg_logits
        })
        return batch_data

    def loss(self, batch_data):
        batch_size = batch_data['batch_size']
        # get predictions
        cls_preds = batch_data['cls_logits'].view(batch_size, self.n_cls, -1).permute(0, 2, 1)
        # cls_preds = torch.sigmoid(cls_preds)
        reg_preds = batch_data['reg_logits'].view(batch_size, self.box_dim, -1).permute(0, 2, 1)
        # get targets
        cls_tgts = batch_data['label_map'].view(batch_size, self.n_cls, -1).permute(0, 2, 1)
        reg_tgts = batch_data['regression_map'].view(batch_size, self.box_dim, -1).permute(0, 2, 1)
        labels = cls_tgts.reshape(batch_size, -1).bool().int()
        cls_weights, reg_weights, cared = prepare_loss_weights(labels, loss_norm=self.loss_norm,
                                                                    dtype=torch.float32)
        cls_loss = self.loss_cls(cls_preds, cls_tgts, weights=cls_weights)
        reg_loss = self.loss_reg(reg_preds, reg_tgts, weights=reg_weights)
        # angle_loss = reg_loss[:, ]
        reg_loss_reduced = self.loss_reg._loss_weight * reg_loss.sum() / batch_size   # 2.0, averaged on batch_size
        cls_loss_reduced = self.loss_cls._loss_weight * cls_loss.sum() / batch_size   # 1.0, average on batch_size
        loss = cls_loss_reduced + reg_loss_reduced

        ret = {
            'loss': loss,
            'cls_loss_reduced': cls_loss_reduced.detach().cpu().mean(),
            'reg_loss_reduced': reg_loss_reduced.detach().cpu().mean()
        }
        return ret

    def post_processing(self, batch_data, test_cfg):
        # get predictions
        cls_preds = batch_data['cls_logits'].permute(0, 2, 3, 1)
        reg_preds = batch_data['reg_logits'].permute(0, 2, 3, 1)
        gt_boxes = batch_data['gt_boxes']
        batch_scores, batch_boxes = self.box_coder.decode_torch(cls_preds, reg_preds,
                                                                test_cfg['score_threshold'], self.dcfg)
        batch_boxes_preds = []
        for boxes_preds, scores, gt in zip(batch_boxes, batch_scores, gt_boxes):
            box_ego = gt[(gt[:, :2]==0).all(dim=1), :]
            boxes_all = torch.cat([boxes_preds, box_ego], dim=0)
            scores_all = torch.cat([scores, torch.ones([1, 1], device=scores.device)], dim=0).view(-1)
            box_inds = self.nms(boxes_all, scores_all, test_cfg['nms_iou_threshold'],
                                pre_max_size=test_cfg['nms_pre_max_size'])
            batch_boxes_preds.append({
                'box_lidar': boxes_all[box_inds[0], :],
                'scores': scores_all[box_inds[0]]
            })
        return batch_boxes_preds


class UpSample(nn.Module):
    def __init__(self, in_planes_conv, in_planes_deconv, out_planes):
        super(UpSample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes_conv, out_planes, 1),
            nn.BatchNorm2d(out_planes)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_planes_deconv, out_planes, 2, stride=2, padding=0),
            nn.BatchNorm2d(out_planes)
        )
        self.out_layer = nn.Sequential(
            nn.ConvTranspose2d(out_planes, out_planes, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, branch_side, branch_bottom):
        x = self.relu(self.conv(branch_side) + self.deconv(branch_bottom))
        x = self.out_layer(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_planes, planes, n_blocks, expansion=4):
        super(ResBlock, self).__init__()

        self.expansion = expansion
        self.res_block = self._make_block(in_planes, planes, n_blocks, expansion)

    def _make_block(self, in_planes, planes, n_blocks, expansion):
        block = []
        block.append(BaseBlock(in_planes, planes, downsample=True, expansion=expansion))
        for _ in range(n_blocks - 1):
            block.append(BaseBlock(planes*self.expansion, planes, downsample=False, expansion=expansion))

        return nn.Sequential(*block)

    def forward(self, x):
        return self.res_block(x)


class BaseBlock(nn.Module):
    def __init__(self, in_planes, planes, downsample, expansion=4):
        super(BaseBlock, self).__init__()

        self.expansion = expansion
        self.residual_block = self._make_residual_block(in_planes, planes, downsample)
        self.identity = self._identity_layer(in_planes, planes, downsample)
        self.relu = nn.ReLU(inplace=True)

    def _make_residual_block(self, in_planes, planes, downsample=False):
        if downsample:
            out_layer = nn.Sequential(
                nn.Conv2d(planes, planes*self.expansion, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes*self.expansion),
            )
        else:
            out_layer = nn.Sequential(
                nn.Conv2d(planes, planes*self.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes*self.expansion),
            )
        return nn.Sequential(
                conv1x1(in_planes, planes),
                conv3x3(planes, planes),
                out_layer
            )

    def _identity_layer(self, in_planes, planes, downsample=False):
        if downsample:
            return nn.Sequential(
                nn.Conv2d(in_planes, planes*self.expansion, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes*self.expansion),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, planes*self.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes*self.expansion),
            )

    def forward(self, x):
        residual = self.residual_block(x)
        indentity = self.identity(x)

        return self.relu(residual + indentity)


class Header(nn.Module):
    def __init__(self, in_channels, mid_channels, n_layers, n_cls, n_reg):
        super(Header, self).__init__()
        layer_list = []
        layer_list.append(conv3x3(in_channels, mid_channels))
        for _ in range(n_layers - 1):
            layer_list.append(conv3x3(mid_channels, mid_channels))
        self.convs = nn.Sequential(*layer_list)
        self.cls_layer = nn.Conv2d(mid_channels, n_cls, 3, padding=1)
        self.reg_layer = nn.Conv2d(mid_channels, n_reg, 3, padding=1, bias=True)

    def forward(self, x):
        x = self.convs(x)
        cls_logits = self.cls_layer(x)
        reg_logits = self.reg_layer(x)

        return cls_logits, reg_logits

