from datasets.kitti.augmentor import Augmentor
from datasets.kitti.processor import PointsProcessor
from utils.box_utils import *
from utils.common_utils import *
from datasets.kitti.kitti_base import KittiBaseDataset

import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
import copy


class Kitti3DDataset(KittiBaseDataset):
    def __init__(self, cfg=None, training=True):
        super().__init__(cfg=cfg, training=training)
        self.augmentor = Augmentor(self.root_path, self.cfg.AUGMENTOR, self.cls_dict) if self.training else None
        self.processor = PointsProcessor(self.cfg.DATA_PROCESSOR,
                                   point_cloud_range=self.point_cloud_range,
                                   training=self.training)
        if getattr(cfg, 'TARGET_ASSIGNER'):
            from datasets import assign_target
            self.target_assigner = assign_target.AssignTarget(cfg=self.cfg.TARGET_ASSIGNER, mode=self.mode)
        self.data_info = {
            'num_point_features': cfg.n_point_features,
            'grid_size': self.processor.grid_size,
            'point_cloud_range': self.point_cloud_range,
            'voxel_size': self.processor.voxel_size
        }

    def __getitem__(self, index):
        info = copy.deepcopy(self.kitti_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']

        points = self.get_lidar(sample_idx)
        points = points[:, :self.cfg.n_point_features]
        calib = self.get_calib(sample_idx)

        img_shape = info['image']['image_shape']
        if self.cfg.fov_points_only:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })      # {points(in rectified lidar cs), frame_id, calib, gt_names, gt_boxes(in lidar cs)}
            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        return data_dict

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)  in rectified lidar coordinate system
                frame_id:  string
                calib:
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...] in lidar coordinate system
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.cls_names for n in data_dict['gt_names']], dtype=np.bool_)

            data_dict = self.augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )
            if len(data_dict['gt_boxes']) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)

        if data_dict.get('gt_boxes', None) is not None:
            cls_names = []
            for names in self.cls_dict.values():
                cls_names.extend(names)
            selected = keep_arrays_by_name(data_dict['gt_names'], cls_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            cls_name_dic_reverse = {}
            for k, v in self.cls_dict.items():
                cls_name_dic_reverse.update({name: k for name in v})
            data_dict['gt_names'] = np.array([cls_name_dic_reverse[name] for name in data_dict['gt_names']])
            gt_classes = np.array([self.cls_names.index(n) + 1 for n in data_dict['gt_names']],
                                  dtype=np.int32)
            data_dict['gt_classes'] = gt_classes
            # gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)),
            #                          axis=1)   # set the last column as class label
            # data_dict['gt_boxes'] = gt_boxes

        data_dict = self.processor.forward(
            data_dict=data_dict
        )     #  {points, frame_id, calib, gt_boxes(label in last col)}

        # data_dict['gt_boxes'] = data_dict['gt_boxes'][:, :-1]
        # data_dict['gt_classes'] = data_dict['gt_boxes'][:, -1].astype(np.int32)
        data_dict['batch_types'] = {
            'points': 'gpu_float',
            'frame_id': 'gpu_none',
            'image_shape': 'gpu_none',
            'gt_boxes': 'gpu_float',
            'voxels': 'gpu_float',
            'voxel_coords': 'gpu_float',
            'voxel_num_points': 'gpu_float',
        }
        data_dict = self.target_assigner(data_dict)
        data_dict.get('gt_names', None)
        data_dict.pop('gt_classes', None)

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        batch_types = [data.pop("batch_types") for data in batch_list][0]
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['points_labels']:
                    max_gt = max([len(x) for x in val])
                    batch_points = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_points[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_points
                elif key in ['gt_boxes', 'encoded_gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ["frame", "gt_boxes", "positive_gt_id"]:
                    ret[key] = val
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        batch_types["batch_size"] = "cpu_none"
        ret["batch_types"] = batch_types
        return ret


if __name__=="__main__":
    from cfg.config import *
    cfg_file = "cfg/datasets/kitti_3d.yaml"
    cfg_from_yaml_file(cfg_file, cfg)
    dataset = KittiDataset(cfg=cfg, cls_names=['Car', 'Pedestrian', 'Cyclist'], training=True)
    sampler = None
    dataloader = DataLoader(
        dataset, batch_size=2, pin_memory=True, num_workers=4,
        shuffle=True, collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0
    )

    for data in dataloader:
        print(data.keys())