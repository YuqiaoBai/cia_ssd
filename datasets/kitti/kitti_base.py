from datasets.kitti.calibration import Calibration
from datasets.kitti import object3d_kitti

import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import pickle
import io
import cv2


class KittiBaseDataset(Dataset):
    def __init__(self, cfg=None, training=True):
        super().__init__()
        self.cfg = cfg
        self.cls_names = list(cfg.class_dict.keys())
        self.cls_dict = cfg.class_dict
        self.training = training
        self.root_path = Path(self.cfg.data_path)

        self.point_cloud_range = np.array(self.cfg.pc_range, dtype=np.float32)

        self.split = 'train' if self.mode=='train' else 'val'
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.kitti_infos = []
        self.info_path = {
            'train': ['kitti_infos_train.pkl'],
            'test': ['kitti_infos_val.pkl']
        }
        self.include_kitti_data(self.mode)

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def include_kitti_data(self, mode):
        kitti_infos = []

        for info_path in self.info_path[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

    def get_image(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return cv2.imread(str(img_file))

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return Calibration(calib_file)

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib, img=None):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
        inds = np.where(pts_valid_flag)[0]
        if img is None:
            return pts_valid_flag
        else:
            flagged_points_colors = img[pts_img[inds, 1].astype(np.int), pts_img[inds, 0].astype(np.int), :]
            return pts_valid_flag, flagged_points_colors

    def __len__(self):
        return len(self.kitti_infos)

    def __getitem__(self, index):

        return NotImplementedError


