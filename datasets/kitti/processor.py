
from functools import partial
from utils.box_utils import *
from utils.common_utils import *
from ops.roiaware_pool3d import roiaware_pool3d_utils


class PointsProcessor():
    def __init__(self, processor_configs, point_cloud_range, training):
        self.cfg = processor_configs
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []
        self.occupancy_mask = None
        self.boxes_mean = np.zeros((6))
        self.boxes_std = np.ones((6))
        for cur_name, cur_cfg in self.cfg.items():
            if cur_cfg=='encode_gt_boxes':
                self.boxes_mean = np.array(cur_cfg['mean'])
                self.boxes_std = np.array(cur_cfg['std'])
            cur_processor = getattr(self, cur_name)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        """
        Can be regarded as pass-through filter.
        """
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)
        mask = mask_points_by_range(data_dict['points'], self.point_cloud_range)
        data_dict['points'] = data_dict['points'][mask]
        if data_dict.get('gt_boxes', None) is not None and config['remove_outside_boxes'] and self.training:
            mask = mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
            data_dict['gt_names'] = data_dict['gt_names'][mask]
            data_dict['gt_classes'] = data_dict['gt_classes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config['shuffle_enabled'][self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None, voxel_generator=None):
        if data_dict is None:
            try:
                from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            except:
                from spconv.utils import VoxelGenerator

            voxel_generator = VoxelGenerator(
                voxel_size=config['voxel_size'],
                point_cloud_range=self.point_cloud_range,
                max_num_points=config['max_points_per_voxel'],
                max_voxels=config['max_number_of_voxels'][self.mode]
            )
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config['voxel_size'])
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config['voxel_size']
            return partial(self.transform_points_to_voxels, voxel_generator=voxel_generator)

        points = data_dict['points']
        voxel_output = voxel_generator.generate(points)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        # if not data_dict['use_lead_xyz']:
        #     voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        """
        To make sure that we get equal number of points. The stragegy is:
        1. Too many points?
            - randomly get rid of some of those near range points(range smaller than 40)
              if far range points are too less and sparse
            - randomly sample both far and near points if they are both enough and dense
        2. Too few points?
            - sample with replacement
        """
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else:
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def encode_gt_boxes(self, data_dict=None, config=None):
        """
        transform gt boxes to ratio format to make the training PVT more numerical stable
        x_out = x_in / (cloud_range_x_max - cloud_range_x_max)
        dx_out = log(dx_in)
        r_out = r_in / pi
        """
        if data_dict is None:
            return partial(self.encode_gt_boxes, config=config)
        gt_boxes = data_dict['gt_boxes']
        gt_boxes_encoded = np.zeros((gt_boxes.shape[0], 9), dtype=gt_boxes.dtype)
        gt_boxes_encoded[:, -1] = gt_boxes[:, -1]
        pc_size = self.point_cloud_range[3:] - self.point_cloud_range[:3]
        for i in range(3):
            gt_boxes_encoded[:, i] = (gt_boxes[:, i] - self.point_cloud_range[i]) / pc_size[i]
        gt_boxes_encoded[:, 3:6] = np.log(np.clip(gt_boxes[:, 3:6], a_min=1e-6, a_max=None))
        gt_boxes_encoded[:, 6] = np.sin(gt_boxes[:, 6])
        gt_boxes_encoded[:, 7] = np.cos(gt_boxes[:, 6])
        gt_boxes_encoded[:, :-1] = (gt_boxes_encoded[:, :-1] - self.boxes_mean[None, :]) / self.boxes_std[None, :]
        data_dict['encoded_gt_boxes'] = gt_boxes_encoded
        return data_dict

    def get_pointwise_labels(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.get_pointwise_labels, config=config)
        gt_boxes = data_dict['gt_boxes']
        points = data_dict['points']
        points_mask = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, :3], gt_boxes[:, :7])
        classes, inds = np.where(points_mask)
        labels = np.zeros((points.shape[0], 1))
        labels[inds] = gt_boxes[classes, -1:]

        data_dict['points_labels'] = np.concatenate([points, labels], axis=1)
        data_dict.pop('points')

        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict


class BEVProcessor():
    def __init__(self, processor_configs, point_cloud_range, training):
        self.cfg = processor_configs
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []
        self.occupancy_mask = None
        for cur_cfg in self.cfg:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        """
        Can be regarded as pass-through filter.
        """
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)
        mask = mask_points_by_range(data_dict['points'], self.point_cloud_range)
        data_dict['points'] = data_dict['points'][mask]
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def get_grid_maps(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.get_grid_maps, config=config)
        data_dict['grid_maps'] = {}
        for feature in config.FEATURES:
            data_dict['grid_maps'][feature] = getattr(self, 'get_'+ feature +'_map') (data_dict, config)
        data_dict['grid_maps'] = np.concatenate(list(data_dict['grid_maps'].values()), axis=0)
        data_dict['grid_maps_size'] = tuple(data_dict['grid_maps'].shape[-2:])
        boxes_bev = box3d_to_2d_bev_pixel(data_dict['gt_boxes'],
                                          self.point_cloud_range, config.MAP_RESOLUTION)
        data_dict['gt_boxes_bev'] = np.concatenate((boxes_bev, data_dict['gt_boxes'][:, -1:]),axis=1)
        return data_dict

    def get_detection_map(self, data_dict, config):
        points = data_dict['points']
        map_detection = np.zeros(config.MAP_SIZE, dtype=np.int32)
        for p in points:
            row = min(int(np.floor((p[0] - self.point_cloud_range[0]) / config.MAP_RESOLUTION)), config.MAP_SIZE[0]-1)
            col = min(int(np.floor((p[1] - self.point_cloud_range[1]) / config.MAP_RESOLUTION)), config.MAP_SIZE[1]-1)
            map_detection[row, col] += 1  # some points have 0 intensity
        self.occupancy_mask = map_detection > 0
        return map_detection

    def get_height_diff_map(self, data_dict, config):
        points = data_dict['points']
        map_min = np.ones(config.MAP_SIZE, dtype=np.float32) * 1000
        map_max = - np.ones(config.MAP_SIZE, dtype=np.float32) * 1000
        for p in points:
            row = min(int(np.floor((p[0] - self.point_cloud_range[0]) / config.MAP_RESOLUTION)), config.MAP_SIZE[0]-1)
            col = min(int(np.floor((p[1] - self.point_cloud_range[1]) / config.MAP_RESOLUTION)), config.MAP_SIZE[1]-1)
            map_min[row, col] = min(p[2], map_min[row, col])
            map_max[row, col] = max(p[2], map_max[row, col])
        map_diff = map_max - map_min + 1 # 1 means max==min in this cell, only 1 point falls in this cell
        if self.occupancy_mask is None:
            self.occupancy_mask = map_diff < -1000
        map_diff[self.occupancy_mask==False] = 0
        return map_diff

    def get_intensity_map(self, data_dict, config):
        points = data_dict['points']
        energy = np.zeros(config.MAP_SIZE, dtype=np.float32)
        map_detection = np.zeros(config.MAP_SIZE, dtype=np.int32)
        for p in points:
            row = min(int(np.floor((p[0] - self.point_cloud_range[0]) / config.MAP_RESOLUTION)), config.MAP_SIZE[0]-1)
            col = min(int(np.floor((p[1] - self.point_cloud_range[1]) / config.MAP_RESOLUTION)), config.MAP_SIZE[1]-1)
            energy[row, col] += p[3]
            map_detection[row, col] += 1
        map_intensity = energy / (map_detection + np.array(map_detection==0, dtype=np.int32))
        return map_intensity[None, ...]

    def get_occlusion_height_map(self, data_dict, config):
        size = config.MAP_SIZE
        points = data_dict['points']
        occlusion_mask = np.ones((int(np.ceil(np.sqrt(size[0] ** 2 + (0.5 * size[1]) ** 2))),
                                    int(180 / config.AZIMUTH_RESOLUTION)), dtype=np.float32) * self.point_cloud_range[2]
        for p in points:
            range_idx = int(np.floor(np.linalg.norm(p[0:2]) / config.MAP_RESOLUTION))
            angle_idx = int(np.floor((np.arctan2(p[1], p[0]) + 0.5 * np.pi) / np.pi * 180 / config.AZIMUTH_RESOLUTION))
            for r, height in enumerate(occlusion_mask[range_idx:, angle_idx]):
                if height < p[2]:
                    occlusion_mask[range_idx + r, angle_idx] = p[2]
        map_occlusion_height = np.zeros(size, dtype=np.float32)
        for i in range(size[0]):
            for j in range(size[1]):
                x = i + 0.5
                y = (j - size[1] * 0.5) + 0.5
                range_idx = int(np.floor(np.sqrt(x**2 + y**2)))
                angle_idx = int(np.floor((np.arctan2(y, x) + 0.5 * np.pi) / np.pi * 180 / config.AZIMUTH_RESOLUTION))
                map_occlusion_height[i, j] = occlusion_mask[range_idx, angle_idx]
        return map_occlusion_height

    def get_pixel_labels(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.get_pixel_labels, config=config)
        boxes = data_dict['gt_boxes']
        boxes[:, 2] = 0.0
        boxes[:, 5] = 1.0
        pc_range = self.point_cloud_range
        downsample = config.DOWNSAMPLE_RATIO
        resolution = config.LABEL_RESOLUTION
        x = np.arange(pc_range[0], pc_range[3], resolution, dtype=np.float32)
        y = np.arange(pc_range[1], pc_range[4], resolution, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        points = np.zeros((xx.size, 3), dtype=np.float32)
        points[:, 0] = xx
        points[:, 1] = yy

        point_indices = np.where(roiaware_pool3d_utils.points_in_boxes_cpu(points, boxes[:, :7]) > 0)
        box_idxs_of_pts = - np.ones(xx.size, dtype=np.int)
        box_idxs_of_pts[point_indices[1]] = point_indices[0]
        label_size = (int(data_dict['grid_maps_size'][0] / downsample),
                      int(data_dict['grid_maps_size'][1] / downsample))
        label = np.zeros(label_size, dtype=np.float32)
        x_inds = ((xx - pc_range[0]) / resolution).astype(np.int)
        y_inds = ((yy - pc_range[1]) / resolution).astype(np.int)
        mask = (box_idxs_of_pts >= 0)
        label[x_inds, y_inds] = mask.astype(np.int)

        reg = np.zeros(label_size + (6,), dtype=np.float32)
        reg[x_inds[mask], y_inds[mask], 0] = np.cos(boxes[box_idxs_of_pts[mask], 6])
        reg[x_inds[mask], y_inds[mask], 1] = np.sin(boxes[box_idxs_of_pts[mask], 6])
        reg[x_inds[mask], y_inds[mask], 2] = boxes[box_idxs_of_pts[mask], 0] - xx[mask]
        reg[x_inds[mask], y_inds[mask], 3] = boxes[box_idxs_of_pts[mask], 1] - yy[mask]
        reg[x_inds[mask], y_inds[mask], 4] = np.log(boxes[box_idxs_of_pts[mask], 4])
        reg[x_inds[mask], y_inds[mask], 5] = np.log(boxes[box_idxs_of_pts[mask], 5])

        data_dict['gt_pixel_cls'] = label
        data_dict['gt_pixel_reg'] = reg.transpose(2, 0, 1)

        return data_dict

    def get_occupancy_map(self, data_dict, config):
        z, y, x = np.split(data_dict['voxel_coords'], 3, axis=1)
        map = np.zeros((35, config.MAP_SIZE[0], config.MAP_SIZE[1]), dtype=np.float32)
        map[z, x, y] = 1.0
        return map

    def transform_points_to_voxels(self, data_dict=None, config=None, voxel_generator=None):
        if data_dict is None:
            try:
                from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            except:
                from spconv.utils import VoxelGenerator

            voxel_generator = VoxelGenerator(
                voxel_size=config.VOXEL_SIZE,
                point_cloud_range=self.point_cloud_range,
                max_num_points=config.MAX_POINTS_PER_VOXEL,
                max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode]
            )
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels, voxel_generator=voxel_generator)

        points = data_dict['points']
        voxel_output = voxel_generator.generate(points)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        # if not data_dict['use_lead_xyz']:
        #     voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict


class RangeProcessor():
    def __init__(self, processor_configs, point_cloud_range, training):
        self.cfg = processor_configs
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []
        self.occupancy_mask = None
        for cur_cfg in self.cfg:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        """
        Can be regarded as pass-through filter.
        """
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)
        mask = mask_points_by_range(data_dict['points'], self.point_cloud_range)
        data_dict['points'] = data_dict['points'][mask]
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def points_to_range_view(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.points_to_range_view, config=config)
        lidar_info = config.get('LIDAR_INFO', None)
        if lidar_info is None:
            fov_azi = [-38.4, 38.4]  # velodyne horizotal field of view, azimuth
            fov_ele = [-16.0, 4] # velodyne vertical field of view, elevation, official is [-24.9, 2.0]
            resolution_ele = 0.5
            resolution_azi = 0.3 # when rotation rate is 10Hz
        else:
            fov_azi = lidar_info['fov_azi']  # velodyne horizotal field of view, azimuth
            fov_ele = lidar_info['fov_ele'] # velodyne vertical field of view, elevation
            resolution_ele = lidar_info['resolution_ele']
            resolution_azi = lidar_info['resolution_azi'] # when rotation rate is 10Hz
        size = (int((fov_ele[1] - fov_ele[0]) / resolution_ele), int((fov_azi[1] - fov_azi[0]) / resolution_azi))
        range_map = np.zeros(size, dtype=np.float64)
        intensity_map = np.zeros(size, dtype=np.float64)
        rgb_map = np.zeros(size + (3,), dtype=np.float64)
        label = np.zeros(size, dtype=np.float64)
        points = data_dict['points']
        ranges = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
        azi = - np.arctan2(points[:, 1], points[:, 0]) / np.pi * 180
        ele = np.arctan2(points[:, 2], ranges) / np.pi * 180
        # select points in fov of range view
        selected = np.where((azi > fov_azi[0]) & (azi < fov_azi[1])
                            & (ele > fov_ele[0]) & (ele < fov_ele[1]))[0]
        # print(azi.min(), azi.max())
        # print(ele.min(), ele.max())
        points = points[selected]
        inds_x = np.clip((ele[selected] - fov_ele[0]) / resolution_ele, 0, size[0] - 1).astype(np.int)
        inds_y = np.clip((azi[selected] - fov_azi[0]) / resolution_azi, 0, size[1] - 1).astype(np.int)
        range_map[inds_x, inds_y] = ranges[selected]
        intensity_map[inds_x, inds_y] = points[:, 3]
        rgb_map[inds_x, inds_y] = points[:, 4:]
        range_view = np.concatenate(
                [range_map[..., None],
                 intensity_map[..., None],
                 rgb_map], axis=2)

        data_dict.update({
            'selected_points': points,
            'points_indices_in_range_map': np.stack([inds_x, inds_y], axis=1),
            'range_view_maps': range_view.transpose(2, 0, 1)
        })
        # get pointwise labels
        gt_boxes = data_dict['gt_boxes']
        points_mask = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, :3], gt_boxes[:, :7])
        classes, inds = np.where(points_mask)
        labels = np.zeros(points.shape[0])
        labels[inds] = gt_boxes[classes, -1]
        label[inds_x, inds_y] = labels
        data_dict['label_map'] = label[None, :, :]

        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
