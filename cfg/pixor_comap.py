import numpy as np
from vlib.visulization import draw_points_boxes_bev as visualization



def update(dcfg_obj):
    dcfg_obj.n_classes = len(dcfg_obj.classes)
    dcfg_obj.grid_size = np.round((dcfg_obj.pc_range[3:6] - dcfg_obj.pc_range[:3]) /
                         np.array(dcfg_obj.voxel_size)).astype(np.int64)
    dcfg_obj.feature_map_size = [1, *(dcfg_obj.grid_size / 8).astype(np.int64).tolist()[:-1]]
    dcfg_obj.TARGET_ASSIGNER['feature_map_size'] = dcfg_obj.feature_map_size


class Dataset(object):
    LABEL_COLORS = {
        'Unlabeled': (0, 0, 0),  # 0 Unlabeled
        'Buildings': (70, 70, 70),  # 1 Buildings
        'Fences': (100, 40, 40),  # 2 Fences
        'Other': (55, 90, 80),  # 3 Other
        'Pedestrians': (220, 20, 60),  # 4 Pedestrians
        'Poles': (153, 153, 153),  # 5 Poles
        'RoadLines': (157, 234, 50),  # 6 RoadLines
        'Roads': (128, 64, 128),  # 7 Roads
        'Sidewalks': (244, 35, 232),  # 8 Sidewalks
        'Vegetation': (107, 142, 35),  # 9 Vegetation
        'Vehicles': (0, 0, 142),  # 10 Vehicles
        'Walls': (102, 102, 156),  # 11 Walls
        'TrafficSign': (220, 220, 0),  # 12 TrafficSign
        'Sky': (70, 130, 180),  # 13 Sky
        'Ground': (81, 0, 81),  # 14 Ground
        'Bridge': (150, 100, 100),  # 15 Bridge
        'Railtrack': (230, 150, 140),  # 16 Railtrack
        'GuardRail': (180, 165, 180),  # 17 GuardRail
        'TrafficLight': (250, 170, 30),  # 18 TrafficLight
        'Static': (110, 190, 160),  # 19 Static
        'Dynamic': (170, 120, 50),  # 20 Dynamic
        'Water': (45, 60, 150),  # 21 Water
        'Terrain': (145, 170, 100)  # 22 Terrain
    }
    AUGMENTOR = {
        'random_world_flip': ['y'],
        'random_world_rotation': [-10, 10],  # rotation range in degree
        'random_world_scaling': [0.95, 1.05]  # scale range
    }

    BOX_CODER = {
        'type': 'GroundBoxBevGridCoder',
        'linear_dim': False,
        'n_dim': 5,
        'encode_angle_vector': True,
        'box_means': [-0.00161, 0.00198, 1.51452, 0.70422, -0.01229, 0.01600],
        'box_stds': [1.12314, 0.96673, 0.18102, 0.10640, 0.62434, 0.78089]
    }

    def __init__(self):
        super(Dataset, self).__init__()
        self.name = 'comap'
        self.root = '/media/hdd/ophelia/koko/data/synthdata_20veh_30m'
        self.pc_range = np.array([-40, -40, -3, 70.4, 40, 1])
        self.test_split = ['943', '1148', '753', '599', '53',
                      '905', '245', '421', '509']
        self.train_val_split = ['829', '965', '224', '685', '924', '334', '1175', '139',
                           '1070', '1050', '1162', '1260']
        self.train_split_ratio = 0.8
        self.classes = {1: ['Vehicles']}  # 0 is reserved for not-defined class
        self.voxel_size = [0.1, 0.1, 0.1]
        self.n_point_features = 3  # x,y,z
        self.label_downsample = 4
        self.n_coop = 'random'

        self.add_gps_noise = False
        self.gps_noise_std = [0.5, 0.5, 0.0, 0.0, 0.0, 2.0] # [x, y, z, roll, pitch, yaw]

        # This part induct info from the info provided above
        self.n_classes = len(self.classes)
        self.grid_size = np.round((self.pc_range[3:6] - self.pc_range[:3]) /
                             np.array(self.voxel_size)).astype(np.int64)
        self.pixel_label_size = (self.grid_size / self.label_downsample).astype(np.int64).tolist()[:-1]

        self.process_fn = {
            'train': ['mask_points_in_range', 'rm_empty_gt_boxes', 'get_bev_input', 'get_pixel_labels'],
            'test': ['mask_points_in_range', 'get_bev_input']
        }


    def update(self):
        self.n_classes = len(self.classes)
        self.grid_size = np.round((self.pc_range[3:6] - self.pc_range[:3]) /
                             np.array(self.voxel_size)).astype(np.int64)
        self.pixel_label_size = (self.grid_size / self.label_downsample).astype(np.int64).tolist()[:-1]


class Model:
    def __init__(self):
        super(Model, self).__init__()
        self.name = 'pixor'
        self.expansion = 4
        self.c_blocks = [24, 48, 64, 96]
        self.n_blocks = [3, 6, 6, 3]
        self.up_channels = [196, 128, 96]
        self.header_channels = 96
        self.head_layers = 4

        self.LOSS = {
            'loss_norm': {'type': 'NormByNumExamples',
                          'pos_cls_weight': 20.0,
                          'neg_cls_weight': 1.0},
            'loss_cls': {'type': 'WeightedSigmoidClassificationLoss',
                         'loss_weight': 1.0},
            'use_sigmoid_score': True,
            'loss_bbox': {'type': 'WeightedSmoothL1Loss',
                          'sigma': 1.0,
                          'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                          'codewise': True,
                          'loss_weight': 1.0},
        }

        self.NMS = {'name': 'normal', # 'iou_weighted',
                    'score_threshold': 0.8,
                    'nms_pre_max_size': 1000,
                    'nms_post_max_size': 100,
                    'nms_iou_threshold': 0.01,
                    }


class Optimization:
    def __init__(self):
        self.TRAIN = {
        'project_name': None, #None, # 'cia_ssd',
        'visualization_func': visualization,
        'batch_size': 1,
        'lr': 0.0001,
        'weight_decay': 0.0001,
        'betas': [0.95, 0.999],
        'scheduler_step': 150,
        'scheduler_gamma': 0.5,
        'resume': False,
        'epoch': 0,
        'total_epochs': 50,
        'log_every': 20,
        'save_ckpt_every': 5
}

        self.TEST = dict(
            bev=True, n_coop=0,
            score_threshold=0.8,
            cnt_threshold=1.5,
            nms_pre_max_size=1000,
            nms_post_max_size=100,
            nms_iou_threshold=0.01,
            ap_ious=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

        self.PATHS = dict(run='/media/hdd/ophelia/koko/experiments-output/pixor')



