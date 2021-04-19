import numpy as np
from vlib.visulization import draw_points_boxes_bev_3d as visualization



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
        'random_world_flip': ['x'],
        'random_world_rotation': [-45, 45],  # rotation range in degree
        'random_world_scaling': [0.95, 1.05]  # scale range
    }

    BOX_CODER = {
        'type': 'GroundBox3dCoderTorch',
        'linear_dim': False,
        'n_dim': 7,
        'angle_vec_encode': True, # encode_angle_vector
        'angle_residual_encode': False # encode_angle_with_residual
    }

    def __init__(self):
        super(Dataset, self).__init__()
        self.name = 'cofused'
        self.root = '/media/ExtHDD01/mastudent/BAI/HybridV50CAV20'
        self.pc_range = np.array([-40, -40, -3, 40, 40, 1])
        self.test_split = ['965', '224', '685', '924', '334', '1175', '139',
                           '1070', '1050', '1162', '1260']
        self.train_val_split = ['829', '943', '1148', '753', '599', '53', '905', '245', '421', '509']
        self.train_split_ratio = 0.8
        self.ego_cloud_name = 'cloud_ego' # 'noisy_cloud_ego'
        self.coop_cloud_name = 'cloud_coop' # 'noisy_cloud_coop'
        self.node_selection_mode = None # 'kmeans_selection_40'
        self.selected_points = None # 'path to selected points npy file'
        self.classes = {1: ['Vehicles'], 2: ['Roads', 'RoadLines']}  # 0 is reserved for not-defined class

        self.voxel_size = [0.1, 0.1, 0.1]

        self.max_points_per_voxel = 20
        self.max_num_voxels = 100000
        self.cal_voxel_mean_std = False
        self.n_point_features = 3  # x,y,z
        self.label_downsample = 4

        # This part induct info from the info provided above
        self.n_classes = len(self.classes)
        self.grid_size = np.round((self.pc_range[3:6] - self.pc_range[:3]) /
                             np.array(self.voxel_size)).astype(np.int64)
        self.feature_map_size = [1, *(self.grid_size / 8).astype(np.int64).tolist()[:-1]]

        self.add_gps_noise = False
        self.gps_noise_std = [0.5, 0.5, 0.0, 0.0, 0.0, 2.0] # [x, y, z, roll, pitch, yaw]

        self.TARGET_ASSIGNER ={
            'anchor_generator': {
                'type': 'AnchorGeneratorRange',
                 'sizes': [ 4.41, 1.98, 1.64],
                 'rotations': [0, 1.57],
                 'match_threshold': 0.6,
                 'unmatch_threshold': 0.45,
                 'class_name': 'Car'
            },
            'sample_positive_fraction': None,
            'sample_size': 512,
            'pos_area_threshold': -1,
            'box_coder': self.BOX_CODER,
            'out_size_factor': 8,
            'enable_similar_type': True,
            'feature_map_size': self.feature_map_size
        }

        self.process_fn = {
            'train': ['mask_points_in_range', 'rm_empty_gt_boxes', 'shuffle_points',
                      'points_to_voxel', 'assign_target'],
            'test': ['mask_points_in_range', 'points_to_voxel', 'assign_target']
        }


    def update(self):
        self.n_classes = len(self.classes)
        self.grid_size = np.round((self.pc_range[3:6] - self.pc_range[:3]) /
                                      np.array(self.voxel_size)).astype(np.int64)
        self.feature_map_size = [1, *(self.grid_size / 8).astype(np.int64).tolist()[:-1]]
        self.TARGET_ASSIGNER['feature_map_size'] = self.feature_map_size



class Model:
    def __init__(self):
        super(Model, self).__init__()
        self.name = 'cia_ssd'
        self.VFE = None
        self.SPCONV = {
            'num_out_features': 64
        }
        self. MAP2BEV = {
            'num_features': 128
        }
        self.SSFA = {
            'layer_nums': [5],
            'ds_layer_strides': [1],
            'ds_num_filters': [128],
            'us_layer_strides': [1],
            'us_num_filters': [128],
            'num_input_features': 128,
            'norm_cfg': None
        }

        self.HEAD = {
                'type': 'MultiGroupHead',
                'mode': '3d',
                'in_channels': sum([128,]),
                'norm_cfg': None,
                'num_class': 1,
                'class_names': ['Car'],
                'weights': [1, ],
                'with_cls' : True,
                'with_reg' : True,
                'reg_class_agnostic' : False,
                'pred_var': False,
                'box_coder':  Dataset.BOX_CODER,
                'encode_background_as_zeros': True,
                'loss_norm': {'type': 'NormByNumPositives',
                              'pos_cls_weight': 50.0,
                              'neg_cls_weight': 1.0},
                'loss_cls': {'type': 'SigmoidFocalLoss',
                             'alpha': 0.25,
                             'gamma': 2.0,
                             'loss_weight': 1.0},
                'use_sigmoid_score': True,
                'loss_bbox': {'type': 'WeightedSmoothL1Loss',
                              'sigma': 3.0,
                              'code_weights': None,
                              'codewise': True,
                              'loss_weight': 2.0},
                'loss_iou': {'type': 'WeightedSmoothL1Loss',
                              'sigma': 3.0,
                              'code_weights': None,
                              'codewise': True,
                              'loss_weight': 1.0},
                'encode_rad_error_by_sin': False,
                'use_dir_classifier': False,
                'loss_aux': {'type': 'WeightedSoftmaxClassificationLoss',
                             'name': 'direction_classifier',
                             'loss_weight': 0.2},
                'direction_offset': 0.0,
                'nms': {
                    'name': 'normal', # 'iou_weighted',
                    'score_threshold': 0.3,
                    'cnt_threshold': 1,
                    'nms_pre_max_size': 1000,
                    'nms_post_max_size': 100,
                    'nms_iou_threshold': 0.01,
                },
                'logger': None
        }


class Optimization:
    def __init__(self):
        self.TRAIN = {
        'project_name': None, #None, # 'cia-ssd',
        'visualization_func': visualization,
        'batch_size': 8,
        'lr': 0.0001,
        'weight_decay': 0.0001,
        'betas': [0.95, 0.999],
        'scheduler_step': 150,
        'scheduler_gamma': 0.5,
        'resume': False,
        'epoch': 0,
        'total_epochs': 50,
        'log_every': 20,
        'save_ckpt_every': 10
    }
        self.TEST = {
        'bev': False,
        'n_coop': 0,
        'com_range': 40,
        'score_threshold': 0.3,
        'cnt_threshold': 2,
        'nms_pre_max_size': 1000,
        'nms_post_max_size': 100,
        'nms_iou_threshold': 0.01,
        'ap_ious': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    }
        self.PATHS = {
        'run': '/media/ExtHDD01/mastudent/BAI/HybridV50CAV20'
    }



