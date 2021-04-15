import numpy as np


class Dataset:
    name = 'kitti_3d'
    data_path = '/media/hdd/ophelia/data/kitti'
    pc_range = np.array([0, -40.0, -3.0, 70.4, 40.0, 1.0])
    fov_points_only = True
    voxel_size = [0.05, 0.05, 0.1]
    max_points_per_voxel = 65
    max_num_voxels = 60000
    cal_voxel_mean_std = False
    n_point_features = 3  # x,y,z
    label_downsample = 4

    # This part induct info from the info provided above
    class_dict = {'Car': ['Car', 'Van']}
    n_classes = len(class_dict)
    grid_size = np.round((pc_range[3:6] - pc_range[:3]) /
                         np.array(voxel_size)).astype(np.int64)
    feature_map_size = [1, *(grid_size / 8).astype(np.int64).tolist()[:-1]]

    AUGMENTOR = {
        'gt_sampling': {
            'use_road_plane': True,
            'db_info_path': ['kitti_dbinfos_train.pkl'],
            'prepare': {'filter_by_min_points': {'Car': 5, 'Van': 5},
                        'filter_by_difficulty': [-1]},
            'sample_groups': {'Car': 25, 'Van': 15},
            'num_point_features': 3,
            'database_with_fakelidar': False,
            'remove_extra_width': [0.0, 0.0, 0.0],
            'limit_whole_scene': True
        },
        'gt_transformation': {
            'random_shift': [1.0, 1.0, 0.5],
            'random_rotation': [-0.785, 0.785]
        },
        'random_world_flip': ['x'],
        'random_world_rotation': [-0.78539816, 0.78539816],  # rotation range in degree
        'random_world_scaling': [0.95, 1.05]  # scale range
    }

    BOX_CODER = {
        'type': 'GroundBox3dCoderTorch',
        'linear_dim': False,
        'n_dim': 7,
        'encode_angle_vector': False
    }

    TARGET_ASSIGNER ={
        'anchor_generator': {
            'type': 'AnchorGeneratorRange',
             'sizes': [1.6, 3.9, 1.56],
             'anchor_ranges': [0, -40.0, -1.0, 70.4, 40.0, -1.0],
             'rotations': [0, 1.57],
             'match_threshold': 0.6,
             'unmatch_threshold': 0.45,
             'class_name': 'Car'
        },
        'sample_positive_fraction': None,
        'sample_size': 512,
        'pos_area_threshold': -1,
        'box_coder': BOX_CODER,
        'out_size_factor': 8,
        'enable_similar_type': True,
        'feature_map_size': feature_map_size
    }

    DATA_PROCESSOR = {
        'mask_points_and_boxes_outside_range': {'remove_outside_boxes': True},
        'shuffle_points': {
            'shuffle_enabled': {'train': True, 'test': False}
        },
        'transform_points_to_voxels': {
            'voxel_size': [0.05, 0.05, 0.1],
            'max_points_per_voxel': 5,
            'max_number_of_voxels': {
                'train': 16000,
                'test': 40000
            }
        }
    }


class Model:
    name = 'cia-ssd'
    VFE = None
    SPCONV = {
        'num_out_features': 64
    }
    MAP2BEV = {
        'num_features': 128
    }
    SSFA = {
        'layer_nums': [5],
        'ds_layer_strides': [1],
        'ds_num_filters': [128],
        'us_layer_strides': [1],
        'us_num_filters': [128],
        'num_input_features': 128,
        'norm_cfg': None
    }

    HEAD = {
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
            'box_coder':  Dataset.BOX_CODER,
            'encode_background_as_zeros': True,
            'loss_norm': {'type': 'NormByNumPositives',
                          'pos_cls_weight': 1.0,
                          'neg_cls_weight': 1.0},
            'loss_cls': {'type': 'SigmoidFocalLoss',
                         'alpha': 0.25,
                         'gamma': 2.0,
                         'loss_weight': 1.0},
            'use_sigmoid_score': True,
            'loss_bbox': {'type': 'WeightedSmoothL1Loss',
                          'sigma': 3.0,
                          'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                          'codewise': True,
                          'loss_weight': 2.0},
            'loss_iou': {'type': 'WeightedSmoothL1Loss',
                          'sigma': 3.0,
                          'code_weights': None,
                          'codewise': True,
                          'loss_weight': 1.0},
            'encode_rad_error_by_sin': True,
            'loss_aux': {'type': 'WeightedSoftmaxClassificationLoss',
                         'name': 'direction_classifier',
                         'loss_weight': 0.2},
            'direction_offset': 0.0,
            'logger': None
    }


class Optimization:
    TRAIN = {
        'batch_size': 1,
        'lr': 0.0001,
        'weight_decay': 0.0001,
        'betas': [0.95, 0.999],
        'scheduler_step': 150,
        'scheduler_gamma': 0.5,
        'resume': False,
        'epoch': 0,
        'total_epochs': 30,
        'log_every': 20,
        'save_ckpt_every': 10
    }
    TEST = {
        'thr_conf': 0.4,
        'thr_nms_iou': 0.05,
        'ap_ious': [0.3, 0.5, 0.7, 0.9],
        'use_multi_class_nms': False,
        'nms_pre_max_size': 1000,
        'nms_post_max_size': 100,
        'nms_iou_threshold': 0.01
    }
    PATHS = {
        'run': '/media/hdd/ophelia/koko/runs3/debug'
    }


dcfg = Dataset
cfg = Optimization
mcfg = Model