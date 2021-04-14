import numpy as np
from vlib.visulization import draw_points_boxes_bev_3d as visualization



def update(dcfg_obj):
    dcfg_obj.n_classes = len(dcfg_obj.classes)
    dcfg_obj.grid_size = np.round((dcfg_obj.pc_range[3:6] - dcfg_obj.pc_range[:3]) /
                         np.array(dcfg_obj.voxel_size)).astype(np.int64)
    dcfg_obj.feature_map_size = [1, *(dcfg_obj.grid_size / 8).astype(np.int64).tolist()[:-1]]
    dcfg_obj.TARGET_ASSIGNER['feature_map_size'] = dcfg_obj.feature_map_size



class Dataset(object):

    def __init__(self):
        super(Dataset, self).__init__()
        self.name = 'comap_tf'
        self.root = '/media/hdd/ophelia/koko/data/synthdata_20veh'
        self.pc_range = np.array([-70.4, -70.4, -3, 70.4, 70.4, 1])
        self.test_split = ['943', '1148', '753', '599', '53',
                      '905', '245', '421', '509']
        self.train_val_split = ['829', '965', '224', '685', '924', '334', '1175', '139',
                           '1070', '1050', '1162', '1260']
        self.train_split_ratio = 0.8
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

        self.add_gps_noise = True
        self.gps_noise_std = [1.0, 1.0, 0.0, 0.0, 0.0, 2.0] # [x, y, z, roll, pitch, yaw]

        self.process_fn = {
            'train': ['filter_ground_points', 'mask_points_in_range', 'points_to_voxel'],
            'test': ['filter_ground_points', 'mask_points_in_range', 'points_to_voxel']
        }


class Model:
    def __init__(self):
        super(Model, self).__init__()
        self.name = 'reginet'
        self.VFE = None
        self.SPCONV = {
            'num_out_features': 64
        }
        self. MAP2BEV = {
            'num_features': 128
        }
        self.LOSS = {
            'loss_reg': {'type': 'WeightedSmoothL1Loss',
                          'sigma': 1.0,
                          'code_weights': [1.0, 1.0, 1.0],
                          'codewise': True,
                          'loss_weight': 10.0},
        }


class Optimization:
    def __init__(self):
        self.TRAIN = {
        'project_name': None, #None, # 'cia_ssd',
        'visualization_func': visualization,
        'batch_size': 8,
        'lr': 0.00001,
        'weight_decay': 0.0001,
        'betas': [0.95, 0.999],
        'scheduler_step': 150,
        'scheduler_gamma': 0.5,
        'resume': False,
        'epoch': 0,
        'total_epochs': 60,
        'log_every': 20,
        'save_ckpt_every': 10
    }
        self.TEST = {
        'bev': False,
        'n_coop': 0,
        'score_threshold': 0.3,
        'cnt_threshold': 1.5,
        'nms_pre_max_size': 1000,
        'nms_post_max_size': 100,
        'nms_iou_threshold': 0.01,
        'ap_ious': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    }
        self.PATHS = {
        'run': '/media/hdd/ophelia/koko/experiments-output/reginet'
    }



