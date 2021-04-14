import torch
from collections import namedtuple
import numpy as np
import os
from torch.utils.data import DataLoader

from datasets.kitti import kitti_3d
from datasets.comap import comap, comap_tf
from datasets.cofused import cofused
from models.cia_ssd import CIASSD
from models.pixor import PIXOR
from models.reginet import RegiNet


def cfg_from_py(filename):
    '''
    Input:
        filename: a python file containing config info, this file should be in dir root/cfg
    Return:
        dcfg: data config
        mcfg: model config
        cfg: optimization and others
    '''
    root_dir = os.path.abspath(__file__).rsplit('/', 2)[0]
    if os.path.exists(os.path.join(root_dir, 'cfg', filename + '.py')):
        cfg_file_hd = getattr(__import__("cfg.%s" % filename), filename)
        dcfg = cfg_file_hd.Dataset
        mcfg = cfg_file_hd.Model
        cfg = cfg_file_hd.Optimization
    else:
        raise FileNotFoundError('\"{}.py\" not found. Make sure config file is in \"rootdir/cfg\"'.format(filename))
    return dcfg, mcfg, cfg


def build_dataloader(dcfg, cfg, train=True):
    datasets_dict = {
        'comap': comap.CoMapDataset,
        'comap_tf': comap_tf.CoMapTFDataset,
        'kitti_3d': kitti_3d.Kitti3DDataset,
        'cofused': cofused.CoFusedDataset,
    }
    assert dcfg.name in datasets_dict
    # Dataset
    if not train and 'n_coop' in list(cfg.TEST.keys()) and 'com_range' in list(cfg.TEST.keys()):
        dataset = datasets_dict[dcfg.name](dcfg, training=train,
                                           n_coop=cfg.TEST['n_coop'],
                                           com_range=cfg.TEST['com_range'])
    else:
        dataset = datasets_dict[dcfg.name](dcfg, training=train)

    sampler = None
    dataloader = DataLoader(
        dataset, batch_size=cfg.TRAIN['batch_size'], pin_memory=True, num_workers=1,
        shuffle=train,  collate_fn=dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0
    )

    return dataloader


def build_model(mcfg, cfg, dcfg):
    model_dict = {
        'cia_ssd': CIASSD,
        'pixor': PIXOR,
        'reginet': RegiNet,
    }
    assert mcfg.name in model_dict
    return model_dict[mcfg.name](mcfg, cfg, dcfg)


def build_optmizer(model, cfg):
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params,  lr=cfg.TRAIN['lr'],
                                 weight_decay=cfg.TRAIN['weight_decay'],
                                 betas=tuple(cfg.TRAIN['betas']))

    # construct a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=cfg.TRAIN['scheduler_step'],
                                                   gamma=cfg.TRAIN['scheduler_gamma'])

    return optimizer, lr_scheduler


def _load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if key in ['frame_id', 'metadata', 'calib', 'image_shape',
                   'grid_maps_size', 'num_boxes_per_sample', 'frame', 'batch_size']:
            continue
        if key in ['gt_pixel_label']:
            if isinstance(batch_dict[key], list):
                for i in range(len(val)):
                    batch_dict[key][i] = torch.from_numpy(val[i]).long().cuda()
            else:
                batch_dict[key] = torch.from_numpy(val).long().cuda()
        else:
            if isinstance(batch_dict[key], list):
                for i in range(len(val)):
                    batch_dict[key][i] = torch.from_numpy(val[i]).float().cuda()
            else:
                batch_dict[key] = torch.from_numpy(val).float().cuda()


def load_data_to_gpu(batch_dict):
    batch_types = batch_dict.pop('batch_types')
    for key, val in batch_dict.items():
        if 'none' in batch_types[key]:
            continue
        if 'gpu' in batch_types[key] and 'int' in batch_types[key]:
            if isinstance(batch_dict[key], list):
                for i in range(len(val)):
                    batch_dict[key][i] = torch.from_numpy(val[i]).long().cuda()
            else:
                batch_dict[key] = torch.from_numpy(val).long().cuda()
        elif 'gpu' in batch_types[key] and 'float' in batch_types[key]:
            if isinstance(batch_dict[key], list):
                for i in range(len(val)):
                    batch_dict[key][i] = torch.from_numpy(val[i]).float().cuda()
            else:
                batch_dict[key] = torch.from_numpy(val).float().cuda()


def load_data_as_tensor(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape', 'grid_maps_size', 'num_boxes_per_sample']:
            continue
        if key=='label_map':
            batch_dict[key] = torch.from_numpy(val).long()
        else:
            batch_dict[key] = torch.from_numpy(val).float()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)