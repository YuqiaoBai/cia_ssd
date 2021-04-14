import numpy as np
import os, sys
from tools.train import train
from tools.train_tf import train_tf, test_tf
from tools.test2 import test
from utils.train_utils import cfg_from_py
from copy import deepcopy


def experiment1(data_path, run_path, range, mode, cfgs):
    """Train seperated models for different ranges"""
    dcfg, mcfg, cfg = (cfg() for cfg in cfgs)

    dcfg.root = data_path
    dcfg.pc_range =  np.array([-range, -range, -3.0, range, range, 1.0]) # set z_max to 3.0 will dramatically increase num of voxels
    dcfg.cloud_name = mode
    cfg.PATHS['run'] = os.path.join(run_path, 'comap-{}-{:d}m'.format(mode, range))
    cfg.TRAIN['epoch'] = 0
    dcfg.update()
    cfgs_train = deepcopy((dcfg, mcfg, cfg)) # dict pop will make the cfg can only be used once
    cfgs_test = deepcopy((dcfg, mcfg, cfg))
    train(cfgs_train)
    test(cfgs_test)


def exp_train_test(data_path, run_path, cfgs):
    """Train and test all in one model"""
    dcfg, mcfg, cfg = (cfg() for cfg in cfgs)
    dcfg.root = data_path
    cfg.PATHS['run'] = os.path.join(run_path, 'sin_cos')
    cfgs_train = deepcopy((dcfg, mcfg, cfg))
    train(cfgs_train)
    cfg.TEST['com_range'] = 30
    _test(dcfg, mcfg, cfg)
    cfg.TEST['com_range'] = 40
    _test(dcfg, mcfg, cfg)
    cfg.TEST['com_range'] = 50
    _test(dcfg, mcfg, cfg)
    cfg.TEST['com_range'] = 60
    _test(dcfg, mcfg, cfg)


def exp_test(data_path, run_path, cfgs):
    """Test all in one model with different configurations"""
    dcfg, mcfg, cfg = (cfg() for cfg in cfgs)
    dcfg.root = data_path

    cfg.PATHS['run'] = os.path.join(run_path, 'box_encode_vec7_pos_dir')
    cfg.TEST['com_range'] = 30
    dcfg.node_selection_mode = 'kmeans_selection_30'
    _test(dcfg, mcfg, cfg)

    cfg.TEST['com_range'] = 40
    dcfg.node_selection_mode = 'kmeans_selection_40'
    _test(dcfg, mcfg, cfg)

    cfg.TEST['com_range'] = 50
    dcfg.node_selection_mode = 'kmeans_selection_50'
    _test(dcfg, mcfg, cfg)

    cfg.TEST['com_range'] = 60
    dcfg.node_selection_mode = 'kmeans_selection_60'
    _test(dcfg, mcfg, cfg)


def _test(dcfg, mcfg, cfg):
    for i in range(0, 5):
        cfg.TEST['n_coop'] = i
        cfgs_test = deepcopy((dcfg, mcfg, cfg))
        test(cfgs_test)


def experiment4(data_path, run_path, cfgs):
    """Test all in one model with different configurations"""
    dcfg, mcfg, cfg = (cfg() for cfg in cfgs)
    dcfg.root = data_path

    cfg.PATHS['run'] = os.path.join(run_path, 'comap-all-in-one-72m-com30m-tf2')
    cfgs_test = deepcopy((dcfg, mcfg, cfg))
    test_tf(cfgs_test)


def experiments(cfgs, data_path, run_path):
    # expetiment 1
    #experiment(data_path, run_path, 48, 'ego', cfgs)
    #experiment(data_path, run_path, 48, 'fused', cfgs)
    #experiment(data_path, run_path, 60, 'ego', cfgs)
    #experiment(data_path, run_path, 60, 'fused', cfgs)
    # experiment(data_path, run_path, 72, 'ego', cfgs)
    # experiment(data_path, run_path, 72, 'fused', cfgs)
    exp_train_test(data_path, run_path,  cfgs)


if __name__=="__main__":
    assert len(sys.argv)>2
    cfgs = cfg_from_py(sys.argv[2])
    if sys.argv[1]=='train':
        train((cfg() for cfg in cfgs))
    elif sys.argv[1]=='test':
        test((cfg() for cfg in cfgs))
    elif sys.argv[1]=='experiments':
        assert len(sys.argv)==5, 'data path and log path should be given for experiments'
        experiments(cfgs, sys.argv[3], sys.argv[4])
    else:
        raise ValueError('Argument #1 can only be \'train\', \'test\' or \'experiments\' \n'                         
                         'Argument #2 cfg file name\n'
                         'Argument #3 data path\n'
                         'Argument #4 log path')
        # experiments /media/hdd/ophelia/koko/data/synthdata3 /media/hdd/ophelia/koko/runs3
    # train cia_ssd_cofused.py