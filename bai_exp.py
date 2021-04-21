from utils.train_utils import *
from pathlib import Path
import torch
from CNN.model import ConvNet
from CNN.dataloader import coopDataset


def data_masking(data_clouds, mask):
    inds = torch.nonzero(mask)
    a = torch.zeros(data_clouds.size)
    a[inds]=data_clouds[inds]
    return a


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


def bai_experiment(cfgs):
    # configure
    dcfg, mcfg, cfg = cfgs

    # data loader convnet
    # Load data
    train_data = coopDataset()
    data_loader_convnet = DataLoader(train_data)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # construct model
    cnn_model = ConvNet().to(device)
    od_model = build_model(mcfg, cfg, dcfg).to(device)

    # resume/ reload parems
    # load checkpoint
    log_path = Path(cfg.PATHS['run'])
    od_model.load_state_dict(torch.load(str(log_path / 'epoch{:03}.pth'.format(cfg.TRAIN['total_epochs']))))
    # optimizer
    optimizer, lr_scheduler = build_optmizer(od_model, cfg)

    # forward cnn
    for data in data_loader_convnet:
        # data_clouds = data["clouds"] # 256x256x21
        data = torch.squeeze(data, dim=0).cuda()
        out = cnn_model(data) # 256x256x20 as batch
        input_od = data_masking(data, out)
        pred = od_model(input_od)
        print(pred)


