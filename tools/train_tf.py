from utils.logger import LogMeter
from pathlib import Path
import wandb, shutil
from utils.train_utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


def train_tf(cfgs):
    dcfg, mcfg, cfg = cfgs
    cfg.TRAIN['pc_range'] = dcfg.pc_range
    # make checkpoints and log path
    log_path = Path(cfg.PATHS['run'])
    log_path.mkdir(exist_ok=True)
    # copy cfg file to log path
    cur_dir = os.path.abspath(__file__).rsplit('/', 2)[0]
    cfg_file = mcfg.name + '_' + dcfg.name + '.py'
    shutil.copy(os.path.join(cur_dir, 'cfg', cfg_file),
                os.path.join(cfg.PATHS['run'], cfg_file))

    train_dataloader = build_dataloader(dcfg, cfg)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    model = build_model(mcfg, cfg, dcfg).to(device)
    optimizer, lr_scheduler = build_optmizer(model, cfg)

    num_epochs = cfg.TRAIN['total_epochs']

    # get logger
    total_iterations = len(train_dataloader.dataset) // cfg.TRAIN['batch_size']
    logger = LogMeter(total_iterations, log_path, log_every=cfg.TRAIN['log_every'],
                      wandb_project=cfg.TRAIN['project_name'])

    # load checkpoint if resume training
    if cfg.TRAIN['resume']:
        if cfg.TRAIN['epoch']=='latest':
            model.load_state_dict(torch.load(str(log_path / 'latest.pth')))
        else:
            model.load_state_dict(torch.load(str(log_path / 'epoch{:03}.pth'.format(cfg.TRAIN['epoch']))))

    for epoch in range(cfg.TRAIN['epoch'], num_epochs):
        train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch + 1, cfg, device, logger=logger)

        lr_scheduler.step()

        # save checkpoint
        if (epoch + 1) % cfg.TRAIN['save_ckpt_every']==0:
            torch.save(model.state_dict(), str(log_path / 'epoch{:03}.pth'.format((epoch + 1))))
        else:
            torch.save(model.state_dict(), str(log_path / 'latest.pth'.format((epoch + 1))))
    print("Train finished.")
    wandb.finish()


def train_one_epoch(train_dataloader, model,optimizer, lr_scheduler, epoch, cfg, device, logger=None):
    load_data_to_device = load_data_to_gpu if device.type == 'cuda' else load_data_as_tensor
    iteration = 1
    len_data = len(train_dataloader)
    model.train()
    for batch_data in train_dataloader:
        load_data_to_device(batch_data)

        optimizer.zero_grad()
        # Forward pass
        batch_data = model(batch_data)
        loss_dict = model.loss(batch_data)
        loss = loss_dict['loss']
        # Getting gradients
        loss.backward()

        # clip gradients
        max_norm = 0.1
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # Updating parameters
        optimizer.step()

        iteration += 1

        # Log training
        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        if logger is not None:
            logger.log(epoch, iteration, lr_scheduler.get_last_lr()[0], **loss_dict)


def test_tf(cfgs):
    dcfg, mcfg, cfg = cfgs
    test_dataloader = build_dataloader(dcfg, cfg, train=False)
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    model = build_model(mcfg, cfg, dcfg).to(device)

    log_path = Path(cfg.PATHS['run'])
    test_out_path = log_path / 'test_result'
    test_out_path.mkdir(exist_ok=True)

    # load checkpoint
    model.load_state_dict(torch.load(str(log_path / 'epoch{:03}.pth'
                                         .format(cfg.TRAIN['total_epochs']))))
    # dir for save test images
    images_path = (test_out_path / 'images')
    images_path.mkdir(exist_ok=True)
    load_data_to_device = load_data_to_gpu if device.type == 'cuda' else load_data_as_tensor
    model.eval()

    i = 0
    with torch.no_grad():
        t = time.time()
        preds_list = []
        for batch_data in tqdm(test_dataloader):
            i += 1
            # if i > 50:
            #     break
            src_points = batch_data['src_points']
            src_points = src_points[src_points[:, 0]==0, 1:]
            src_points = src_points[(src_points!=0).all(axis=1)]
            tgt_points = batch_data['tgt_points']
            tgt_points = tgt_points[tgt_points[:, 0]==0, 1:]
            tgt_points = tgt_points[(tgt_points!=0).all(axis=1)]
            load_data_to_device(batch_data)

            # Forward pass
            batch_data = model(batch_data)
            preds, src_points_rot = model.post_processing(batch_data)
            src_points_rot = src_points_rot[0].cpu().numpy()
            preds_list.append(preds)

            plt.plot(tgt_points[:, 0], tgt_points[:, 1], 'g.', markersize=0.3)
            plt.plot(src_points[:, 0], src_points[:, 1], 'r.', markersize=0.1)
            plt.savefig(images_path / (batch_data['frame'][0] + '.png'))
            plt.close()
            plt.plot(tgt_points[:, 0], tgt_points[:, 1], 'g.', markersize=0.3)
            plt.plot(src_points_rot[:, 0], src_points_rot[:, 1], 'r.', markersize=0.1)
            plt.savefig(images_path / (batch_data['frame'][0] + '_rot.png'))
            plt.close()
        ave_run_time = (time.time() - t) / len(test_dataloader) / 8
        print('average rum time: ', ave_run_time)






