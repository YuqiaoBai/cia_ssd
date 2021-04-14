from utils.logger import LogMeter
from pathlib import Path
import wandb, shutil
from utils.train_utils import *
from tqdm import tqdm
from utils.batch_statistics import StatsRecorder



def train(cfgs):
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

    # get logger
    total_iterations = len(train_dataloader.dataset) // cfg.TRAIN['batch_size']
    logger = LogMeter(total_iterations, log_path, log_every=cfg.TRAIN['log_every'],
                      wandb_project=cfg.TRAIN['project_name'])

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    model = build_model(mcfg, cfg, dcfg).to(device)
    optimizer, lr_scheduler = build_optmizer(model, cfg)

    num_epochs = cfg.TRAIN['total_epochs']

    # load checkpoint if resume training
    if cfg.TRAIN['resume']:
        if cfg.TRAIN['epoch']==0:
            model.load_state_dict(torch.load(str(log_path / 'latest.pth')))
        else:
            model.load_state_dict(torch.load(str(log_path / 'epoch{:03}.pth'.format(cfg.TRAIN['epoch']))))

    for epoch in tqdm(range(cfg.TRAIN['epoch'], num_epochs)):
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
        # if iteration==120:
        #     print("debug")
        points = batch_data.pop('points')
        points = points[points[: ,0]==0, 1:4]
        # bev = batch_data['bev_input'][0]
        gt_boxes = batch_data['gt_boxes'][0]
        load_data_to_device(batch_data)

        optimizer.zero_grad()
        # Forward pass
        batch_data = model(batch_data)
        if (len_data * (epoch-1) + iteration) % 40== 0 and cfg.TRAIN['project_name'] is not None:
            with torch.no_grad():
                # regs_pred = batch_data['reg_logits'][0].permute(1, 2, 0)
                # regs_tgt = batch_data['regression_map'][0].permute(1, 2, 0)
                # xs, ys = torch.where(regs_tgt.sum(dim=-1))
                # tmp = regs_tgt[xs, ys, -2:]
                # wandb.log({'reg/cos': wandb.Histogram(regs_pred[xs, ys, -1].detach().cpu().numpy()),
                #            'reg/diff_cos': torch.abs(regs_pred[xs, ys, -1] - regs_tgt[xs, ys, -1]).mean().item(),
                #            'reg/diff_cos_histo': wandb.Histogram(torch.abs(regs_pred[xs, ys, -1] -
                #                                            regs_tgt[xs, ys, -1]).detach().cpu().numpy()),
                #            })
                # wandb.log({'cls/input': wandb.Image((bev.sum(axis=0) > 0).transpose()[::-1, :].astype(np.float))})
                # scores = torch.sigmoid(batch_data['cls_logits'][0]).squeeze()
                # target = batch_data['label_map'][0].squeeze()
                # reg_tgt = batch_data['regression_map'][0]
                # wandb.log({'cls/predictions': wandb.Image(scores.detach().cpu().numpy().transpose()[::-1,:]),
                #            'cls/targets': wandb.Image(target.detach().cpu().numpy().transpose()[::-1,:]),
                #            'reg/targets': wandb.Image((reg_tgt.detach()[0]!=0).int().cpu().numpy().transpose()[::-1,:]),})
                if epoch>2 and iteration % 40 == 0:
                    wandb.log({'test': wandb.Image(np.zeros([100, 100]))})
                    predictions_dicts = model.post_processing(batch_data, cfg.TEST)
                    n_boxes = len(predictions_dicts[0]['box_lidar'])
                    pred_boxes = predictions_dicts[0]['box_lidar'] if n_boxes>0 else None
                    cfg.TRAIN['visualization_func'](points, pred_boxes, gt_boxes, cfg.TRAIN['pc_range'])
                    # cfg.TRAIN['visualization_func'](points, pred_boxes=None, gt_boxes=None,
                    #                                 pc_range=cfg.TRAIN['pc_range'])
                    # pass

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
        loss_dict = {k: v.item() for k, v in loss_dict.items() if v}
        if logger is not None:
            logger.log(epoch, iteration, lr_scheduler.get_last_lr()[0], **loss_dict)




