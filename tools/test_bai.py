from tqdm import tqdm
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from vlib.image import draw_box_plt
from utils.train_utils import *


def test(cfgs):
    dcfg, mcfg, cfg = cfgs
    n_coop = cfg.TEST['n_coop'] if 'n_coop' in list(cfg.TEST.keys()) else 0
    com_range = cfg.TEST['com_range'] if 'com_range' in list(cfg.TEST.keys()) else 0

    test_dataloader = build_dataloader(dcfg, cfg, train=False)
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    model = build_model(mcfg, cfg, dcfg).to(device)

    log_path = Path(cfg.PATHS['run'])
    test_out_path = log_path / 'test_result_{:2d}m'.format(com_range)
    test_out_path.mkdir(exist_ok=True)
    # get metric
    from utils.eval import MetricAP
    metric = MetricAP(cfg.TEST, test_out_path, device='cuda', bev=cfg.TEST['bev'])
    thrs = cfg.TEST['ap_ious']

    if metric.has_test_detections:
        aps = [metric.cal_ap_all_point(IoU_thr=thr)[0] for thr in thrs]
        with open(test_out_path / 'thr{}_ncoop{}.txt'.format(cfg.TEST['score_threshold'], n_coop), 'w') as fh:
            for thr,  ap in zip(thrs, aps):
                fh.writelines('mAP@{}: {:.2f}\n'.format(thr, ap * 100))

    # load checkpoint
    model.load_state_dict(torch.load(str(log_path / 'epoch{:03}.pth'
                                         .format(cfg.TRAIN['total_epochs']))))
    # dir for save test images
    images_path = (test_out_path / 'images_{}_{}'.format(cfg.TEST['score_threshold'], n_coop))
    images_path.mkdir(exist_ok=True)
    load_data_to_device = load_data_to_gpu if device.type == 'cuda' else load_data_as_tensor
    model.eval()
    direcs = []
    i = 0
    with torch.no_grad():
        for batch_data in tqdm(test_dataloader):
            i += 1
            points = batch_data['points']
            points = points[points[:, 0] == 0, 1:3]
            boxes = batch_data['gt_boxes'][0]
            load_data_to_device(batch_data)

            # Forward pass
            batch_data = model(batch_data)
            predictions_dicts = model.post_processing(batch_data, cfg.TEST)
            pred_boxes = [pred_dict['box_lidar'] for pred_dict in predictions_dicts]
            scores = [pred_dict['scores'] for pred_dict in predictions_dicts]
            direcs.extend([boxes[:, -1] for boxes in pred_boxes])
            if i % 1 == 0:
                ax = plt.figure(figsize=(8, 8)).add_subplot(1, 1, 1)
                ax.set_aspect('equal', 'box')
                ax.set(xlim=(dcfg.pc_range[0], dcfg.pc_range[3]),
                       ylim=(dcfg.pc_range[1], dcfg.pc_range[4]))
                ax.plot(points[:, 0], points[:, 1], 'y.', markersize=0.3)
                ax = draw_box_plt(boxes, ax, color='green', linewidth_scale=2)
                ax = draw_box_plt(pred_boxes[0], ax, color='red')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.savefig(str(images_path / '{}.png'.format(batch_data['frame'][0])))
                plt.close()

            metric.add_samples(batch_data['frame'], pred_boxes, batch_data['gt_boxes'], scores)

    metric.save_detections()
    aps = [metric.cal_ap_all_point(IoU_thr=thr)[0] for thr in thrs]
    with open(test_out_path / 'thr{}_ncoop{}.txt'.format(cfg.TEST['score_threshold'], n_coop), 'w') as fh:
        for thr, ap in zip(thrs, aps):
            fh.writelines('mAP@{}: {:.2f}\n'.format(thr, ap * 100))
    return predictions_dicts

