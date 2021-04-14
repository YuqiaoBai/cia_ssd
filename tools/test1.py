from tqdm import tqdm
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from vlib.image import draw_box_plt
from utils.train_utils import *


def test(cfgs):
    dcfg, mcfg, cfg = cfgs

    test_dataloader = build_dataloader(dcfg, cfg, train=False)
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    model = build_model(mcfg, cfg, dcfg).to(device)

    log_path = Path(cfg.PATHS['run'])
    # get metric
    from utils.eval import MetricAP
    metric = MetricAP(cfg.TEST, log_path, device='cuda')
    thrs = cfg.TEST['ap_ious']

    if metric.has_test_detections:
        aps = [metric.cal_ap_all_point(IoU_thr=thr)[0] for thr in thrs]
        with open(log_path / 'test_result_{}.txt'.format(cfg.TEST['score_threshold']), 'w') as fh:
            for thr,  ap in zip(thrs, aps):
                fh.writelines('mAP@{}: {:.2f}\n'.format(thr, ap * 100))
        return

    # load checkpoint
    model.load_state_dict(torch.load(str(log_path / 'epoch{:03}.pth'
                                         .format(cfg.TRAIN['total_epochs']))))
    # dir for save test images
    (log_path / 'test_{}'.format(cfg.TEST['score_threshold'])).mkdir(exist_ok=True)
    load_data_to_device = load_data_to_gpu if device.type == 'cuda' else load_data_as_tensor
    model.eval()

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
            pred_boxes = [pred_dict['box3d_lidar'] for pred_dict in predictions_dicts]
            scores = [pred_dict['scores'] for pred_dict in predictions_dicts]

            if i % 5 == 0:
                ax = plt.figure(figsize=(8, 8)).add_subplot(1, 1, 1)
                ax.set_aspect('equal', 'box')
                ax.set(xlim=(dcfg.pc_range[0], dcfg.pc_range[3]),
                       ylim=(dcfg.pc_range[1], dcfg.pc_range[4]))
                ax.plot(points[:, 0], points[:, 1], 'y.', markersize=0.3)
                ax = draw_box_plt(boxes, ax, color='green')
                ax = draw_box_plt(pred_boxes[0], ax, color='red')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.savefig(str(log_path / 'test_{}/{}.png'.format(cfg.TEST['score_threshold'],
                                                                   batch_data['frame'][0])))
                plt.close()

            metric.add_samples(batch_data['frame'], pred_boxes, batch_data['gt_boxes'], scores)

    metric.save_detections()
    aps = [metric.cal_ap_all_point(IoU_thr=thr)[0] for thr in thrs]
    with open(log_path / 'test_result_{}.txt'.format(cfg.TEST['score_threshold']), 'w') as fh:
        for thr, ap in zip(thrs, aps):
            fh.writelines('mAP@{}: {:.2f}\n'.format(thr, ap * 100))


def experiment(data_path, run_path, range, mode):
    dcfg.root = data_path
    dcfg.pc_range =  np.array([-range, -range, -3, range, range, 3])
    dcfg.cloud_name = mode
    cfg.PATHS['run'] = os.path.join(run_path, 'pcnet-{}-{:d}m'.format(mode, range))
    cfg.TRAIN['epoch'] = 0
    train()
    test()


def experiments(data_path, run_path):
    # expetiment 1
    experiment(data_path, run_path, 48, 'ego')
    experiment(data_path, run_path, 48, 'fused')
    # expetiment 2
    experiment(data_path, run_path, 60, 'ego')
    experiment(data_path, run_path, 60, 'fused')
    # expetiment 3
    experiment(data_path, run_path, 72, 'ego')
    experiment(data_path, run_path, 72, 'fused')


if __name__=="__main__":
    if sys.argv[1]=='train':
        train()
    elif sys.argv[1]=='test':
        test()
    elif sys.argv[1]=='experiments':
        assert len(sys.argv)==4, 'data path and log path should be given for experiments'
        experiments(sys.argv[2], sys.argv[3])
    else:
        raise ValueError('Argument #1 can only be \'train\', \'test\' or \'experiments\' \n'
                         'Argument #2 data path\n'
                         'Argument #3 log path')
        # experiments /media/hdd/ophelia/koko/data/synthdata3 /media/hdd/ophelia/koko/runs3


