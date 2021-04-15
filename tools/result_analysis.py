import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
import json, tqdm
from ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu, boxes_iou_bev
from utils.batch_statistics import StatsRecorder
from utils.common_utils import limit_period



def read_aps(path):
    com_ranges = [30, 40, 50, 60]
    result = dict()
    for r in com_ranges:
        files = glob(os.path.join(path, 'test_result_{:d}m'.format(r), '*.txt'))
        files.sort()
        result[r] = dict()
        for f in files:
            with open(f, 'r') as fh:
                aps = [float(line.split(':')[-1]) for line in fh.readlines()]
                result[r][int(f[-5])] = aps

    return result


def cal_avg_iou(path, com, n_veh, IoU=0.3, thr=0.3, bev=False):
    file = os.path.join(path, 'test_result_{:d}m'.format(com), 'thr{:.1f}_ncoop{:d}.pth'.format(thr, n_veh))
    data = torch.load(file)
    samples = data['samples']
    pred_boxes = data['pred_boxes']
    gt_boxes = data['gt_boxes']
    confidences = data['confidences']
    pos_ious = []
    n_gt = 0
    iou_fn = boxes_iou_bev if bev else boxes_iou3d_gpu
    for sample in samples:
        if len(pred_boxes[sample]) > 0 and len(gt_boxes[sample]) > 0:
            ious = iou_fn(pred_boxes[sample], gt_boxes[sample])
            n, m = ious.shape
            n_gt = n_gt + m - 1
            max_iou_pred_to_gts = ious.max(dim=1)
            max_iou_gt_to_preds = ious.max(dim=0)
            tp = max_iou_pred_to_gts[0] > IoU
            is_best_match = max_iou_gt_to_preds[1][max_iou_pred_to_gts[1]] \
                            == torch.tensor([i for i in range(len(tp))], device=tp.device)
            tp[torch.logical_not(is_best_match)] = False
            tp[max_iou_pred_to_gts[0]>0.9999] = False # remove ego vehicle iou
            pos_ious.append(max_iou_pred_to_gts[0][tp])

    avg_ious = torch.cat(pos_ious, dim=0).cpu().data.numpy()
    return avg_ious.sum() / n_gt, avg_ious.mean(), avg_ious


def cal_avg_ious(path, bev=False, thr=0.3):
    result = dict()
    histos = dict()
    result_pred_norm = dict()
    result_gt_norm = dict()
    for com in tqdm.tqdm(range(30, 70, 10)):
        result_pred_norm[com] = []
        result_gt_norm[com] = []
        result[com] = []
        histo = []
        labels = [0.3, 0.6, 0.7, 0.8, 1.0]
        for i in range(5):
            avg_iou_over_gt, avg_iou_over_pred, avg_ious = cal_avg_iou(path, com, i, thr=thr, bev=bev)
            result_pred_norm[com].append(avg_iou_over_pred.item())
            result_gt_norm[com].append(avg_iou_over_gt.item())
            result[com].append(avg_ious)
            bins = []
            for j in range(1, len(labels)):
                bins.append(np.logical_and(avg_ious > labels[j-1], avg_ious <= labels[j]).sum())
            bins = np.array(bins)
            histo.append(bins / bins.sum())
        histos[com] = histo
    return result_pred_norm, result_gt_norm, result, histos



def plot1(result, iou, ax):
    ious_idx = {0.3: 0, 0.5: 2, 0.7: 4}
    ax.set_ylim([0.6, 1])
    plt.xticks(list(range(5)))
    ax.grid()
    for r, aps in result.items():
        line, = ax.plot(list(range(5)), [ap[ious_idx[iou]] / 100.0 for ap in aps.values()], '^-')
        line.set_label(str(r) + 'm')
        ax.legend()
    ax.set_ylabel('mAP@IoU=' + str(iou))


def plot2(result, ax, com_range=50):
    data = result[com_range]
    # ax.set_ylim([0.0, 1])
    aps_ = [0.4, 0.5, 0.6, 0.7, 0.8]
    plt.xticks(aps_)
    for n, aps in data.items():
        line, = ax.plot(aps_, (np.array(aps[1:]) - aps[0])/ 100.0, '^-')
        line.set_label(str(n))
        ax.legend()
    ax.set_ylabel('mAP drop @ ' + str(com_range) + 'm')
    ax.set_xlabel('IoU')


def plot_iou(result, ax):
    plt.xticks(list(range(5)))
    ax.grid()
    for r, ious in result.items():
        line, = ax.plot(list(range(5)), ious, '^-')
        line.set_label(str(r) + 'm')
        ax.legend()


def plot_iou_bar(histos, com=50):
    histo = histos[com]
    labels = [0.3, 0.6, 0.7, 0.8]

    x = np.arange(len(labels)) # the label locations
    width = 0.8  # the width of the bars

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for i in range(len(histo)):
        rects = ax.bar(x + (i + 1) * width/len(histo), histo[i] * 100, width/len(histo), label=str(i))
        autolabel(rects, ax)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('TP predictions in %')
    # ax.set_title('Overall title')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.1f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def group_plots1(path):
    result = read_aps(path)
    # plot1
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharex=True)
    fig.text(0.5, 0.01, 'Number of cooperative vehicles', ha='center')
    plot1(result, 0.3, axes[0])
    plot1(result, 0.5, axes[1])
    plot1(result, 0.7, axes[2])

    fig.tight_layout()
    plt.savefig(os.path.join(path, 'v_num_r_com.png'))
    plt.close()

    # plot2
    file = os.path.join(path, 'avg_ious.json')
    if os.path.exists(file):
        with open(file, 'r') as fh:
            result = json.load(fh)
        histos = np.load(os.path.join(path, 'ious_histos.npy'), allow_pickle=True).item()
    else:
        result_pred_norm, result_gt_norm, ious, histos = cal_avg_ious(path, bev=True, thr=0.8)
        result = {'result_pred_norm': result_pred_norm,
                  'result_gt_norm': result_gt_norm}
        with open(file, 'w') as fh:
            json.dump(result, fh)
        np.save(os.path.join(path, 'ious_histos.npy'), histos, allow_pickle=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True)
    fig.text(0.5, 0.01, 'Number of cooperative vehicles', ha='center')
    axes[1].set_ylabel('Average IoU (normalized over GT boxes)')
    axes[0].set_ylabel('Average IoU (normalized over Pred. boxes)')
    plot_iou(result['result_pred_norm'], axes[0])
    plot_iou(result['result_gt_norm'], axes[1])
    fig.tight_layout()
    plt.savefig(os.path.join(path, 'avg_ious_line.png'))
    plt.close()

    plot_iou_bar(histos, 50)
    plt.savefig(os.path.join(path, 'ious_histo.png'))
    plt.close()


def noise_comparison(path):
    # noise comparison
    with_noise = read_aps(path + "kmeans_selection_noise")
    without_noise = read_aps(path + "kmeans_selection")
    stats = StatsRecorder()
    xs = np.array(range(5))
    ax = plt.subplot(111)
    plt.xticks(list(range(5)))
    plt.axhline(y=0.0, color='k', linestyle='--', linewidth=1)
    for r in [30, 40, 50, 60]:
        aps1 = [ap[4] for ap in without_noise[r].values()]
        aps2 = [ap[4] for ap in with_noise[r].values()]
        ap_diff = np.array(aps2) - np.array(aps1)
        stats.update(ap_diff.reshape(1, -1))
        line, = ax.plot(xs, ap_diff, '^--', alpha=0.5)
        line.set_label(str(r))
    line = plt.errorbar(xs, stats.mean, yerr=stats.std, fmt='ko-', elinewidth=2, capsize=5)
    ax.legend()
    ax.set_xlabel('Number of cooperative vehicles')
    ax.set_ylabel('Percentage drop of mAP@IoU0.7 by introducing noise')
    plt.savefig("/media/hdd/ophelia/koko/experiments-output/cia-ssd/noise_comparison.png")
    plt.close()


def nodes_selection_comparison(path):
    # noise comparison
    random_selection = read_aps(path + "com60-72m")
    kmeans_selection = read_aps(path + "kmeans_selection")
    stats = StatsRecorder()
    xs = np.array(range(5))
    ax = plt.subplot(111)
    plt.xticks(list(range(5)))
    plt.axhline(y=0.0, color='k', linestyle='--', linewidth=1)
    for r in [30, 40, 50, 60]:
        aps1 = [ap[4] for ap in random_selection[r].values()]
        aps2 = [ap[4] for ap in kmeans_selection[r].values()]
        ap_diff = np.array(aps2) - np.array(aps1)
        stats.update(ap_diff.reshape(1, -1))
        line, = ax.plot(xs, ap_diff, '^--', alpha=0.5)
        line.set_label(str(r))
    line = plt.errorbar(xs, stats.mean, yerr=stats.std, fmt='ko-', elinewidth=2, capsize=5)
    ax.legend()
    ax.set_xlabel('Number of cooperative vehicles')
    ax.set_ylabel('Difference of mAP@IoU0.7 with kmeans-delaunay and random selection')
    plt.savefig("/media/hdd/ophelia/koko/experiments-output/cia-ssd/nodes_selection_comparison.png")
    plt.close()


def _box_direction_acc(filename):
    data = torch.load(filename)
    samples = data['samples']
    pred_boxes = data['pred_boxes']
    gt_boxes = data['gt_boxes']
    confidences = data['confidences']
    list_sample = []
    list_confidence = []
    list_tp = []
    N_gt = 0
    ##############
    IoU_thr = 0.7
    bev = False
    ##############
    iou_fn = boxes_iou_bev if bev else boxes_iou3d_gpu
    for sample in samples:
        if len(pred_boxes[sample]) > 0 and len(gt_boxes[sample]) > 0:
            pred_dirs = pred_boxes[sample][:, -1]
            gt_dirs = gt_boxes[sample][:, -1]
            ious = iou_fn(pred_boxes[sample], gt_boxes[sample])
            n, m = ious.shape
            list_sample.extend([sample] * n)
            list_confidence.extend(confidences[sample])
            N_gt += len(gt_boxes[sample])
            max_iou_pred_to_gts = ious.max(dim=1)
            max_iou_gt_to_preds = ious.max(dim=0)
            tp = max_iou_pred_to_gts[0] > IoU_thr
            is_best_match = max_iou_gt_to_preds[1][max_iou_pred_to_gts[1]] \
                            == torch.tensor([i for i in range(len(tp))], device=tp.device)
            tp[torch.logical_not(is_best_match)] = False
            inds = max_iou_pred_to_gts[1].reshape(-1, 1)[tp].squeeze()
            pred_dirs = limit_period(pred_dirs.reshape(-1, 1)[tp])
            gt_dirs = limit_period(gt_dirs.reshape(-1, 1)[inds])
            diff_dir = torch.abs(pred_dirs - gt_dirs)
            tp_dir = torch.min(3.1415 * 2 - diff_dir, diff_dir) < 3.1415 / 2
            list_tp.append(tp_dir)
    tps = torch.cat(list_tp, dim=0)
    acc = tps.sum().item() / tps.shape[0]
    return acc


def box_direction_acc(path):
    result_filename = os.path.join(path, "direction_acc.txt")
    if os.path.exists(result_filename):
        accs = np.loadtxt(result_filename)
    else:
        accs = []
        for r in tqdm.tqdm([30, 40, 50, 60]):
            for n in range(5):
                filename = path + "/test_result_{:d}m/thr0.3_ncoop{:d}.pth".format(r, n)
                accs.append(_box_direction_acc(filename))
        accs = np.array(accs).reshape(4, 5)
        np.savetxt(os.path.join(path, "direction_acc.txt"), accs, fmt='%.3f')



if __name__=="__main__":
    path = "/media/hdd/ophelia/koko/experiments-output/cia-ssd/com60-72m"
    # nodes_selection_comparison(path)
    # noise_comparison(path)
    box_direction_acc(path)











