import numpy as np
from glob import glob
from ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu as points_in_boxes


def read_gt_data(path):
    gt_boxes_path = path + "/label_box"
    clouds_path = path + "/cloud_ego"
    for file in glob(clouds_path + "/*.bin"):
        points = np.fromfile(file, "float32").reshape(-1, 4)
        bbox_filename = gt_boxes_path + "/" + file.rsplit("/")[-1].replace("bin", "txt")
        gt_boxes = np.loadtxt(bbox_filename, dtype=str)[:, [2,3,4,8,9,10,7]].astype(np.float)


if __name__=="__main__":
    path = "/media/hdd/ophelia/koko/data/synthdata_20veh_60m"
    data = read_gt_data(path)