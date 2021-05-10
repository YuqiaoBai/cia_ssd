import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from glob import glob
from collections import defaultdict
import open3d as o3d
import matplotlib.pyplot as plt
from cnn_utils import draw_box_plt

class coopDataset(Dataset):
    def __init__(self, com_range=40, view_range=62.4,  training=True):
        self.com_range = com_range
        self.training = training
        self.root = '/media/ExtHDD01/mastudent/BAI/HybridV50CAV20'
        self.view_range = view_range
        self.map_bin = np.load('../data/map_colsed.npy').astype(int)
        self.test_split = ['965', '224', '685', '924', '334', '1175', '139',
                           '1070', '1050', '1162', '1260']
        self.train_val_split = ['829', '943', '1148', '753', '599', '53', '905', '245', '421', '509']
        # train or test
        if not Path(self.root + "/train_val.txt").exists():
            # method taht writes train_val and test txt file
            self.split()
        # read train or test data txt file
        if self.mode == "train":
            with open(self.root + "/train_val.txt", "r") as f:
                self.file_list = f.read().splitlines()
        else:
            with open(self.root + "/test.txt", "r") as f:
                self.file_list = f.read().splitlines()
        # remove vehicle out of communication range
        self.coop_files = self.update_file_list()  # path list


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data_dict = self.load_data(idx)
        return data_dict

    def update_file_list(self):
        """
        Update file list according to communication range
        remove frame with no coop vehicle
        file_list = 'frame list'
        """
        coop_files_list = []
        selected = []
        # file list from read txt file
        for file in self.file_list:
            # all coop bin files
            coop_files = glob(os.path.join(self.root, 'cloud_coop', file, '*.bin'))
            # get all the location infos(ego and coop)
            tfs = np.load(os.path.join(self.root, "tfs", file + '.npy'), allow_pickle=True).item()
            coop_ids = [file.rsplit("/")[-1][:-4] for file in coop_files]
            ego_loc = tfs['tf_ego'][0:2, -1]
            coop_locs = [tfs[coop_id][0:2, -1] for coop_id in coop_ids]
            # in communication range
            coop_in_com_range_inds = np.linalg.norm(np.array(coop_locs) - np.array(ego_loc), axis=1) < self.com_range
            # vehicles in com_range in same frame
            coop_files = np.array(coop_files)[coop_in_com_range_inds].tolist() # path
            # remove frame with no coop vehicle
            if len(coop_files) > 0:
                # selected frame
                selected.append(file)
                coop_files_list.append(coop_files)
        self.file_list = selected
        return coop_files_list # path list

    @property
    def mode(self):
        return "train" if self.training else "test"

    def load_data(self, index):
        # load point cloud and gt
        cloud_filename_ego = os.path.join(self.root, 'cloud_ego',
                                          self.file_list[index] + ".bin")
        tfs = np.load(os.path.join(self.root, "tfs", self.file_list[index] + '.npy'), allow_pickle=True).item()
        cloud_ego = np.fromfile(cloud_filename_ego, dtype="float32").reshape(-1, 3)
        # data fix
        cloud_ego[:, :2] *= -1
        points = np.clip(cloud_ego, a_min=-self.view_range, a_max=self.view_range)
        ego_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud_ego_transformed = ego_cloud.transform(tfs['tf_ego'])
        points_ego = np.array(cloud_ego_transformed.points).astype(np.float32)
        clouds = [points_ego[:,:2]]

        # load cooperative clouds
        coop_files = self.coop_files[index]

        selected = []


        for cf in coop_files:
            cloud_coop = np.fromfile(cf, dtype="float32").reshape(-1, 3)
            # data fix
            cloud_coop[:, :2] *= -1
            coop_loc = tfs[cf.rsplit("/")[-1][:-4]][0:2, -1]
            # transform coop cloud in world coordinate sys
            coop_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud_coop))
            cloud_coop_transformed = coop_cloud.transform(tfs[cf.rsplit("/")[-1][:-4]])
            points_coop = np.array(cloud_coop_transformed.points).astype(np.float32)
            # in view range
            points_in_egoCS = points_coop[:, :2] - coop_loc
            points = np.clip(points_in_egoCS, a_min=-self.view_range, a_max=self.view_range)
            points = points[:, :2] + coop_loc
            clouds.append(points)

        clouds = np.concatenate(clouds, axis=0)
        gt_boxes_all = np.loadtxt(os.path.join(self.root, 'vehicle_info', 'j' +
                                               self.file_list[index].split('_')[0] + '.csv'), delimiter=",", skiprows=1,
                                  usecols=(0, 2, 3, 4, 5, 9, 10, 11, 8)).astype(np.float)
        gt_boxes = gt_boxes_all[gt_boxes_all[:, 0] == int(self.file_list[index].split('_')[1])]
        gt_boxes = gt_boxes[:, 1:]

        coop_ids = [file.rsplit("/")[-1][:-4] for file in coop_files]
        gt_idxs = [np.where(gt_boxes[:, 0] == float(coop_id)) for coop_id in coop_ids]
        gt_boxes = np.squeeze(gt_boxes[gt_idxs, :]).reshape(-1, 8)
        gt_boxes = gt_boxes[:, 1:]

        batch_type = {
            "points": "cpu_float",
            "gt_boxes": "gpu_float",
            "frame": "cpu_none"
        }

        # =============================vis============================================
        ax = plt.figure(figsize=(8, 8)).add_subplot(1, 1, 1)
        ax.set_aspect('equal', 'box')
        points = clouds
        ax.plot(points[:, 0], points[:, 1], 'b.', markersize=0.3)
        ax = draw_box_plt(gt_boxes, ax, color='red')
        # ax = draw_box_plt(pred_boxes[0], ax, color='red')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('temp2.png')
        plt.close()
        # ==============================================================================
        return {
            "points": clouds,
            "gt_boxes": gt_boxes,
            "frame": self.file_list[index],
            "batch_types": batch_type,
        }

    def split(self):
        """
        write train and test txt files
        :param cfg:
        :return:
        """
        path_ego = Path(self.root) / "cloud_ego"
        list_train_val = []
        list_test = []
        for filename in path_ego.glob("*.bin"):
            if filename.name.split("_")[0] in self.train_val_split:
                list_train_val.append(filename.name[:-4])
            else:
                list_test.append(filename.name[:-4])

        with open(self.root + "/train_val.txt", "w") as fa:
            for line in list_train_val:
                fa.writelines(line + "\n")
        with open(self.root + "/test.txt", "w") as fb:
            for line in list_test:
                fb.writelines(line + "\n")



if __name__ == '__main__':

    dataset = coopDataset()
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True)
    for data in dataloader:
        print(data["points"].shape)
