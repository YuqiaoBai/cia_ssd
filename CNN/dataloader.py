import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from glob import glob
from collections import defaultdict
import open3d as o3d


class coopDataset(Dataset):
    def __init__(self, com_range=40, training=True):
        self.com_range = com_range
        self.training = training
        self.pc_range = np.array([-60.4, -60.4, -3, 60.4, 60.4, 1])
        self.root = '/media/ExtHDD01/mastudent/BAI/HybridV50CAV20'
        self.test_split = ['965', '224', '685', '924', '334', '1175', '139',
                           '1070', '1050', '1162', '1260']
        self.train_val_split = ['829', '943', '1148', '753', '599', '53', '905', '245', '421', '509']
        self.view_range = 60.4

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
        self.update_file_list()
        self.mask_points_in_range(data_dict)
        return self.drop_intermidiate_data(data_dict)

    @property
    def mode(self):
        return "train" if self.training else "test"

    def limit_period(self, val, offset=0.5, period=2 * np.pi):
        return val - np.floor(val / period + offset) * period

    def update_file_list(self):
        """
        Update file list according to communication range
        remove frame with no coop vehicle
        file_list = 'frame list'
        """
        coop_files_list = []
        selected = []
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
            if self.training or len(coop_files) > 0:
                selected.append(file)
                coop_files_list.append(coop_files)
        self.file_list = selected
        return coop_files_list # path list

    def mask_points_in_range(self, data_dict):
        # mask points and boxes outside range
        gt_boxes = data_dict["gt_boxes"]
        box_centers = gt_boxes[:, :3].astype(float)
        mask = self._mask_points_in_range(box_centers, self.pc_range[3])
        data_dict["batch_types"]["points_labels"] = "gpu_long"
        data_dict["gt_boxes"] = gt_boxes[mask]
        data_dict["gt_names"] = data_dict["gt_names"][mask]
        data_dict["gt_classes"] = data_dict["gt_classes"][mask]
        return data_dict

    def _mask_points_in_range(self, points, dist):
        return np.linalg.norm(points[:, :2], axis=1) < dist

    def load_data(self, index):
        grids = []
        tfs = np.load(os.path.join(self.root, "tfs", self.file_list[index] + '.npy'), allow_pickle=True).item()
        ego_loc = tfs['tf_ego'][0:2, -1]
        Map = self.load_map_patch(ego_loc)
        grids.append(Map)
        # load point cloud and gt
        cloud_filename_ego = os.path.join(self.root, 'cloud_ego',
                                          self.file_list[index] + ".bin")
        bbox_filename = os.path.join(self.root, "label_box",
                                     self.file_list[index] + ".txt")
        # ego point cloud
        cloud_ego = np.fromfile(cloud_filename_ego, dtype="float32").reshape(-1, 3)
        # data fix
        cloud_ego[:, :2] *= -1
        points = np.clip(cloud_ego, a_min=-self.view_range, a_max=self.view_range)
        ego_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud_ego_transformed = ego_cloud.transform(tfs['tf_ego'])
        points_ego = np.array(cloud_ego_transformed.points).astype(np.float32)

        ego_grid = self.pc2grid(points_ego, ego_loc)

        grids.append(ego_grid)

        # load cooperative clouds
        coop_files = self.coop_files[index]
        selected = []
        for i, f in enumerate(coop_files):
            # which frame which coop
            # selected_points 20*256*256
            selected.append(i)
        # select n_coop out of coop_files randomly

        for cf in selected:
            cloud_coop = np.fromfile(coop_files[cf], dtype="float32").reshape(-1, 3)
            # data fix
            cloud_coop[:, :2] *= -1
            coop_loc = tfs[coop_files[cf].rsplit("/")[-1][:-4]][0:2, -1]
            # transform coop cloud in world coordinate sys
            coop_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud_coop))
            cloud_coop_transformed = coop_cloud.transform(tfs[coop_files[cf].rsplit("/")[-1][:-4]])
            points_coop = np.array(cloud_coop_transformed.points).astype(np.float32)
            # in view range
            points_in_egoCS = points_coop[:, :2] - coop_loc
            points = np.clip(points_in_egoCS, a_min=-self.view_range, a_max=self.view_range)
            points = points[:, :2] + coop_loc
            grid = self.pc2grid(points, ego_loc)
            grids.append(grid)

        # fill with zeros
        while len(grids) < 21:
            a = np.zeros((256, 256))
            grids.append(a)
        grids = np.array(grids)
        gt_boxes = np.loadtxt(bbox_filename, dtype=str)[:, [0, 2, 3, 4, 8, 9, 10, 7]].astype(np.float)
        coop_ids = [file.rsplit("/")[-1][:-4] for file in coop_files]
        gt_idxs = [np.where(gt_boxes[:, 0] == float(coop_id)) for coop_id in coop_ids]
        gt_boxes = np.squeeze(gt_boxes[gt_idxs,:]).reshape(-1,8)
        gt_boxes = gt_boxes[:, 1:]
        while gt_boxes.shape[0] < 20:
            gt_boxes = np.insert(gt_boxes, 0, values=0, axis=0)

        batch_type = {
            "points": "cpu_float",
            "gt_boxes": "gpu_float",
            "frame": "cpu_none"
        }

        return {
            "points": grids,
            "gt_boxes": gt_boxes,
            "gt_names": np.array(["Car"] * len(gt_boxes)),
            "gt_classes": np.array([1] * len(gt_boxes)),
            "frame": self.file_list[index],
            "batch_types": batch_type
        }

    def pc2grid(self, points, ego_loc, grid_size=0.8, ceils=256):
        grid = np.zeros(shape=(ceils, ceils))
        x = points[:, 0] - ego_loc[0]
        y = points[:, 1] - ego_loc[1]
        inds_x = (x / grid_size + ceils / 2).astype(np.int)
        inds_y = (y / grid_size + ceils / 2).astype(np.int)
        grid[inds_x, inds_y] = 1
        return grid

    def load_map_patch(self, ego_loc, grid_size=0.8, ceils=256):
        map_bin = np.load('../data/map_colsed.npy').astype(int)
        map_size = map_bin.shape
        netoffset = np.array([270.80, 200.32])
        map_view_size = (np.array([ceils, ceils])).astype(np.int)
        l = [-ceils * grid_size / 2, -ceils * grid_size / 2, ceils * grid_size / 2, ceils * grid_size / 2]
        # view in grids
        x = np.arange(l[0], l[2], grid_size)
        y = np.arange(l[1], l[3], grid_size)
        xx, yy = np.meshgrid(x, y)
        points = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
        points = points + ego_loc
        points = points + netoffset
        inds = (points / 0.1).astype(np.int)
        inds_x = np.clip(inds[:, 0], a_min=0, a_max=map_size[0] - 1)
        inds_y = np.clip(inds[:, 1], a_min=0, a_max=map_size[1] - 1)
        view = map_bin[inds_x, inds_y].astype(int)
        view = view.reshape(map_view_size[0], map_view_size[1])
        return view

    def drop_intermidiate_data(self, data_dict):
        data_dict.pop('gt_names')
        data_dict.pop('gt_classes')
        # inds = np.random.choice(len(data_dict['points']), len(data_dict['points']) // 4)
        # data_dict['points'] = data_dict['points'][inds, :] # only for visualization
        return data_dict

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


# if __name__ == '__main__':
#
#     dataset = coopDataset()
#     dataloader = DataLoader(dataset, batch_size=6, shuffle=True)
#     for data in dataloader:
#         print(data["points"].shape)
