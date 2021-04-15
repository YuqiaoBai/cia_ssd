import os
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
from ops.roiaware_pool3d import roiaware_pool3d_utils
from utils.points_utils import add_gps_noise_bev
from utils.box_np_ops import limit_period
from glob import glob


class CoFusedDataset(Dataset):
    def __init__(self, cfg, training=True, n_coop='random', com_range=60):
        super().__init__()
        self.training = training
        self.n_coop = n_coop
        self.com_range = com_range
        self.cfg = cfg
        self.root = Path(self.cfg.root)
        self.add_gps_noise = cfg.add_gps_noise
        self.gps_noise_std = cfg.gps_noise_std

        # node selection
        # if cfg.node_selection_mode is not None:
        #     self.node_selection = np.load(os.path.join(cfg.root, cfg.node_selection_mode + '.npy'),
        #                                   allow_pickle=True).item()
        # else:
        #     self.node_selection = None
        if cfg.selected_points is not None:
            self.selected_points = np.load(os.path.join(cfg.root, cfg.selected_points + '.npy'),
                                          allow_pickle=True).item()
        else:
            self.selected_points= None

        # train or test
        if not Path(self.cfg.root + "/train_val.txt").exists():
            split(cfg)
        if self.mode == "train":
            with open(self.cfg.root + "/train_val.txt", "r") as f:
                self.file_list = f.read().splitlines()
        else:
            with open(self.cfg.root + "/test.txt", "r") as f:
                self.file_list = f.read().splitlines()

        self.coop_files = self.update_file_list()
        str = 'train val set: ' if self.training else 'test set: '
        print(str, len(self.file_list))

        # point label map, 0 as non-relevant classes
        self.label_color_map = {}
        for i, classes in self.cfg.classes.items():
            self.label_color_map[i] = [list(self.cfg.LABEL_COLORS.keys()).index(c) for c in classes]

        self.augmentor = None
        if self.training:
            if getattr(self.cfg, "AUGMENTOR", None):
                from datasets.cofused.augmentor import Augmentor
                self.augmentor = Augmentor(cfg)

        if hasattr(self.cfg, "TARGET_ASSIGNER"):
            from datasets import assign_target
            self.target_assigner = assign_target.AssignTarget(cfg.pc_range,
                                                              cfg=self.cfg.TARGET_ASSIGNER,
                                                              mode=self.mode)

        process_fn = self.cfg.process_fn["train"] if self.training else self.cfg.process_fn["test"]
        self.processors = []
        for fn in process_fn:
            if fn == "points_to_voxel":
                self.voxel_generator = VoxelGenerator(
                    voxel_size=self.cfg.voxel_size,
                    point_cloud_range=self.cfg.pc_range,
                    max_num_points=self.cfg.max_points_per_voxel,
                    max_voxels=self.cfg.max_num_voxels
                )
            self.processors.append(getattr(self, fn))
        if self.cfg.BOX_CODER['type'] == 'GroundBoxBevGridCoder':
            from models.box_coders import GroundBoxBevGridCoder
            self.box_coder = GroundBoxBevGridCoder(**self.cfg.BOX_CODER)

    @property
    def mode(self):
        return "train" if self.training else "test"

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        data_dict = self.load_data(index)
        if self.augmentor is not None:
            data_dict = self.augmentor.forward(data_dict)
        data_dict["gt_boxes"][:, 6] = limit_period(data_dict["gt_boxes"][:, 6], 0.5, 2 * np.pi)
        for processor in self.processors:
            data_dict = processor(data_dict)

        return self.drop_intermidiate_data(data_dict)

    def update_file_list(self):
        """
        Update file list according to number of cooperative vehicles, frames will be removed if
        the number of cooperative vehicles is less than the given number
        """
        coop_files_list = []
        selected = []
        for file in self.file_list:
            coop_files = glob(os.path.join(self.cfg.root, self.cfg.coop_cloud_name, file, '*.bin'))
            tfs = np.load(os.path.join(self.cfg.root, "tfs", file + '.npy'), allow_pickle=True).item()
            coop_ids = [file.rsplit("/")[-1][:-4] for file in coop_files]
            ego_loc = tfs['tf_ego'][0:2, -1]
            coop_locs = [tfs[coop_id][0:2, -1] for coop_id in coop_ids]
            # in communication range
            coop_in_com_range_inds = np.linalg.norm(np.array(coop_locs) - np.array(ego_loc), axis=1) < self.com_range
            coop_files = np.array(coop_files)[coop_in_com_range_inds].tolist()
            if self.training or self.n_coop == 'random' or self.n_coop == 0 or len(coop_files) >= self.n_coop:
                selected.append(file)
                coop_files_list.append(coop_files)
        self.file_list = selected
        return coop_files_list

    def load_data(self, index):
        # load point cloud and gt
        cloud_filename_ego = os.path.join(self.cfg.root, self.cfg.ego_cloud_name,
                                          self.file_list[index] + ".bin")
        bbox_filename = os.path.join(self.cfg.root, "label_box",
                                     self.file_list[index] + ".txt")
        cloud_ego = np.fromfile(cloud_filename_ego, dtype="float32").reshape(-1, 3)
        # data fix
        cloud_ego[:, :2] *= -1
        # create a list, first element is cloud_ego
        clouds = [cloud_ego]
        if not self.n_coop == 0:
            # load cooperative clouds with selected points
            coop_files = self.coop_files[index]
            if self.node_selection is not None:
                selected = []
                for i, f in enumerate(coop_files):
                    # which frame which coop
                    # selected_points 20*256*256
                    if f.rsplit("/")[-1][:-4] in self.node_selection[self.file_list[index]][self.n_coop - 1]:
                        selected.append(i)
            # selected points
            #
            # random choose points, compared with our algorithm
            if self.n_coop == 'random':
                selected = np.random.choice(list(np.arange(0, len(coop_files))),
                                            np.random.randint(0, len(coop_files) + 1), replace=False)
            # select n_coop out of coop_files randomly
            else:
                selected = np.random.choice(list(np.arange(0, len(coop_files))),
                                            self.n_coop, replace=False)

            for cf in selected.tolist():
                cloud_coop = np.fromfile(coop_files[cf], dtype="float32").reshape(-1, 3)
                if self.add_gps_noise:
                    cloud_coop = add_gps_noise_bev(cloud_coop, self.gps_noise_std)
                # data fix
                cloud_coop[:, :2] *= -1
                clouds.append(cloud_coop)

        clouds = np.concatenate(clouds, axis=0)
        gt_boxes = np.loadtxt(bbox_filename, dtype=str)[:, [2, 3, 4, 8, 9, 10, 7]].astype(np.float)

        # clouds = clouds[self._mask_points_in_box(clouds[:, :3], self.cfg.pc_range)]
        # from vlib.visulization import draw_points_boxes_plt
        # draw_points_boxes_plt(self.cfg.pc_range, clouds, None, gt_boxes, False)
        # print('debug')
        # gt_boxes[:, 3:6] = gt_boxes[:, 3:6] / 2
        batch_type = {
            "points": "cpu_float",
            "gt_boxes": "gpu_float",
            "frame": "cpu_none"
        }

        return {
            "points": clouds,
            "gt_boxes": gt_boxes,
            "gt_names": np.array(["Car"] * len(gt_boxes)),
            "gt_classes": np.array([1] * len(gt_boxes)),
            "frame": self.file_list[index],
            "batch_types": batch_type
        }

    def mask_points_in_range(self, data_dict):
        # mask points and boxes outside range
        cloud = data_dict["points"]  # (104557, 3)
        gt_boxes = data_dict["gt_boxes"]
        box_centers = gt_boxes[:, :3].astype(float)
        if self.cfg.pc_range[3] == self.cfg.pc_range[4] and False:
            cloud = cloud[self._mask_points_in_range(cloud[:, :], self.cfg.pc_range[3])]
            mask = self._mask_points_in_range(box_centers, self.cfg.pc_range[3])
        else:
            cloud = cloud[self._mask_points_in_box(cloud[:, :], self.cfg.pc_range)]
            mask = self._mask_points_in_box(box_centers, self.cfg.pc_range)
        data_dict["points"] = cloud[:, :3]
        data_dict["points_labels"] = cloud[:, 3:]
        data_dict["batch_types"]["points_labels"] = "gpu_long"
        data_dict["gt_boxes"] = gt_boxes[mask]
        data_dict["gt_names"] = data_dict["gt_names"][mask]
        data_dict["gt_classes"] = data_dict["gt_classes"][mask]
        return data_dict

    def rm_empty_gt_boxes(self, data_dict):
        cloud = data_dict["points"]
        gt_boxes = data_dict["gt_boxes"]
        point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(cloud, gt_boxes)
        selected = point_indices.sum(axis=1) > 5
        data_dict["gt_boxes"] = gt_boxes[selected]
        data_dict["gt_names"] = data_dict["gt_names"][selected]
        data_dict["gt_classes"] = data_dict["gt_classes"][selected]

        return data_dict

    def filter_point_label(self, data_dict):
        labels = data_dict["points_labels"]
        labels_filtered = np.zeros_like(labels)
        for k, vs in self.label_color_map.items():
            for v in vs:
                labels_filtered[labels == v] = k
        data_dict["points_labels"] = labels_filtered
        return data_dict

    def shift_points_to_ground_base(self, data_dict):
        # find ego_vehicle height remove z offset to groud
        cloud = data_dict["points"]
        gt_boxes = data_dict["gt_boxes"]
        idx_ego = np.where(np.abs(gt_boxes[:, :2].astype(np.float)).sum(axis=1) == 0)[0]
        ego_height = gt_boxes[idx_ego, 5]
        z_offset = 0.1 + ego_height
        cloud[:, 2] = cloud[:, 2] + z_offset
        gt_boxes[:, 2] = gt_boxes[:, 2] + z_offset
        self.range_offseted = self.cfg.pc_range.copy()
        self.range_offseted[[2, 5]] = self.range_offseted[[2, 5]] + z_offset
        self.voxel_generator._point_cloud_range = self.range_offseted

        data_dict["points"] = cloud
        data_dict["gt_boxes"] = gt_boxes
        return batch_data

    def shuffle_points(self, data_dict):
        cloud = data_dict["points"]
        labels = data_dict["points_labels"]
        shuffle_idx = np.random.permutation(cloud.shape[0])
        data_dict["points"] = cloud[shuffle_idx]
        data_dict["points_labels"] = labels[shuffle_idx]

        return data_dict

    def points_to_voxel(self, data_dict):
        cloud = data_dict["points"]
        voxel_output = self.voxel_generator.generate(cloud[:, :3])
        data_dict.update({
            "voxels": voxel_output["voxels"],
            "voxel_coords": voxel_output["coordinates"],
            "voxel_num_points": voxel_output["num_points_per_voxel"]
        })
        data_dict["batch_types"].update({
            "voxels": "gpu_float",
            "voxel_coords": "gpu_float",
            "voxel_num_points": "gpu_float"
        })

        return data_dict

    def get_bev_input(self, data_dict):
        points = data_dict["points"]
        bev = np.zeros(self.cfg.grid_size, dtype=np.bool)
        inds = ((points[:, :3] - self.cfg.pc_range[None, :3]) / np.array(self.cfg.voxel_size)[None, :]).astype(np.int)
        inds = [np.clip(inds[:, i], 0, self.cfg.grid_size[i]) for i in range(len(self.cfg.grid_size))]
        bev[inds[0], inds[1], inds[2]] = 1

        data_dict['bev_input'] = bev.transpose((2, 0, 1))
        data_dict["batch_types"].update({"bev_input": "gpu_float"})

        return data_dict

    def get_pixel_labels(self, data_dict):
        boxes = data_dict["gt_boxes"]
        label, reg = self.box_coder.encode(boxes, self.cfg)
        # plt.imshow(reg[..., 0], cmap="hot", interpolation="nearest")
        # plt.savefig("/media/hdd/ophelia/tmp/tmp.png")

        data_dict.update({
            "label_map": label[None, ...],
            "regression_map": reg.transpose((2, 0, 1))
        })
        data_dict["batch_types"].update({
            "label_map": "gpu_float",
            "regression_map": "gpu_float"
        })

        return data_dict

    def assign_target(self, data_dict):
        assert self.target_assigner is not None
        data_dict = self.target_assigner(data_dict)

        return data_dict

    def drop_intermidiate_data(self, data_dict):
        data_dict.pop('gt_names')
        data_dict.pop('gt_classes')
        # inds = np.random.choice(len(data_dict['points']), len(data_dict['points']) // 4)
        # data_dict['points'] = data_dict['points'][inds, :] # only for visualization
        data_dict.pop('points_labels')
        return data_dict

    def _mask_values_in_range(self, values, min, max):
        return np.logical_and(values > min, values < max)

    def _mask_points_in_box(self, points, pc_range):
        n_ranges = len(pc_range) // 2  # 3
        list_mask = [self._mask_values_in_range(points[:, i], pc_range[i],
                                                pc_range[i + n_ranges]) for i in range(n_ranges)]
        return np.array(list_mask).all(axis=0)

    def _mask_points_in_range(self, points, dist):
        return np.linalg.norm(points[:, :2], axis=1) < dist

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        batch_types = [data.pop("batch_types") for data in batch_list][0]
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)

        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ["voxels", "voxel_num_points"]:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ["points", "voxel_coords"]:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode="constant", constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ["points_labels"]:
                    max_gt = max([len(x) for x in val])
                    assert len(val[0].shape) == 2
                    batch_points = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_points[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_points
                elif key in ["encoded_gt_boxes"]:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ["frame", "gt_boxes", "positive_gt_id"]:
                    ret[key] = val
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print("Error in collate_batch: key=%s" % key)
                raise TypeError

        ret["batch_size"] = batch_size
        batch_types["batch_size"] = "cpu_none"
        ret["batch_types"] = batch_types
        return ret


def split(cfg):
    path_ego = Path(cfg.root) / "cloud_ego"
    list_train_val = []
    list_test = []
    for filename in path_ego.glob("*.bin"):
        if filename.name.split("_")[0] in cfg.train_val_split:
            list_train_val.append(filename.name[:-4])
        else:
            list_test.append(filename.name[:-4])

    with open(cfg.root + "/train_val.txt", "w") as fa:
        for line in list_train_val:
            fa.writelines(line + "\n")
    with open(cfg.root + "/test.txt", "w") as fb:
        for line in list_test:
            fb.writelines(line + "\n")


if __name__ == "__main__":
    from cfg.cia_ssd_cofused import dcfg

    train_dataset = CoFusedDataset(dcfg, training=True)
    sampler = None
    train_dataloader = DataLoader(
        train_dataset, batch_size=4, pin_memory=True, num_workers=1,
        shuffle=True, collate_fn=train_dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0
    )
    for batch_data in train_dataloader:
        pass
