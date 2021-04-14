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


class CoMapTFDataset(Dataset):
    def __init__(self, cfg, training=True, n_coop='random'):
        super().__init__()
        self.training = training
        self.n_coop = n_coop
        self.cfg = cfg
        self.root = Path(self.cfg.root)
        self.add_gps_noise = cfg.add_gps_noise
        self.gps_noise_std = cfg.gps_noise_std
        if not Path(self.cfg.root + "/train_val.txt").exists():
            split(cfg)
        if self.mode=="train":
            with open(self.cfg.root + "/train_val.txt", "r") as f:
                self.file_list = f.read().splitlines()
        else:
            with open(self.cfg.root + "/test.txt", "r") as f:
                self.file_list = f.read().splitlines()
                self.update_file_list()

        self.augmentor = None
        if self.training:
            if getattr(self.cfg, "AUGMENTOR", None):
                from  datasets.comap.augmentor import Augmentor
                self.augmentor = Augmentor(cfg)

        process_fn = self.cfg.process_fn["train"] if self.training else self.cfg.process_fn["test"]
        self.processors = []
        for fn in process_fn:
            if fn=="points_to_voxel":
                self.voxel_generator = VoxelGenerator(
                    voxel_size=self.cfg.voxel_size,
                    point_cloud_range=self.cfg.pc_range,
                    max_num_points=self.cfg.max_points_per_voxel,
                    max_voxels=self.cfg.max_num_voxels
                )
            self.processors.append(getattr(self, fn))

    @property
    def mode(self):
        return "train" if self.training else "test"

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        data_dict = self.load_data(index)
        for processor in self.processors:
            data_dict = processor(data_dict)

        return data_dict

    def update_file_list(self):
        """
        Update file list according to number of cooperative vehicles, frames will be removed if
        the number of cooperative vehicles is less than the given number
        """
        if isinstance(self.n_coop, int) and self.n_coop>0:
            selected = []
            for file in self.file_list:
                coop_files = glob(os.path.join(self.cfg.root, "cloud_coop_in_egoCS", file, '*.bin'))
                if len(coop_files)>=self.n_coop:
                    selected.append(file)
            self.file_list = selected

    def load_data(self, index):
        # load point cloud and gt
        cloud_filename_ego = os.path.join(self.cfg.root, "cloud_ego",
                                      self.file_list[index] + ".bin")
        cloud_ego = np.fromfile(cloud_filename_ego, dtype="float32").reshape(-1, 4)[:, :3]
        # randomly load cooperative cloud
        coop_files = glob(os.path.join(self.cfg.root, "cloud_coop_in_egoCS", self.file_list[index], '*.bin'))
        selected = np.random.choice(list(np.arange(0, len(coop_files))), 1, replace=False)[0]

        cloud_coop = np.fromfile(coop_files[selected], dtype="float32").reshape(-1, 4)[:, :3]
        cloud_coop, noise = add_gps_noise_bev(cloud_coop, self.gps_noise_std)

        batch_type = {
            "src_points": "gpu_float",
            "tgt_points": "gpu_float",
            "frame": "cpu_none",
            "reg_target": "gpu_float"
        }

        return {
            "src_points": cloud_coop,
            "tgt_points": cloud_ego,
            "reg_target": np.array(noise).reshape(3,) / np.array([self.gps_noise_std])[0, [0, 1, 5]],
            "frame": self.file_list[index],
            "batch_types": batch_type
        }

    def filter_ground_points(self, data_dict):
        ego_cloud = data_dict["tgt_points"][:, :3]
        coop_cloud = data_dict["src_points"][:, :3]
        # find plane hight using ransac
        h = 0
        max_consensus = 0
        for i in range(50):
            sample_h = np.random.choice(ego_cloud[:, 2], )
            num_consensus = (np.abs(ego_cloud[:, 2] - sample_h) < 0.2).sum()
            if num_consensus > max_consensus:
                max_consensus = num_consensus
                h = sample_h
        h = h + 0.2
        ego_cloud = ego_cloud[ego_cloud[:, 2] > h]
        coop_cloud = coop_cloud[coop_cloud[:, 2] > h]
        data_dict["tgt_points"] = ego_cloud
        data_dict["src_points"] = coop_cloud
        return data_dict

    def mask_points_in_range(self, data_dict):
        src = data_dict["src_points"]
        tgt = data_dict["tgt_points"]
        data_dict["src_points"] = src[self._mask_points_in_range(src[:, :-1], self.cfg.pc_range[3])]
        data_dict["tgt_points"] = tgt[self._mask_points_in_range(tgt[:, :-1], self.cfg.pc_range[3])]
        return data_dict

    def points_to_voxel(self, data_dict):
        cloud_src = data_dict["src_points"]
        cloud_tgt = data_dict["src_points"]
        voxel_output_src = self.voxel_generator.generate(cloud_src[:, :3])
        voxel_output_tgt = self.voxel_generator.generate(cloud_tgt[:, :3])
        data_dict.update({
            "src_voxels": voxel_output_src["voxels"],
            "src_voxel_coords": voxel_output_src["coordinates"],
            "src_voxel_num_points": voxel_output_src["num_points_per_voxel"],
            "tgt_voxels": voxel_output_tgt["voxels"],
            "tgt_voxel_coords": voxel_output_tgt["coordinates"],
            "tgt_voxel_num_points": voxel_output_tgt["num_points_per_voxel"]
        })
        data_dict["batch_types"].update({
            "src_voxels": "gpu_float",
            "src_voxel_coords": "gpu_float",
            "src_voxel_num_points": "gpu_float",
            "tgt_voxels": "gpu_float",
            "tgt_voxel_coords": "gpu_float",
            "tgt_voxel_num_points": "gpu_float"
        })

        return data_dict

    def _mask_values_in_range(self, values, min,  max):
        return np.logical_and(values>min, values<max)

    def _mask_points_in_box(self, points, pc_range):
        n_ranges = len(pc_range) // 2
        list_mask = [self._mask_values_in_range(points[:,i], pc_range[i],
                                                pc_range[i+n_ranges]) for i in range(n_ranges)]
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
                if key in ["src_voxels", "src_voxel_num_points", "tgt_voxels", "tgt_voxel_num_points"]:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ["src_points",  "src_voxel_coords", "tgt_points",  "tgt_voxel_coords"]:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode="constant", constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ["points_labels"]:
                    max_gt = max([len(x) for x in val])
                    assert len(val[0].shape)==2
                    batch_points = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_points[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_points
                elif key in ["frame"]:
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

if __name__=="__main__":
    from cfg.cia_ssd_comap import dcfg
    train_dataset = CoMapDataset(dcfg, training=True)
    sampler = None
    train_dataloader = DataLoader(
        train_dataset, batch_size=4, pin_memory=True, num_workers=1,
        shuffle=True, collate_fn=train_dataset.collate_batch,
        drop_last=False, sampler=sampler, timeout=0
    )
    for batch_data in train_dataloader:
        pass



