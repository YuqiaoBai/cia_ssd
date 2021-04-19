import torch.utils.data as data
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import os
import json


class coopDataset(data.Dataset):
    def __init__(self):
        self.coop_data = torch.from_numpy(np.load('./data/input_data.npy', allow_pickle=True))


    def __len__(self):
        return len(self.coop_data)

    def __getitem__(self, idx):
        coop_data = self.coop_data[idx]
        return coop_data


# if __name__ == '__main__':
#
#     root_path = '/media/ExtHDD01/mastudent/BAI/HybridV50CAV20'
#     data = coopDataset()
