import numpy as np
import math
import os
from glob import glob
from tqdm import tqdm
import open3d as o3d
import matplotlib.pyplot as plt
import torch


def load_map_patch(ego_loc, grid_size=0.8, ceils=256):
    map_bin = np.load('../data/map_colsed.npy').astype(int)
    map_size = map_bin.shape
    netoffset = np.array([270.80, 200.32])
    map_view_size = (np.array([ceils, ceils])).astype(np.int)
    l = [-ceils * grid_size/2, -ceils * grid_size/2, ceils * grid_size/2, ceils * grid_size/2]
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


def pc2grid(points, ego_loc,  grid_size=0.8, ceils=256):
    grid = np.zeros(shape=(ceils, ceils))
    x = points[:, 0] - ego_loc[0]
    y = points[:, 1] - ego_loc[1]
    inds_x = (x/grid_size + ceils/2).astype(np.int)
    inds_y = (y/grid_size + ceils/2).astype(np.int)
    grid[inds_x, inds_y] = 1
    return grid


def load_data(root_path, communication_range=40, grid_size=0.8, ceils=256):
    view_range = ceils * grid_size / 2 - communication_range
    frames = os.listdir(os.path.join(root_path, "cloud_coop"))
    input_data = []
    # delete old input array file
    if os.path.exists('../data/input_data.npy'):
        os.remove('../data/input_data.npy')

    for frame in tqdm(frames):
        grids = []
        tf = np.load(os.path.join(root_path, 'tfs', frame + ".npy"), allow_pickle=True).item()
        ego_loc = tf['tf_ego'][0:2, -1]
        # map patch
        Map = load_map_patch(ego_loc)
        grids.append(Map)
        # ego point cloud
        ego_path = os.path.join(root_path, 'cloud_ego', frame + '.bin')
        cloud_ego = np.fromfile(ego_path, dtype=np.float32, count=-1).reshape([-1, 4])
        #cloud_ego[:, :2] *= -1
        cloud_ego = cloud_ego[:,0:3]
        points = np.clip(cloud_ego, a_min=-view_range, a_max=view_range)
        ego_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud_ego_transformed = ego_cloud.transform(tf['tf_ego'])
        points_ego = np.array(cloud_ego_transformed.points).astype(np.float32)

        ego_grid = pc2grid(points_ego, ego_loc)
        grids.append(ego_grid)

        for v in glob(os.path.join(root_path, "cloud_coop", frame, "*.bin")):
            # if in cmmunication_range
            coop_loc = tf[v.split('/')[-1][:6]][0:2, -1]
            dis = math.hypot((coop_loc - ego_loc)[0], (coop_loc - ego_loc)[1])
            if dis < communication_range:
                # read coop cloud
                cloud_coop = np.fromfile(v, dtype=np.float32, count=-1).reshape([-1, 4])
                # data fix
                #cloud_coop[:, :2] *= -1
                cloud_coop = cloud_coop[:,0:3]

                # transform coop cloud in world coordinate sys
                coop_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud_coop))
                cloud_coop_transformed = coop_cloud.transform(tf[v.rsplit("/")[-1][:-4]])
                points_coop = np.array(cloud_coop_transformed.points).astype(np.float32)

                # in view range
                points_in_egoCS = points_coop[:, :2] - coop_loc
                points = np.clip(points_in_egoCS, a_min=-view_range, a_max=view_range)
                points = points[:, :2] + coop_loc
                grid = pc2grid(points, ego_loc)
                grids.append(grid)
        # fill with zeros
        while len(grids) < 21:
            a = np.zeros((256,256))
            grids.append(a)
        test1 = torch.Tensor(grids)
        test = torch.sum(test1, dim=0)
        plt.imshow(test)
        plt.savefig('test.png')
        input_data.append(grids)

        # write data in file
        np.save('../data/input_data.npy', input_data)
    return input_data



#root_path = '/media/ExtHDD01/mastudent/BAI/HybridV50CAV20'
root_path = '/media/ExtHDD01/mastudent/formated_data'
grid_size = 0.8
ceils = 256
communication_range = 40
view_range = ceils*grid_size/2 - communication_range
input_data = load_data(root_path)