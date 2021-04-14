import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import open3d as o3d
from tqdm import tqdm


def read_meta_info(meta_file):
    id = meta_file.split('/')[-3]
    with open(meta_file, 'r') as f:
        line = f.readline().split(',')
        frame = line[0]
        line = f.readline().split(',')
        rot = - np.array([float(l) / 180 * np.pi for l in line[:3]], dtype=np.float64)
        # rot[2] = - rot[2]
        loc = np.array([[float(line[3])], [-float(line[4])], [float(line[5])]], dtype=np.float64)
        line = f.readline().split(',')
        channels = np.array([int(c) for c in line], dtype=np.int32)

        return frame, id, rot, loc, channels


def plot_rings(path):
    for filename in glob(os.path.join(path, "*.pcd")):
        pcd= o3d.io.read_point_cloud(filename)
        frame, id, rot, loc, channels = read_meta_info(filename.replace('.pcd', '_meta.txt'))
        points_np = np.array(pcd.points).astype(np.float32)
        idx_begin = 0
        idx_end = 0
        plt.figure(figsize=(8, 8))
        for c in channels:
            idx_end += c
            cur_points = points_np[idx_begin:idx_end]
            plt.plot(cur_points[:, 0], cur_points[:, 1], '.', markersize=0.3)
            idx_begin = idx_end
        plt.savefig("/media/hdd/ophelia/tmp/tmp.png")
        plt.close()


def add_sensor_noise(points, std_ver=0.05, std_hor=0.05, std_dist_min=0.02, std_dist_max=0.06):
    """
    Args:
        std_ver: angle noise std in vertiacal direction
        std_hor: angle noise std in horizontal direction
        std_dist_max: distance std at the maximum distance
    """
    dist = np.linalg.norm(points[:, :3], axis=1)
    atmosphere_attenuation_rate = 0.002
    intensity = np.exp(- atmosphere_attenuation_rate * dist)
    drop = np.random.uniform(0, 1, dist.size) < intensity
    std_dist = std_dist_min + (std_dist_max - std_dist_min) * dist / 80
    dist = dist + np.random.randn(dist.size) * std_dist
    depth = np.linalg.norm(points[:, :2], axis=1)
    angle_hor = - np.arctan2(points[:, 1], points[:, 0]) / np.pi * 180 + np.random.randn(dist.size) * std_hor
    angle_ver = np.arctan2(points[:, 2], depth) / np.pi * 180 + np.random.randn(dist.size) * std_ver
    noisy_points = np.zeros_like(points)
    noisy_points[:, 2] = dist * np.sin(angle_ver / 180 * np.pi)
    depth = dist * np.cos(angle_ver / 180 * np.pi)
    noisy_points[:, 0] = depth * np.cos(angle_hor / 180 * np.pi)
    noisy_points[:, 1] = - depth * np.sin(angle_hor / 180 * np.pi)
    noisy_points[:, -1] = points[:, -1]
    noisy_points = noisy_points[drop]
    # mask points by range
    pc_range = [0, -40, -3, 70.4, 40, 7]
    points_ = _mask_points_in_box(points, pc_range)
    noisy_points_ = _mask_points_in_box(noisy_points, pc_range)
    fig = plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.plot(points_[:, 0], points_[:, 1], '.', markersize=1)
    plt.subplot(122)
    plt.plot(noisy_points_[:, 0], noisy_points_[:, 1], '.', markersize=1)
    plt.savefig("/media/hdd/ophelia/tmp/tmp.png")
    plt.close()

    return noisy_points


def _mask_points_in_box(points, pc_range):
    n_ranges = len(pc_range) // 2
    list_mask = [_mask_values_in_range(points[:,i], pc_range[i],
                                            pc_range[i+n_ranges]) for i in range(n_ranges)]
    return points[np.array(list_mask).all(axis=0)]


def _mask_values_in_range(values, min,  max):
    return np.logical_and(values>min, values<max)




if __name__=="__main__":
    path = "/media/hdd/ophelia/koko/data/synthdata_20veh_60m/cloud_ego"
    os.makedirs(path.replace('cloud', 'noisy_cloud'), exist_ok=True)
    os.makedirs(path.replace('cloud_ego', 'noisy_cloud_coop'), exist_ok=True)
    for file in tqdm(glob(os.path.join(path, '*.bin'))):
        points_ego = np.fromfile(file, 'float32').reshape(-1, 4)
        points_ego = add_sensor_noise(points_ego)
        points_ego.astype('float32').tofile(file.replace('cloud', 'noisy_cloud'))
        coop_dir = file.replace('cloud_ego', 'cloud_coop_in_egoCS')[:-4]
        os.makedirs(coop_dir.replace('cloud_coop_in_egoCS', 'noisy_cloud_coop'), exist_ok=True)
        for coop in glob(coop_dir + '/*.bin'):
            points_coop = np.fromfile(coop, 'float32').reshape(-1, 4)
            points_coop = add_sensor_noise(points_coop)
            points_coop.astype('float32').tofile(coop.replace('cloud_coop_in_egoCS', 'noisy_cloud_coop'))



