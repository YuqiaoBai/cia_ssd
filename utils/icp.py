import open3d as o3d
from pathlib import Path
import numpy as np
import copy
import matplotlib.pyplot as plt
from utils.points_utils import add_gps_noise_bev


def registrate_pair(source, target, xy_offset):
    # downsample
    voxel_size = 0.2
    source = o3d.geometry.PointCloud.voxel_down_sample(source, voxel_size)
    target = o3d.geometry.PointCloud.voxel_down_sample(target, voxel_size)
    # filter non-overlap points
    source_np = np.array(source.points).astype(np.float32)
    target_np = np.array(target.points).astype(np.float32)

    plt.plot(target_np[:, 0], target_np[:, 1], 'y.', markersize=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('/media/hdd/ophelia/tmp/tmp.png')
    r = 60
    xy_offset = np.array(xy_offset).reshape((1, 2))
    overlap_pts_src_inds = np.logical_and(np.linalg.norm(source_np[:, :2], axis=1) < r,
                                          np.linalg.norm(source_np[:, :2]-xy_offset, axis=1) < r)
    overlap_pts_tgt_inds = np.logical_and(np.linalg.norm(target_np[:, :2], axis=1) < r,
                                          np.linalg.norm(target_np[:, :2]-xy_offset, axis=1) < r)
    overlap_pts_src = source_np[overlap_pts_src_inds]
    overlap_pts_src[:, 2] = 0
    overlap_pts_tgt = target_np[overlap_pts_tgt_inds]
    overlap_pts_tgt[:, 2] = 0
    plt.plot(overlap_pts_src[:, 0], overlap_pts_src[:, 1], 'r.', markersize=0.3)
    plt.plot(overlap_pts_tgt[:, 0], overlap_pts_tgt[:, 1], 'b.', markersize=0.1)
    plt.savefig('/media/hdd/ophelia/tmp/tmp.png')
    plt.close()
    transf_init = np.diag(np.ones(4, dtype=np.float))
    threshold = 0.1
    source.points = o3d.utility.Vector3dVector(overlap_pts_src)
    target.points = o3d.utility.Vector3dVector(overlap_pts_tgt)
    evaluation = o3d.registration.evaluate_registration(source, target, threshold, transf_init)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, transf_init,
        o3d.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    pass




if __name__=="__main__":
    data_path = Path('/media/hdd/ophelia/koko/data/synthdata_20veh_50m')
    ego_clouds_files = (data_path / 'cloud_ego').glob('*.bin')
    noise = [0.5, 0.5, 0.0, 0.0, 0.0, 3.0]
    for ego_file in ego_clouds_files:
        ego_cloud = np.fromfile(str(ego_file), 'float32').reshape(-1, 4)[:, :3]
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
        ego_cloud[:, 2] = 0

        pcd_ego = o3d.geometry.PointCloud()
        pcd_ego.points = o3d.utility.Vector3dVector(ego_cloud)
        pcd_coop = o3d.geometry.PointCloud()
        coop_files = Path(ego_file.__str__()[:-4].replace('ego', 'coop')).glob('*.bin')
        for coop_file in coop_files:
            coop_cloud = np.fromfile(str(coop_file), 'float32').reshape(-1, 4)[:, :3]
            coop_cloud = coop_cloud[coop_cloud[:, 2] > h]
            coop_cloud[:, 2] = 0
            coop_cloud = add_gps_noise_bev(coop_cloud, noise)
            pcd_coop.points = o3d.utility.Vector3dVector(coop_cloud)
            # get transformations
            tf_file = Path(ego_file.__str__()[:-4].replace('cloud_ego', 'tfs') + '.npy')
            tfs = np.load(tf_file, allow_pickle=True).item()
            tf_coop = tfs[coop_file.__str__().rsplit('/')[-1][:-4]]
            xy_offset = [tf_coop[0, -1] - tfs['tf_ego'][0, -1], tf_coop[1, -1] - tfs['tf_ego'][1, -1]]
            # registration
            registrate_pair(pcd_coop, pcd_ego, xy_offset)


