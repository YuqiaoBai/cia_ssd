def get_grid_maps_hard_coded(self, data_dict=None, config=None):
    if data_dict is None:
        return partial(self.get_grid_maps, config=config)
    points = data_dict['points']
    size = config.MAP_SIZE
    map_detection = np.zeros(config.MAP_SIZE, dtype=np.int32)
    map_min = np.ones(config.MAP_SIZE, dtype=np.float32) * 1000
    map_max = - np.ones(config.MAP_SIZE, dtype=np.float32) * 1000
    energy = np.zeros(config.MAP_SIZE, dtype=np.float32)
    occlusion_mask = np.ones((int(np.ceil(np.sqrt(size[0] ** 2 + (0.5 * size[1]) ** 2))),
                              int(180 / config.AZIMUTH_RESOLUTION)), dtype=np.float32) * self.point_cloud_range[2]
    map_occlusion_height = np.zeros(size, dtype=np.float32)
    for p in points:
        row = int(np.floor((p[0] - self.point_cloud_range[0]) / config.MAP_RESOLUTION))
        col = int(np.floor((p[1] - self.point_cloud_range[1]) / config.MAP_RESOLUTION))
        if p[3] > 0:
            map_detection[row, col] += 1  # some points have 0 intensity
        map_min[row, col] = min(p[2], map_min[row, col])
        map_max[row, col] = max(p[2], map_max[row, col])
        energy[row, col] += p[3]
        range_idx = int(np.floor(np.linalg.norm(p[0:2]) / config.MAP_RESOLUTION))
        angle_idx = int(np.floor((np.arctan2(p[1], p[0]) + 0.5 * np.pi) / np.pi * 180 / config.AZIMUTH_RESOLUTION))
        for r, height in enumerate(occlusion_mask[range_idx:, angle_idx]):
            if height < p[2]:
                occlusion_mask[range_idx + r, angle_idx] = p[2]
    occupancy_mask = map_detection > 0
    map_diff = map_max - map_min + 1  # 1 means max==min in this cell, only 1 point falls in this cell
    map_diff[occupancy_mask == False] = -1
    map_intensity = energy / (map_detection + np.array(map_detection == 0, dtype=np.int32))
    for i in range(size[0]):
        for j in range(size[1]):
            x = i + 0.5
            y = (j - size[1] * 0.5) + 0.5
            range_idx = int(np.floor(np.sqrt(x ** 2 + y ** 2)))
            angle_idx = int(np.floor((np.arctan2(y, x) + 0.5 * np.pi) / np.pi * 180 / config.AZIMUTH_RESOLUTION))
            map_occlusion_height[i, j] = occlusion_mask[range_idx, angle_idx]

    data_dict['grid_maps'] = np.dstack((map_detection, map_diff, map_intensity, map_occlusion_height))

    return data_dict