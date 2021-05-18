from ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
import numpy as np
import torch
points = np.stack(np.meshgrid(
    np.arange(-128, 128, 0.8),
    np.arange(-128, 128, 0.8),
    np.arange(-128, 128, 0.8),), axis=0).reshape(3, -1).T.astype(np.float32)[None, ...]
print(points.shape)
print(points)

res = points_in_boxes_gpu(
    torch.tensor(points, device='cuda:0'),
    torch.tensor([0, 0, 1.4, 4, 4, 4, 0], device='cuda:0').float().unsqueeze(0).unsqueeze(0)
)
print((res != -1).sum())