import numpy as np
import torch

n_object = torch.zeros(5,3)
n_predict = torch.zeros(5,3)

intersection = (n_object & n_predict)
union = (n_object | n_predict)

iou = intersection/union

print(iou)