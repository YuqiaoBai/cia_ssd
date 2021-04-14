from model import ConvNet
import numpy as np
from dataloader import coopDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from os import path
import sys
from tools.test2 import test
from utils.train_utils import cfg_from_py


def train(train_data, model):
    data_size = len(train_data)
    model.train()
    print(data_size)
    return


if __name__ == '__main__':
    epoch = 2
    batch_size = 20

    # Load data
    train_data = coopDataset()
    train_loader = DataLoader(train_data, batch_size=batch_size)
    print(train_loader)

    # Model
    Coop = ConvNet()

    # Train
    for epoch in range(epoch):
        cfgs = cfg_from_py('cia_ssd_cofused')
        train_loss = train(train_loader, Coop)
        test((cfg() for cfg in cfgs))

