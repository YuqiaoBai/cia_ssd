from model import ConvNet
import numpy as np
from dataloader import coopDataset
from torch.utils.data import DataLoader
from tools.test_bai import test



def train(train_data, model):
    model.train()
    test
    return


if __name__ == '__main__':
    epoch = 2
    batch_size = 20

    # Load data
    train_data = coopDataset()
    train_loader = DataLoader(train_data, batch_size=batch_size)

    # Model
    Coop = ConvNet()

    # Train
    for epoch in range(epoch):
        train_loss = train(train_loader, Coop)

