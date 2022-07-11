""" 
A minimal example in Pytorch(Lightning) showing training of Neural Net for 
getting better embedding representation of image using Triplet Loss with online
Hard Sample Mining.
"""
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50

from torch.utils.data import DataLoader, random_split, Dataset
import pytorch_lightning as pl

from pytorch_metric_learning import miners, losses

from PIL import Image
import random
from pathlib import Path
import numpy as np


class DogsCatsDataset(Dataset):
    def __init__(self, img_dir: str = '/Users/harshraj22/Downloads/data/dogs-vs-cats/train') -> None:
        self.img_dir = img_dir

    def __len__(self):
        return len(os.listdir(self.img_dir)) // 100

    def __getitem__(self, index):
        img_file = list(os.listdir(self.img_dir))[index]
        return torch.from_numpy(
                np.asarray(Image.open(Path(self.img_dir) / img_file).resize((250, 250)))
            ).view(3, 250, 250).float(), torch.tensor(random.randint(0, 1), dtype=torch.long)

dataset = DogsCatsDataset()
dataloader = DataLoader(dataset, batch_size=2)


class MyClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = resnet50(pretrained=False)
        # self.l2 = nn.Linear(1000, 2)
        self.miner = miners.MultiSimilarityMiner()
        self.loss_func = losses.TripletMarginLoss()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.l1(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        embedding = self(x)
        
        # https://kevinmusgrave.github.io/pytorch-metric-learning/
        hard_pair = self.miner(embedding, y)
        loss = self.loss_func(embedding, y, hard_pair)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

trainer = pl.Trainer()
trainer.fit(MyClassifier(), dataloader)
