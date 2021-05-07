# https://arxiv.org/pdf/1712.08036.pdf

import torch
import torch.nn as nn



class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()

        # input.shape: (1, 105, 105)

        self.cnn1 = nn.Conv2d(1, 64, kernel_size=10) # (1, 105, 105) -> (64, 96, 96)
        self.relu1 = nn.ReLU()

        # feature maps.shape: (64, 96, 96)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2) # (64, 96, 96) -> (64, 48, 48)

        # feature maps.shape: (64, 48, 48)
        self.cnn2 = nn.Conv2d(64, 128, kernel_size=7) # (64, 48, 48) -> (128, 42, 42)
        self.relu2 = nn.ReLU()

        # feature maps.shape: (128, 42, 42)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2) # (128, 42, 42) -> (128, 21, 21)

        # feature maps.shape: (128, 21, 21)
        self.cnn3 = nn.Conv2d(128, 128, kernel_size=4) # (128, 21, 21) -> (128, 18, 18)
        self.relu3 = nn.ReLU()

        # feature maps.shape: (128, 18, 18)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2) # (128, 18, 18) -> (128, 9, 9)

        # feature maps.shape: (128, 9, 9)
        self.cnn4 = nn.Conv2d(128, 256, kernel_size=4) # (128, 9, 9) -> (256, 6, 6)
        self.relu4 = nn.ReLU()

        # feature maps.shape: (256, 6, 6)
        # torch.flatten: (256, 6, 6) -> (9216)

        # feature maps.shape: (9216)
        self.fc1 = nn.Linear(9216, 4096) # (9216) -> (4096)
        self.sigmoid1 = nn.Sigmoid()

        # feature maps.shape: (4096)
        self.fc2 = nn.Linear(4096, 1)
        self.sigmoid2 = nn.Sigmoid()


    def forward(self, x):
        x = self.max_pool1(self.relu1(self.cnn1(x)))

        x = self.max_pool2(self.relu2(self.cnn2(x)))

        x = self.max_pool3(self.relu3(self.cnn3(x)))

        x = self.relu4(self.cnn4(x))

        x = torch.flatten(x, start_dim=1)

        x = self.sigmoid1(self.fc1(x))

        x = self.sigmoid2(self.fc2(x))

        return x


if __name__ == '__main__':
    model = SiameseNet()
    batch_size = 5
    input = torch.rand(size=(batch_size, 1, 105, 105))
    out = model(input)
    print(input.shape, out.shape)
    assert tuple(out.shape) == (batch_size, 1)