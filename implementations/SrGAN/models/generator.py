import torch
import torch.nn as nn
from .residual_block import ResidualBlock
from .upsampling_block import UpsamplingBlock

class Generator(nn.Module):
  def __init__(self, in_channels=3, num_res_block=5):
    super(Generator, self).__init__()
    self.in_channels = in_channels
    self.num_res_block = num_res_block

    self.conv1 = nn.Conv2d(self.in_channels, 64, 9, 1, padding='same')
    self.prelu = nn.PReLU()

    self.res_block = nn.ModuleList([
        ResidualBlock() for _ in range(self.num_res_block)
    ])

    self.conv2 = nn.Conv2d(64, 64, 3, 1, padding='same')
    self.batch_norm = nn.BatchNorm2d(64)

    self.up_sample1 = UpsamplingBlock(64) # outputs 64 channels # 64 = C * (2*2), gets converted to C channels
    self.up_sample2 = UpsamplingBlock(64)

    self.conv3 = nn.Conv2d(64, 3, 9, 1, padding='same')


  def forward(self, x):
    x = self.prelu(self.conv1(x))
    y = x
    for layer in self.res_block:
      y = layer(y)
    y = self.batch_norm(self.conv2(y))

    x = y + x
    x = self.up_sample1(x)
    x = self.up_sample2(x)
    x = self.conv3(x)

    return x
