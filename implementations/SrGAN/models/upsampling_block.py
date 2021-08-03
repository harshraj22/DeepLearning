import torch
import torch.nn as nn

class UpsamplingBlock(nn.Module):
  def __init__(self, in_channels=64):
    super(UpsamplingBlock, self).__init__()
    self.in_channels = in_channels

    self.conv = nn.Conv2d(self.in_channels, 256, 3, 1, padding='same')
    self.pixel_shuffle = nn.PixelShuffle(2)
    self.relu = nn.PReLU()

  def forward(self, x):
    return self.relu(self.pixel_shuffle(self.conv(x)))
