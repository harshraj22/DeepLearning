import torch
import torch.nn as nn
import logging


class ResidualBlock(nn.Module):
  def __init__(self, kernel_size=3, feature_maps=64, stride=1, in_channels=64):
    super(ResidualBlock, self).__init__()
    self.kernel_size = kernel_size
    self.feature_maps = feature_maps
    self.stride = stride
    self.in_channels = in_channels

    self.conv1 = nn.Conv2d(self.in_channels, self.feature_maps, self.kernel_size, self.stride, padding='same')
    self.bn1 = nn.BatchNorm2d(self.feature_maps)
    self.relu = nn.PReLU()

    self.conv2 = nn.Conv2d(self.in_channels, self.feature_maps, self.kernel_size, self.stride, padding='same')
    self.bn2 = nn.BatchNorm2d(self.feature_maps)
    
  
  def forward(self, x):
    y = self.relu(self.bn1(self.conv1(x)))
    y = self.bn2(self.conv2(y))

    return x + y
  
  
if __name__ == '__main__':
    # To Do: Change use of logging directly to create a logger object and then use that
    logging.basicConfig(format='%(name)s %(levelname)s %(message)s', 
                            level=logging.DEBUG)
    logging.info('Running Test...')
    # (batch_size, num_channels, height_width)
    img = torch.rand(2, 64, 100, 100)
    model = ResidualBlock()
    out = model(img)
    assert tuple(out.shape) == (2, 64, 100, 100)
