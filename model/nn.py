import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class GeneNet(nn.Module):
  def __init__(self, **params):
    super(GeneNet, self).__init__()

  # def forward(self, x):
