import torch
import sys
import lpips
import wandb
import os
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms as T

from typing import Callable
from omegaconf import OmegaConf
from PIL import Image


# ----------------------------------
# Basic building blocks
# ----------------------------------

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=1e-6, affine=True)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=1e-6, affine=True)
        self.act2 = nn.SiLU()

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()


    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(x)
        h = self.act1(x)

        h = self.conv2(x)
        h = self.norm2(x)
        h = self.act2(x)

        return h + self.skip(x)


class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=4):
        self.res1 = ResnetBlock(in_ch, out_ch, num_groups)
        self.res2 = ResnetBlock(out_ch, out_ch, num_groups)
        # Downsample
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.down(x)
        return x_down, x


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=4):
        pass

    def forward(self, x, skip):
        pass


# Simple self-attention block
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        pass

    def forward(self, x):
        pass

# MidBlock: ResNet → Attention → ResNet
class MidBlock(nn.Module):
    def __init__(self, channels):
        pass

    def forward(self, x):
        pass


class LatentBottleneck(nn.Module):
    def __init__(self, in_channels, latent_channels):
        pass

    def forward(self, x):
        pass


class Encoder(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


# 1) Re-use your building blocks:
#    - ResnetBlock (post-activation)
#    - DownEncoderBlock2D
#    - UpDecoderBlock2D
#    - AttentionBlock
#    - MidBlock
#    - LatentBottleneck

# (Assume you’ve defined them exactly as tested before, 
#  just reducing GroupNorm groups for this small test.)

class AutoencoderKLSmall(nn.Module):
    def __init__(self, in_channels=3, base_channels=(8,16,16,16), latent_channels=4, num_groups=2):
        pass

    def forward(self, x):
        pass