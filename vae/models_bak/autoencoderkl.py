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
        h = self.norm1(h)
        h = self.act1(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)

        return h + self.skip(x)


class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=4):
        super().__init__()
        self.res1 = ResnetBlock(in_ch, out_ch, num_groups)
        self.res2 = ResnetBlock(out_ch, out_ch, num_groups)
        # Downsample
        self.down = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.down(x)
        return x


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=4):
        pass

    def forward(self, x, skip):
        pass


# Simple self-attention block
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=4, num_channels=channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(B, C, H*W).permute(0, 2, 1)
        k = self.k(h).reshape(B, C, H*W)
        v = self.v(h).reshape(B, C, H*W).permute(0, 2, 1) # B, HW, C

        attn = torch.bmm(q, k) * (C ** -0.5) # (B, HW, HW)
        attn = torch.softmax(attn, dim=-1)

        h_attn = torch.bmm(attn, v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj(h_attn)


# MidBlock: ResNet → Attention → ResNet
class MidBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.res1 = ResnetBlock(channels, channels, num_groups=4)
        self.attn = AttentionBlock(channels)
        self.res2 = ResnetBlock(channels, channels, num_groups=4)

    def forward(self, x):
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x

class LatentBottleneck(nn.Module):
    def __init__(self, in_channels, latent_channels):
        super().__init__()
        self.conv_mu = nn.Conv2d(in_channels, latent_channels, 1)
        self.conv_logvar = nn.Conv2d(in_channels, latent_channels, 1)

    def forward(self, x):
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, mu, logvar


class Encoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        chan1 = 32
        self.down1 = DownEncoderBlock2D(in_ch=3, out_ch=chan1) # (32, 64, 64)
        self.down2 = DownEncoderBlock2D(in_ch=chan1, out_ch=chan1)

        self.mid1 = MidBlock(channels=chan1)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.mid1(x)
        return x

class AutoencoderKLSmall(nn.Module):
    def __init__(self, in_channels=3, base_channels=(8,16,16,16), latent_channels=4, num_groups=2):
        super().__init__()

    def forward(self, x):
        pass