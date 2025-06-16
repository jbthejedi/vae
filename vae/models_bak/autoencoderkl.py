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
    def __init__(self, in_ch, out_ch, num_groups=32):
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


# Simple self-attention block
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
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
        self.res1 = ResnetBlock(channels, channels, num_groups=32)
        self.attn = AttentionBlock(channels)
        self.res2 = ResnetBlock(channels, channels, num_groups=32)

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


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=32):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.res1 = ResnetBlock(in_ch, out_ch, num_groups)
        self.res2 = ResnetBlock(out_ch, out_ch, num_groups)

    def forward(self, x):
        x = self.up(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        base_channels=(128,256,512,512),
        latent_channels=4
    ):
        super().__init__()
        self.downblocks = nn.ModuleList([
            DownEncoderBlock2D(in_ch=in_channels, out_ch=base_channels[0]), # (32, 64, 64)
            DownEncoderBlock2D(in_ch=base_channels[0], out_ch=base_channels[1]),
            DownEncoderBlock2D(in_ch=base_channels[1], out_ch=base_channels[2]),
            DownEncoderBlock2D(in_ch=base_channels[2], out_ch=base_channels[3]),
        ])
        self.mid = MidBlock(channels=base_channels[3])
        self.lb = LatentBottleneck(
            in_channels=base_channels[3],
            latent_channels=latent_channels,
        )

    def forward(self, x):
        for block in self.downblocks:
            x = block(x)
        x = self.mid(x)
        z, mu, logvar = self.lb(x)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, in_ch, base_channels=(128,256,512,512), out_ch=3):
        super().__init__()
        self.upblock = nn.ModuleList([
            UpDecoderBlock2D(in_ch=in_ch, out_ch=base_channels[3]),
            UpDecoderBlock2D(in_ch=base_channels[3], out_ch=base_channels[2]),
            UpDecoderBlock2D(in_ch=base_channels[2], out_ch=base_channels[1]),
            UpDecoderBlock2D(in_ch=base_channels[1], out_ch=base_channels[0]),
        ])
        self.conv_out = nn.Conv2d(base_channels[0], out_ch, kernel_size=3, padding=1)
    
    def forward(self, z):
        x = z
        for block in self.upblock:
            x = block(x)
        return self.conv_out(x)


class AutoencoderKLSmall(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_channels=(128,256,512,512),
        latent_channels=4,
        num_groups=32
    ):
        super().__init__()
        self.enc = Encoder(in_channels, base_channels, latent_channels)
        self.dec = Decoder(in_ch=latent_channels, base_channels=base_channels, out_ch=3)

    def forward(self, x):
        z, mu, logvar = self.enc(x)
        x_hat = self.dec(z)
        x_hat = torch.sigmoid(x_hat)
        return x_hat, mu, logvar