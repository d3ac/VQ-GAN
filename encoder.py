import torch
from torch import nn
import torch.nn.functional as F
from utils import ResBlock, NonLocalBlock, DownSampleBlock, Swish


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        channels = args.channels
        attn_resolutions = [16]
        latent_dim = args.latent_dim
        resolution = args.resolution
        num_res_blocks = 2
        layers = [nn.Conv2d(args.image_channels, channels[0], 3, 1, 1)]
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if resolution > latent_dim:
                layers.append(DownSampleBlock(channels[i+1]))
                resolution //= 2
        layers.append(ResBlock(channels[-1], channels[-1]))
        layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResBlock(channels[-1], channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], args.latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)