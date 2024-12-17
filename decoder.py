import torch
from torch import nn
import torch.nn.functional as F
from utils import ResBlock, NonLocalBlock, DownSampleBlock, Swish, UpSampleBlock


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        channels = args.channels[::-1]
        attn_resolutions = [16]
        num_res_blocks = 2
        resolution = args.latent_size

        in_channels = channels[0]
        layers = [
            nn.Conv2d(args.latent_dim, in_channels, 3, 1, 1),
            ResBlock(in_channels, in_channels),
            NonLocalBlock(in_channels),
            ResBlock(in_channels, in_channels)
        ]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if resolution < args.resolution:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2
        layers.append(nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, args.image_channels, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)