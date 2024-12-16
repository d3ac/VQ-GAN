import torch
import tqdm
import numpy as np
from torch import nn
import argparse


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--code_dim', type=int, default=64)
    parser.add_argument('--num_codebook_vectors', type=int, default=1024)
    parser.add_argument('--channels', type=int, nargs='+', default=[128, 128, 256, 256])
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--latent_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--disc_start', type=int, default=1000)
    parser.add_argument('--disc_factor', type=float, default=0.1)
    parser.add_argument('--rec_loss_factor', type=float, default=1)
    parser.add_argument('--perceptual_loss_factor', type=float, default=0.1)
    parser.add_argument('--dataset_path', type=str, default='/home/d3ac/Desktop/dataset/adroit')
    args = parser.parse_args()

    # dataset 
    pass

    # model
    vqgan = VQGAN(args).to(device=args.device)