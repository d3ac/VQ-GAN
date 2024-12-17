import torch
from torch.nn import functional as F
import tqdm
import numpy as np
from torch import nn
import argparse
from vqgan import VQGAN
from discriminator import Discriminator
from lpips import LPIPS
from utils import weights_init, load_data 
from torchvision import utils as vutils
import os

def configure_optimizers(vqgan, discriminator, args):
    lr = args.learning_rate
    optimizer_vqgan = torch.optim.Adam(
        list(vqgan.encoder.parameters()) +
        list(vqgan.decoder.parameters()) +
        list(vqgan.codebook.parameters()) +
        list(vqgan.quant_conv.parameters()) +
        list(vqgan.post_quant.parameters()),
        lr=lr, eps=1e-8, betas=(args.beta1, args.beta2)
    )
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, eps=1e-8, betas=(args.beta1, args.beta2))
    return optimizer_vqgan, optimizer_discriminator

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
    discriminator = Discriminator(args).to(device=args.device)
    discriminator.apply(weights_init)
    LPIPS_model = LPIPS().eval().to(device=args.device)
    optimizer_vqgan, optimizer_discriminator = configure_optimizers(vqgan, discriminator, args)

    # training
    train_dataset, valid_dataset = load_data(args)
    steps_per_epoch = len(train_dataset)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        trange = tqdm.tqdm(range(len(train_dataset)))
        for i, imgs in zip(trange, train_dataset):
            imgs = imgs.to(device=args.device)
            decoded, indices, q_loss = vqgan(imgs)
            disc_real = discriminator(imgs)      # 真 -> 1
            disc_fake = discriminator(decoded)   # 假 -> -1
            mask_disc = vqgan.adopt_weight(args.disc_factor, epoch*steps_per_epoch+i, threshold=args.disc_start)
            
            # calculate vq_loss
            perceptual_loss = LPIPS_model(imgs, decoded)
            rec_loss = torch.abs(imgs - decoded)
            perceptual_reconstruction_loss = (args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss).mean()
            gan_loss = - torch.mean(disc_fake) # maximize the probability of D(G(z)) = minimize -D(G(z))
            λ = vqgan.calculate_lambda(perceptual_reconstruction_loss, gan_loss)
            vq_loss = perceptual_reconstruction_loss + q_loss + mask_disc * λ * gan_loss # replace the L2 loss used in [63] for Lrec by a perceptual loss (in VQGAN paper)

            # calculate discriminator loss
            discriminator_loss_real = torch.mean(F.relu(1.0 - disc_real))
            discriminator_loss_fake = torch.mean(F.relu(1.0 + disc_fake)) # 改一下好一点, 应该乘一个系数0.5
            discriminator_loss = mask_disc * 0.5 * (discriminator_loss_real + discriminator_loss_fake)

            # update
            optimizer_vqgan.zero_grad()
            vq_loss.backward(retain_graph=True)
            optimizer_discriminator.zero_grad()
            discriminator_loss.backward()

            optimizer_vqgan.step()
            optimizer_discriminator.step()
        
        with torch.no_grad():
            #print(decoded_images.min(), decoded_images.max(), 'here')
            real_fake_images = torch.cat((imgs.add(1).mul(0.5)[:4], decoded.add(1).mul(0.5)[:4]))
            vutils.save_image(real_fake_images, "eg.jpg", nrow=4)