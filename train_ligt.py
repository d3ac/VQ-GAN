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
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import lightning as L
import os


class LitModel(L.LightningModule):
    def __init__(self, vqgan, discriminator, lpips, args):
        super().__init__()
        self.vqgan = vqgan
        self.discriminator = discriminator
        self.lpips = lpips
        self.args = args
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        # get optimizer
        optimizer_vqgan, optimizer_discriminator = self.optimizers()

        x, _ = batch
        decoded, indices, q_loss = self.vqgan(x)
        disc_real = self.discriminator(x)      # 真 -> 1
        disc_fake = self.discriminator(decoded)   # 假 -> -1
        mask_disc = self.vqgan.adopt_weight(self.args.disc_factor, self.global_step, threshold=self.args.disc_start)
        
        # calculate vq_loss
        perceptual_loss = self.lpips(x, decoded)
        rec_loss = torch.abs(x - decoded)
        perceptual_reconstruction_loss = (self.args.perceptual_loss_factor * perceptual_loss + self.args.rec_loss_factor * rec_loss).mean()
        gan_loss = - torch.mean(disc_fake) # maximize the probability of D(G(z)) = minimize -D(G(z))
        λ = self.vqgan.calculate_lambda(perceptual_reconstruction_loss, gan_loss)
        vq_loss = perceptual_reconstruction_loss + q_loss + mask_disc * λ * gan_loss # replace the L2 loss used in [63] for Lrec by a perceptual loss (in VQGAN paper)

        # calculate discriminator loss
        discriminator_loss_real = torch.mean(F.relu(1.0 - disc_real))
        discriminator_loss_fake = torch.mean(F.relu(1.0 + disc_fake)) 
        discriminator_loss = mask_disc * 0.5 * (discriminator_loss_real + discriminator_loss_fake)

        # update
        optimizer_vqgan.zero_grad()
        self.manual_backward(vq_loss, retain_graph=True)

        optimizer_discriminator.zero_grad()
        self.manual_backward(discriminator_loss, retain_graph=True)

        optimizer_vqgan.step()
        optimizer_discriminator.step()

        self.log("vq_loss", vq_loss, prog_bar=True)
        self.log("disc_loss", discriminator_loss, prog_bar=True)

    def configure_optimizers(self):
        lr = self.args.learning_rate
        optimizer_vqgan = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant.parameters()),
            lr=lr, eps=1e-8, betas=(self.args.beta1, self.args.beta2)
        )
        optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, eps=1e-8, betas=(self.args.beta1, self.args.beta2))
        return optimizer_vqgan, optimizer_discriminator

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--code_dim', type=int, default=64)
    parser.add_argument('--num_codebook_vectors', type=int, default=1024)
    parser.add_argument('--channels', type=int, nargs='+', default=[128, 128, 256, 256])
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--latent_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--image_channels', type=int, default=3)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--disc_start', type=int, default=3000)
    parser.add_argument('--disc_factor', type=float, default=0.1)
    parser.add_argument('--rec_loss_factor', type=float, default=1)
    parser.add_argument('--perceptual_loss_factor', type=float, default=0.1)
    parser.add_argument('--path', type=str, default='/home/d3ac/Desktop/dataset')
    args = parser.parse_args()

    # Fabric accelerates
    fabric = L.Fabric()
    fabric.launch()

    # dataset
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataloader = DataLoader(datasets.CIFAR10(root=args.path, train=True, download=True, transform=trans), batch_size=args.batch_size, shuffle=False, num_workers=7, pin_memory=True)
    valid_dataloader = DataLoader(datasets.CIFAR10(root=args.path, train=False, download=True, transform=trans), batch_size=args.batch_size, shuffle=False, num_workers=7, pin_memory=True)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    valid_dataloader = fabric.setup_dataloaders(valid_dataloader)

    # model
    model = LitModel(VQGAN(args), Discriminator(args).apply(weights_init), LPIPS().eval(), args)
    trainer = L.Trainer(max_epochs=args.epochs)
    trainer.fit(model, train_dataloader, valid_dataloader)