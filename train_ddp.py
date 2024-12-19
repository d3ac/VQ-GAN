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
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
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

def main(rank: int, args):
    dist.init_process_group(backend='nccl', world_size=args.world_size, rank=rank)
    print(f"=> init GPU {rank}.")
    torch.cuda.set_device(rank)
    batch_size = args.batch_size // args.world_size

    model = VQGAN(args).cuda(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    discriminator = Discriminator(args).cuda(rank)
    discriminator.apply(weights_init)
    discriminator = nn.parallel.DistributedDataParallel(discriminator, device_ids=[rank])

    LPIPS_model = LPIPS().eval().cuda(rank)

    # criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer_vqgan, optimizer_discriminator = configure_optimizers(model, discriminator, args)
    
    cudnn.benchmark = True

    # dataset
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.CIFAR10(root=args.path, train=True, download=True, transform=trans)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=7, pin_memory=True, sampler=train_sampler)
    
    test_dataset = datasets.CIFAR10(root=args.path, train=False, download=True, transform=trans)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=7, pin_memory=True, sampler=test_sampler)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        


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
    parser.add_argument('--batch_size', type=int, default=256)
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
    
    args.world_size = torch.cuda.device_count()

    mp.spawn(main, args=args, nprocs=args.world_size)

    # dataset
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataloader = DataLoader(datasets.CIFAR10(root=args.path, train=True, download=True, transform=trans), batch_size=args.batch_size, shuffle=True, num_workers=7, pin_memory=True)
    test_dataloader = DataLoader(datasets.CIFAR10(root=args.path, train=False, download=True, transform=trans), batch_size=args.batch_size, shuffle=True, num_workers=7, pin_memory=True)
    
    # model
    vqgan = VQGAN(args).to(device=args.device)
    discriminator = Discriminator(args).to(device=args.device)
    discriminator.apply(weights_init)
    LPIPS_model = LPIPS().eval().to(device=args.device)
    optimizer_vqgan, optimizer_discriminator = configure_optimizers(vqgan, discriminator, args)

    # training
    steps_per_epoch = len(train_dataloader)
    best_loss = float('inf')
    
    # learning decay
    Milestones = [20, 50, 80]
    lr_decay_vqgan = torch.optim.lr_scheduler.MultiStepLR(optimizer_vqgan, milestones=Milestones, gamma=0.1)
    lr_decay_discriminator = torch.optim.lr_scheduler.MultiStepLR(optimizer_discriminator, milestones=Milestones, gamma=0.1)

    trange = tqdm.trange(args.epochs)
    for epoch in trange:
        trange2 = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for i, (imgs, labels) in enumerate(trange2):
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
            trange2.set_postfix(loss=vq_loss.cpu().detach().numpy().item())
        
        with torch.no_grad():
            #print(decoded_images.min(), decoded_images.max(), 'here')
            real_fake_images = torch.cat((imgs.add(1).mul(0.5)[:4], decoded.add(1).mul(0.5)[:4]))
            vutils.save_image(real_fake_images, "eg.jpg", nrow=4)
        
        lr_decay_vqgan.step()
        lr_decay_discriminator.step()
        
        # evaluate
        losses = []
        for imgs, labels in tqdm.tqdm(test_dataloader, desc="Evaluating", leave=False):
            imgs = imgs.to(device=args.device)
            decoded, indices, q_loss = vqgan(imgs)
            disc_fake = discriminator(decoded)
            mask_disc = vqgan.adopt_weight(args.disc_factor, epoch*steps_per_epoch+i, threshold=args.disc_start)
            perceptual_loss = LPIPS_model(imgs, decoded)
            rec_loss = torch.abs(imgs - decoded)
            perceptual_reconstruction_loss = (args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss).mean()
            gan_loss = - torch.mean(disc_fake)
            λ = vqgan.calculate_lambda(perceptual_reconstruction_loss, gan_loss)
            vq_loss = perceptual_reconstruction_loss + q_loss + mask_disc * λ * gan_loss
            losses.append(vq_loss.cpu().detach().numpy().item())
        mean_loss = np.mean(losses)
        trange.set_postfix(loss=mean_loss)

        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(vqgan.state_dict(), 'vqgan.pth')