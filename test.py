import torch
from torch.nn import functional as F
import argparse
from vqgan import VQGAN
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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
    parser.add_argument('--batch_size', type=int, default=8)
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
    vqgan.load_state_dict(torch.load('vqgan.pth', weights_only=True))

    for test_imgs, _ in test_dataloader:
        test_imgs = test_imgs.to(device=args.device)
        vqgan.eval()
        with torch.no_grad():
            test_decoded, indices, loss = vqgan(test_imgs)
        break

    for train_imgs, _ in train_dataloader:
        train_imgs = train_imgs.to(device=args.device)
        vqgan.eval()
        with torch.no_grad():
            train_decoded, indices, loss = vqgan(train_imgs)
        break

    fig, axes = plt.subplots(4, args.batch_size, figsize=(args.batch_size * 4, 4 * 4))
    for j in range(args.batch_size):
        axes[0, j].imshow(test_decoded[j].permute(1, 2, 0).cpu().numpy(), cmap='gray')
        axes[0, j].axis('off')

        axes[1, j].imshow(test_imgs[j].permute(1, 2, 0).cpu().numpy(), cmap='gray')
        axes[1, j].axis('off')

        axes[2, j].imshow(train_decoded[j].permute(1, 2, 0).cpu().numpy(), cmap='gray')
        axes[2, j].axis('off')
        
        axes[3, j].imshow(train_imgs[j].permute(1, 2, 0).cpu().numpy(), cmap='gray')
        axes[3, j].axis('off')
    
    axes[0, 0].set_title('Test Decoded', fontsize=20)
    axes[1, 0].set_title('Test Original', fontsize=20)
    axes[2, 0].set_title('Train Decoded', fontsize=20)
    axes[3, 0].set_title('Train Original', fontsize=20)
    plt.show()
    
