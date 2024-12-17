import torch
from torch import nn
import torch.nn.functional as F
from utils import ResBlock, NonLocalBlock, DownSampleBlock, Swish
from encoder import Encoder
from decoder import Decoder
from codebook import Codebook


class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(args).to(device=args.device)
        self.decoder = Decoder(args).to(device=args.device)
        self.codebook = Codebook(args).to(device=args.device)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device) # 1x1卷积参数只有两个参数w和b
        self.post_quant = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)

    def forward(self, imgs):
        encoded = self.encoder(imgs)
        quantized = self.quant_conv(encoded)
        zq, indices, loss = self.codebook(quantized)
        post_quant = self.post_quant(zq)
        decoded = self.decoder(post_quant)
        return decoded, indices, loss

    def encode(self, imgs):
        encoded = self.encoder(imgs)
        quantized = self.quant_conv(encoded)
        zq, indices, loss = self.codebook(quantized)
        return zq, indices, loss
    
    def decode(self, z):
        pass

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]
        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ
    
    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0):
        """
        在训练过程中, 前期不计入discriminator的loss, 让生成器能够更好的学习到数据的分布，避免判别器过早的产生太强的对抗压力，使得训练不稳定
        """
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))