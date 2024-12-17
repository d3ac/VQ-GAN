import torch
from torch import nn
import torch.nn.functional as F

class Codebook(nn.Module):
    def __init__(self, args):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.beta = args.beta
        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1/self.num_codebook_vectors, 1/self.num_codebook_vectors) # 这样初始化，不然太大了全部加起来直接炸了

    def forward(self, z):
        """
        公式图片 : https://cdn.mathpix.com/snip/images/0-890j6AncvDWQRfJwcg3vLt07dCZHbR3X8YYCq7JuQ.original.fullsize.png
        """
        z = z.permute(0, 2, 3, 1).contiguous() # 把channel放到最后，在这里也就是隐藏层latent_dim
        z_flat = z.view(-1, self.latent_dim)
        dist = torch.sum(z_flat**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        indices = torch.argmin(dist, dim=1)
        z_q = self.embedding(indices).view(z.shape)
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2) 
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2)
        return z_q, indices, loss