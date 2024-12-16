import torch
from torch import nn

class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()