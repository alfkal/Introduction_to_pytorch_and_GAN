import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, z_dim=100, output_dim=28*28):
        super(Generator, self).__init__()
        # TODO: Complete the model architecture


        # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            self._modules[m].weight.data.normal_(mean, std)
            self._modules[m].bias.data.zero_()

    def forward(self, x):
        # TODO: Complete the forward pass
        return x
