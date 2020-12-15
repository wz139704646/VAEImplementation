# Implementation of Variational Auto-Encoder (paper at https://arxiv.org/abs/1312.6114) based on pytorch

import sys
sys.path.append("..")

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from base import BaseVAE


class VAE(BaseVAE):
    """Class that implements of Variational Auto-Encoder"""

    def __init__(self, n_input, n_hidden, dim_z, n_output, binary=True, **kwargs):
        """initialize neural networks
        :param binary: whether the input is binary data (determine which decoder to use)
        """
        super(VAE, self).__init__()
        self.dim_z = dim_z
        self.binary = binary

        # encoder layers, use Gaussian MLP as encoder
        self.fc1 = nn.Linear(n_input, n_hidden) # the first layer
        self.fc2_mean = nn.Linear(n_hidden, dim_z) # the second layer to compute mu
        self.fc2_var = nn.Linear(n_hidden, dim_z) # the second layer to compute sigma

        # decoder layers
        self.fc3 = nn.Linear(dim_z, n_hidden) # the third layer
        if binary:
            # binary input data, use Bernoulli MLP as decoder
            self.fc4 = nn.Linear(n_hidden, n_output) # the fourth layer
        else:
            # not binary data, use Gaussian MLP as decoder
            self.fc4_mean = nn.Linear(n_hidden, n_output)
            self.fc4_var = nn.Linear(n_hidden, n_output)

    def encode(self, x):
        """Gaussian MLP encoder"""
        h1 = F.tanh(self.fc1(x))

        return self.fc2_mean(h1), self.fc2_var(h1)

    def reparameterize(self, mu, logvar):
        """reparameterization trick"""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) # noise epsilon

        return mu + eps * std

    def decode(self, code):
        """Guassian/Bernoulli MLP decoder"""
        h3 = F.tanh(self.fc3(code))

        if self.binary:
            # binary input data, use Bernoulli MLP decoder
            return torch.sigmoid(self.fc4(h3))
        
        # otherwise use Gaussian MLP decoder
        return self.fc4_mean(h3), self.fc4_var(h3)

    def sample(self, num, device, **kwargs):
        """sample from latent space and return the decoded output"""
        z = torch.randn(num, self.dim_z).to(device)
        samples = self.decode(z)

        if not self.binary:
            # Gaussian
            mu_o, logvar_o = samples
            samples = self.reparameterize(mu_o, logvar_o)

        # otherwise Bernoulli
        return samples

    def forward(self, input):
        """autoencoder forward computation"""
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar) # latent variable z

        return self.decode(z), mu, logvar

    def reconstruct(self, input, **kwargs):
        """reconstruct from the input"""
        recon = self.forward(input)[0]
        
        if not self.binary:
            # use Guassian, can not directly return recon
            mu_o, logvar_o = recon
            recon = self.reparameterize(mu_o, logvar_o)

        # use Bernoulli, can directly return as the reconstructed result
        return recon

    def loss_function(self, *inputs, **kwargs):
        """loss function described in the paper (eq. (10))"""
        recon_x = inputs[0]
        x = inputs[1]
        mu = inputs[2]
        logvar = inputs[3]

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # KL divergence term
        if self.binary:
            # loss under Bernolli MLP decoder
            BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') # likelyhood term
            return BCE + KLD

        # otherwise, return loss under Gaussian MLP decoder
        mu_o, logvar_o = recon_x
        MLD = torch.norm((((2*np.pi*logvar_o.exp()).sqrt())*(((x-mu_o).pow(2))/(2*logvar_o.exp()))),dim=1)
        return MLD + KLD



