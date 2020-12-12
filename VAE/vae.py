# Implementation of Variational Auto-Encoder (paper at https://arxiv.org/abs/1312.6114) based on pytorch

import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    """Class that implements of Variational Auto-Encoder"""

    def __init__(self, n_input, n_hidden_en, n_hidden_de, dim_z, n_output):
        """initialize neural networks"""
        super(VAE, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.dim_z = dim_z

        # encoder layers
        self.fc1 = nn.Linear(n_input, n_hidden_en) # the first layer
        self.fc2_mean = nn.Linear(n_hidden_en, dim_z) # the second layer to compute mu
        self.fc2_sd = nn.Linear(n_hidden_en, dim_z) # the second layer to compute sigma

        # decoder layers
        self.fc3 = nn.Linear(dim_z, n_hidden_de) # the third layer
        self.fc4 = nn.Linear(n_hidden_de, n_output) # the fourth layer

    def encoder(self, x):
        """Gaussian MLP encoder"""
        h1 = F.tanh(self.fc1(x))

        return self.fc2_mean(h1), self.fc2_sd(h1)

    def reparameterize(self, mu, logvar):
        """reparameterization trick"""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) # noise epsilon

        return mu + eps * std

    def decoder(self, z):
        """Bernoulli MLP decoder"""
        h3 = F.tanh(self.fc3(z))
        
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        """autoencoder forward computation"""
        mu, logvar = self.encoder(x.view(-1, self.n_input))
        z = self.reparameterize(mu, logvar) # latent variable z

        return self.decoder(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    """loss function described in the paper (eq. (10))"""
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') # likelyhood term
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # KL divergence term

    return BCE + KLD

