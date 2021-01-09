# Implementation of beta-VAE (paper at https://openreview.net/forum?id=Sy2fzU9gl) based on pytorch

import torch
from torch.nn import functional as F
from torch.distributions.normal import Normal

import sys
sys.path.append(".")
sys.path.append("..")
from VAE import VAE


class BetaVAE(VAE):
    """Class that implements beta Variational Auto-Encoder"""

    def __init__(self, n_input, n_hidden, dim_z, n_output, beta, binary=True, **kwargs):
        """initialize neural networks
        :param beta: coefficient applied in loss function for KL term
        """
        super(BetaVAE, self).__init__(n_input, n_hidden, dim_z, n_output, binary, **kwargs)
        self.beta = beta # additional coef compared to original VAE

    def loss_function(self, *inputs, **kwargs):
        """loss function described in the paper (eq. (4))"""
        decoded = inputs[0]
        x = inputs[1]
        encoded = inputs[2]

        mu, logvar = encoded
        KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean() # KL divergence term
        if self.binary:
            # likelyhood term under Bernolli MLP decoder
            MLD = F.binary_cross_entropy(decoded, x, reduction='sum').div(x.size(0))
        else:
            # otherwise, return loss under Gaussian MLP decoder
            mu_o, logvar_o = decoded
            recon_x_distribution = Normal(loc=mu_o, scale=torch.exp(0.5*logvar_o))
            MLD = -recon_x_distribution.log_prob(x).sum(1).mean()
        
        return MLD + self.beta * KLD

