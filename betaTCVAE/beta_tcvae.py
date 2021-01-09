# Implementation of beta-TCVAE (paper at https://arxiv.org/abs/1802.04942) based on pytorch

import math
import torch
from torch.nn import functional as F
from torch.distributions.normal import Normal

import sys
sys.path.append(".")
sys.path.append("..")
from betaVAE import BetaVAE


class BetaTCVAE(BetaVAE):
    """Class that implements beta Total Correlation Variational Auto-Encoder"""

    def __init__(self, n_input, n_hidden, dim_z, n_output,
                 alpha, beta, gamma, binary=True, sampling="mws", **kwargs):
        """initialize neural networks
        :param alpha: weight for Index-Code MI term in loss function
        :param beta: weight for Total Correlation term in loss function
        :param gamma: weight for Dimension-wise KL term in loss function
        """
        super(BetaTCVAE, self).__init__(n_input, n_hidden,
                                        dim_z, n_output, beta, binary, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.sampling = sampling.lower()

    def forward(self, input):
        """autoencoder forward computation"""
        encoded = self.encode(input)
        mu, logvar = encoded
        z = self.reparameterize(mu, logvar) # latent variable z

        return self.decode(z), encoded, z

    def get_log_importance_weight_mat(self, batch_size, dataset_size):
        """calculate the importance weight matrix"""
        # N denotes dataset_size, M denotes batch_size-1
        strat_weight = (dataset_size - batch_size + 1) / \
            (dataset_size * (batch_size - 1))  # (N-M)/(NM)
        # initialize weight matrix with 1/M
        importance_weights = torch.Tensor(
            batch_size, batch_size).fill_(1 / (batch_size-1))
        # set diagonal weight to be 1/N (weight for q(z|n*))
        importance_weights.view(-1)[::batch_size+1] = 1 / dataset_size
        # select one element each row to set it to be (N-M)/(NM) (weight for q(z|n_M))
        importance_weights.view(-1)[1::batch_size+1] = strat_weight
        importance_weights[-1, 0] = strat_weight

        return importance_weights.log()

    def loss_function(self, *inputs, **kwargs):
        """loss function described in the paper (eq. (4))"""
        decoded = inputs[0]
        x = inputs[1]
        encoded = inputs[2]
        z = inputs[3]
        dataset_size = kwargs['dataset_size']
        batch_size = z.size(0)

        mu, logvar = encoded
        # compute likelyhood term
        if self.binary:
            # likelihood term under Bernolli MLP decoder
            MLD = F.binary_cross_entropy(decoded, x, reduction='sum').div(x.size(0))
        else:
            # likelihood term under Gaussian MLP decoder
            mu_o, logvar_o = decoded
            recon_x_distribution = Normal(
                loc=mu_o, scale=torch.exp(0.5*logvar_o))
            MLD = -recon_x_distribution.log_prob(x).sum(1).mean()

        log_q_z_n = Normal(loc=mu, scale=torch.exp(
            0.5*logvar)).log_prob(z).sum(1)  # log q(z|n)
        log_p_z = Normal(loc=torch.zeros_like(
            z), scale=torch.ones_like(z)).log_prob(z).sum(1)  # p(z) (N(0,I))

        # the log(q(z(n_i))|n_j) matrix
        mat_log_q_z_n = Normal(loc=mu.unsqueeze(dim=0), scale=torch.exp(
            0.5*logvar).unsqueeze(dim=0)).log_prob(z.unsqueeze(dim=1))

        # compute log q(z) and log prod_j q(z_j) according to sampling method
        if self.sampling == "mws":
            # MWS(Minibatch Weighted Sampling)
            log_q_z = torch.logsumexp(
                mat_log_q_z_n.sum(2), dim=1, keepdim=False) - math.log(batch_size*dataset_size)
            log_prod_q_z = (torch.logsumexp(
                mat_log_q_z_n, dim=1, keepdim=False) - math.log(batch_size*dataset_size)).sum(1)
        elif self.sampling == "mss":
            # MSS(Minibatch Stratified Sampling)
            log_importance_weights = self.get_log_importance_weight_mat(
                batch_size, dataset_size).type_as(mat_log_q_z_n)

            log_q_z = torch.logsumexp(
                log_importance_weights + mat_log_q_z_n.sum(2), dim=1, keepdim=False)
            log_prod_q_z = torch.logsumexp(
                log_importance_weights.unsqueeze(dim=2) + mat_log_q_z_n, dim=1, keepdim=False).sum(1)
        else:
            raise NotImplementedError

        # decomposition
        index_code_MI = (log_q_z_n - log_q_z).mean()
        TC = (log_q_z - log_prod_q_z).mean()
        dim_wise_KL = (log_prod_q_z - log_p_z).mean()
        # print("MI: {}, TC: {}, KL: {}".format(index_code_MI, TC, dim_wise_KL))

        return MLD + self.alpha * index_code_MI + self.beta * TC + self.gamma * dim_wise_KL
