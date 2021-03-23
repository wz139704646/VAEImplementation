import os
import torch
import numpy as np
from scipy import stats as st
from torchvision.utils import save_image

from joint_vae import ConvJointVAE
from _test import set_basic_config

import sys
sys.path.append(".")
sys.path.append("..")


def set_test_config():
    """set test configuration"""
    conf = set_basic_config()

    conf['save_dir'] = os.path.join(
        os.path.dirname(__file__),
        'results/latent_traversal')
    conf['cont_traversal_vals'] = st.norm.ppf(np.linspace(0.05, 0.95, 10))
    conf['cont_traversal_indices'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    conf['disc_traversal_indices'] = [0]

    if not os.path.exists(conf['save_dir']):
        os.makedirs(conf['save_dir'])

    return conf


def latent_traversal_cont(model, cont_traversal_vals, save_dir, cont_idx,
                          tag='latent_traversal_cont', base_input=None, num=10):
    """latent traversal for continuous variables in joint VAE
    :param model: joint VAE model object
    :param cont_traversal_vals: the values to traverse (list)
    :param save_dir: the directory to save the results
    :param cont_idx: the index of the continuous variable dimension to traverse (start from 0)
    :param tag: tag string added to the name of the results files
    :param base_input: base input to generate base codes
    :param num: the number of random samples (only work when base_input is None)
    """
    model.eval()

    if model.dim_cont <= 0:
        raise RuntimeError("no continuous latent variables")
    if model.dim_cont <= cont_idx:
        raise RuntimeError("continuous variable dimension index is out of bound")

    device = torch.device("cpu")
    if base_input is not None:
        codes = model.reparameterize(*model.encode(base_input))
    else:
        codes = model.sample_latent(num, device)

    imgs = []
    for i in range(codes.size(0)):
        c = codes[i].clone().detach()
        c_mat = torch.cat(
            [c.clone().detach().unsqueeze(0) for _ in range(len(cont_traversal_vals))])
        # set different values
        c_mat[:, cont_idx] = torch.tensor(cont_traversal_vals)
        dec = model.decode(c_mat)
        if not model.binary:
            dec = model.sample_normal(
                torch.flatten(dec[0], start_dim=1),
                torch.flatten(dec[1], start_dim=1)
            ).view(-1, *model.input_size)

        imgs.append(dec)

    save_image(torch.cat(imgs).cpu(),
               os.path.join(save_dir, "{}_contIdx{}.png".format(tag, cont_idx)),
               nrow=len(cont_traversal_vals))


def latent_traversal_disc(model, save_dir, disc_idx, tag='latent_traversal_disc',
                          base_input=None, num=10):
    """latent traversal for discrete variables in joint VAE
    :param model: the joint VAE model object
    :param save_dir: the directory to save the results
    :param disc_idx: the index of the discrete variable
    :param tag: tag string added to the name of the results files
    :param base_input: base input to generate base codes
    :param num: the number of random samples (only work when base_input is None)
    """
    model.eval()

    if model.dim_disc <= 0:
        raise RuntimeError("no discrete latent variables")
    if len(model.latent_disc) <= disc_idx:
        raise RuntimeError("discrete variable index is out of bound")

    device = torch.device("cpu")
    if base_input is not None:
        codes = model.reparameterize(*model.encode(base_input))
    else:
        codes = model.sample_latent(num, device)

    dim_disc = model.latent_disc[disc_idx]
    pre_dims = model.dim_cont + sum(model.latent_disc[:disc_idx])
    # initialize codes, set discrete variable to 0
    codes[:, pre_dims:(pre_dims + dim_disc)] = 0.

    imgs = []
    for i in range(codes.size(0)):
        c = codes[i].clone().detach()
        c_mat = torch.cat([c.clone().detach().unsqueeze(0) for _ in range(dim_disc)])
        # set different values
        c_mat[np.arange(dim_disc), pre_dims + np.arange(dim_disc)] = 1.
        dec = model.decode(c_mat)
        if not model.binary:
            dec = model.sample_normal(
                torch.flatten(dec[0], start_dim=1),
                torch.flatten(dec[1], start_dim=1)
            ).view(-1, *model.input_size)

        imgs.append(dec)

    save_image(torch.cat(imgs).cpu(),
               os.path.join(save_dir, "{}_discIdx{}.png".format(tag, disc_idx)),
               nrow=dim_disc)


if __name__ == '__main__':
    test_conf = set_test_config()
    checkpoint_path = test_conf['checkpoint_path']
    save_dir = test_conf['save_dir']

    model_checkpoint = torch.load(checkpoint_path)
    train_args = model_checkpoint['train_args']
    train_conf = model_checkpoint['train_config']
    model_state_dict = model_checkpoint['state_dict']

    img_size = train_conf['image_size']
    model = ConvJointVAE(
        train_args.temperature, train_args.disc_dims,
        train_args.disc_cap, train_args.cont_cap,
        img_size, train_args.channels, train_args.hidden_size,
        train_args.dim_z)
    model.load_state_dict(model_state_dict)
    model.cont_cap_current = model_checkpoint["extra"]["cont_cap"]
    model.disc_cap_current = model_checkpoint["extra"]["disc_cap"]

    # traversal
    cont_indices = test_conf["cont_traversal_indices"]
    cont_vals = test_conf["cont_traversal_vals"]
    disc_indices = test_conf["disc_traversal_indices"]

    # traverse continuous variables
    for c_idx in cont_indices:
        latent_traversal_cont(model, cont_vals, save_dir, c_idx)

    # traverse discrete variables
    for d_idx in disc_indices:
        latent_traversal_disc(model, save_dir, d_idx)

