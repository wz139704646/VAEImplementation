import os
import torch
from torchvision.utils import save_image

from beta_tcvae import BetaTCVAE
from _test import set_basic_config

import sys
sys.path.append(".")
sys.path.append("..")
from utils import latent_traversal_detailed


def set_test_config():
    """set test configuration"""
    conf = set_basic_config()

    conf['save_dir'] = os.path.join(
        os.path.dirname(__file__),
        'results/latent_traversal')
    conf['traversal_vals'] = torch.arange(-3, 3.1, 6./15).tolist()

    if not os.path.exists(conf['save_dir']):
        os.makedirs(conf['save_dir'])

    return conf


if __name__ == '__main__':
    test_conf = set_test_config()
    checkpoint_path = test_conf['checkpoint_path']
    save_dir = test_conf['save_dir']
    traversal_vals = torch.tensor(test_conf['traversal_vals'])

    model_checkpoint = torch.load(checkpoint_path)
    train_args = model_checkpoint['train_args']
    train_conf = model_checkpoint['train_config']
    model_state_dict = model_checkpoint['state_dict']

    img_size = train_conf['image_size']
    model = BetaTCVAE(img_size[0]*img_size[1], train_args.n_hidden,
                    train_args.dim_z, img_size[0]*img_size[1],
                    train_args.alpha, train_args.beta, train_args.gamma)
    model.load_state_dict(model_state_dict)

    # traversal
    img_size = (1, *img_size)
    latent_traversal_detailed(model, traversal_vals, img_size, save_dir)
