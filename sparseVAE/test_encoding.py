import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from sparse_vae import SparseVAE
from _test import set_basic_config

import sys
sys.path.append(".")
sys.path.append("..")
from utils import save_
from load_data import prepare_data_mnist


def set_test_config():
    """set test configuration"""
    conf = set_basic_config()

    conf['checkpoint_path'] = os.path.join(
        os.path.dirname(__file__),
        'checkpoints/mnist/t1611322929-epoch20_z200_alpha0.01.pth.tar')
    conf['save_dir'] = os.path.join(
        os.path.dirname(__file__),
        'results/encoding')
    
    if not os.path.exists(conf['save_dir']):
        os.makedirs(conf['save_dir'])

    return conf


def plot_image(img, ax):
    ax.imshow(np.transpose(img, (1,2,0)), interpolation='nearest')


def plot_encoding(image, img_size, model, filename, negative=True, width=1/7):
    """plot a single image and its encoding"""
    flatten_img = torch.flatten(image).unsqueeze(0)
    
    encoded = model.encode(flatten_img)
    z = model.reparameterize(*encoded) # codes
    
    _, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(21, 7))
    
    # plot input image
    img = torchvision.utils.make_grid(image).detach().numpy()
    if negative:
        # exchange black and white
        img = 1 - img
    plot_image(img, ax0)
    ax0.set_title('input image', fontsize=20)

    ax1.bar(np.arange(z.shape[1]), height=z.cpu().detach().numpy()[0],
            width=width, align='center')
    ax1.scatter(np.arange(z.shape[1]), z.cpu().detach().numpy()[0],
                color='blue')
    ax1.set_title(r"latent dimension %d - $\alpha$ = %.2f" % \
                  (z.shape[1], model.alpha), fontsize=20)

    # plot reconstructed image
    img = model.reconstruct(flatten_img)
    img = torchvision.utils.make_grid(img.view(1, *img_size)).cpu().detach().numpy()
    if negative:
        # exchange black and white
        img = 1 - img
    plot_image(img, ax2)
    ax2.set_title('reconstruction', fontsize=20)
    
    plt.subplots_adjust(hspace=0.5)

    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    test_conf = set_test_config()
    checkpoint_path = test_conf['checkpoint_path']
    save_dir = test_conf['save_dir']
    
    model_checkpoint = torch.load(checkpoint_path)
    train_args = model_checkpoint['train_args']
    train_conf = model_checkpoint['train_config']
    model_state_dict = model_checkpoint['state_dict']

    # mnist dataset
    test_loader = prepare_data_mnist(train_args.batch_size, train_conf['data_dir'],
                                     train=False, shuffle=False)
    images, _ = iter(test_loader).next()

    # load model and config
    img_size = train_conf['image_size']
    model = SparseVAE(img_size[0]*img_size[1], train_args.hidden_sizes,
                      train_args.dim_z, train_args.alpha)
    model.load_state_dict(model_state_dict)
    img_size = (1, *img_size) # C x W x H

    # plot encoding
    plot_encoding(images[0], img_size, model, "test1.png")
    
    plot_encoding(images[1], img_size, model, "test2.png")

    plot_encoding(images[2], img_size, model, "test3.png")
