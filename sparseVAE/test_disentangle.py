import os
import time
import shutil
import imageio
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from sparse_vae import SparseVAE

import sys
sys.path.append(".")
sys.path.append("..")
from load_data import prepare_data_mnist


def set_test_config():
    """set test configuration"""
    conf = {}
    conf['checkpoint_path'] = os.path.join(
        os.path.dirname(__file__),
        'checkpoints/mnist/t1611322929-epoch20_z200_alpha0.01.pth.tar')
    conf['save_dir'] = os.path.join(
        os.path.dirname(__file__),
        'results/latent_traversal')
    conf['traversal_vals'] = torch.arange(-3, 3.1, 6./15).tolist()

    if not os.path.exists(conf['save_dir']):
        os.makedirs(conf['save_dir'])

    return conf


def interpretability_traversal(model, traversal_vals, img_size, filename,
                               base_input, negative=True, width=1/2):
    flatten_img = torch.flatten(base_input).unsqueeze(0)

    encoded = model.encode(flatten_img)
    z = model.reparameterize(*encoded) # codes

    max_ind = torch.argmax(z).item()
    maxz = z[0, max_ind]

    # create temp directory
    tmpdir = os.path.join(os.path.dirname(filename), './tmp_{}'.format(int(time.time())))
    os.makedirs(tmpdir)

    images = []
    for v in traversal_vals:
        _, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        z1 = z.clone()
        z1[0, max_ind] = v

        # plot encodings
        ax[0].bar(np.arange(z1.shape[1]), height=z1.detach().numpy()[0], width=width)
        ax[0].scatter(np.arange(z1.shape[1]), z1.detach().numpy()[0], color='blue')
        ax[0].scatter(max_ind, z1[0, max_ind].detach().numpy(), color='red')
        ax[0].set_title(r"latent dimension %d - $\alpha$ = %.2f, changed value %.4f "\
                        "(original %.4f)" % (z1.shape[1], model.alpha, v, maxz), fontsize=10)
        ax[0].set_ylim(-3, 3)

        # plot reconstruction
        dec = model.decode(z1).view(-1, *img_size)
        img = torchvision.utils.make_grid(dec)
        if negative:
            # exchange white and black
            img = 1 - img.detach().numpy()
        ax[1].imshow(np.transpose(img, (1, 2, 0)))
        ax[1].set_title('reconstruction', fontsize=10)

        tmpfile = os.path.join(tmpdir, 'tmp_latent_{}.png'.format(int(time.time())))
        plt.savefig(tmpfile, dpi=600)
        plt.close()

        images.append(imageio.imread(tmpfile))

    # save gif file
    imageio.mimsave(filename, images)

    # delete temp directory
    shutil.rmtree(tmpdir)



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
    model = SparseVAE(img_size[0]*img_size[1], train_args.hidden_sizes,
                      train_args.dim_z, train_args.alpha)
    model.c = model_checkpoint['extra']['model_c'] # update c
    model.load_state_dict(model_state_dict)

    # mnist dataset
    test_loader = prepare_data_mnist(train_args.batch_size, train_conf['data_dir'],
                                     train=False, shuffle=False)
    images, _ = iter(test_loader).next()

    # traversal
    print(traversal_vals)
    img_size = (1, *img_size)
    for i in range(5):
        interpretability_traversal(model, traversal_vals, img_size,
                                   os.path.join(save_dir, "test{}.gif").format(i), images[i])
