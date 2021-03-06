import os
import torch
import argparse
from torch import optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt

from factor_vae import FactorVAE

import sys
sys.path.append(".")
sys.path.append("..")
from utils import save_, plot_losses
from load_data import prepare_data_mnist


global_conf = {}


def parse_args():
    """parse command line arguments"""
    desc = "Implementation of FactorVAE based on pytorch, using MNIST dataset"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='batch size for training (default 128)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between logging training status (default 10)')
    parser.add_argument('--n-hidden', type=int, default=400, metavar='N',
                        help='number of hidden units in MLP (default 400)')
    parser.add_argument('--dim-z', type=int, default=10, metavar='N',
                        help='dimension of latent space (default 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate of optimizer (default 1e-3)')
    parser.add_argument('--gamma', type=float, default=30., metavar=':math: `\\gamma`',
                        help='weight on tc loss term of FactorVAE (default 30.)')
    parser.add_argument('--save', action='store_true', default=False,
                        help='save the model (under the checkpoints path (defined in config))')
    parser.add_argument('--tag', type=str, default=None, metavar='T',
                        help='tag string for the save model file name (default None (no tag))')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


def configuration(args):
    """set global configuration for initialization"""
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)

    global_conf['device'] = torch.device("cuda" if args.cuda else "cpu")
    global_conf['image_size'] = (28, 28)
    global_conf['data_dir'] = os.path.join(os.path.dirname(__file__), '../data')
    global_conf['res_dir'] = os.path.join(os.path.dirname(__file__), './results')
    global_conf['checkpoints_dir'] = os.path.join(os.path.dirname(__file__), './checkpoints')


def prepare_data(args, dir_path, shuffle=True):
    """prepare data for training/testing"""
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader1 = prepare_data_mnist(
        args.batch_size, dir_path, train=True, shuffle=shuffle, **kwargs)
    train_loader2 = DataLoader(
        train_loader1.dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader1 = prepare_data_mnist(
        args.batch_size, dir_path, train=False, shuffle=shuffle, **kwargs)
    test_loader2 = DataLoader(
        test_loader1.dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    return [train_loader1, train_loader2], [test_loader1, test_loader2]


def train(model, train_loaders, epoch, optimizers, args, device, img_size):
    """FactorVAE training process"""
    model.train()
    train_loss_vae = 0
    train_loss_discriminator = 0
    dataset_size = len(train_loaders[0].dataset)
    optimizer_vae = optimizers['vae']
    optimizer_discriminator = optimizers['discriminator']

    train_loader = zip(*[enumerate(l) for l in train_loaders])
    for (batch_idx, (data1, _)), (_, (data2, _)) in train_loader:
        data1 = data1.to(device)
        data2 = data2.to(device)

        # train vae
        decoded, encoded, z = model(data1.view(-1, img_size[0]*img_size[1]))
        vae_loss = model.loss_function(
            decoded, data1.view(-1, img_size[0]*img_size[1]), encoded, z, optim_index=0)
        optimizer_vae.zero_grad()
        vae_loss.backward(retain_graph=True)
        train_loss_vae += vae_loss.item()
        optimizer_vae.step()

        # train discriminator
        z_prime = model(data2.view(-1, img_size[0]*img_size[1]), no_dec=True)
        discriminator_loss = model.loss_function(z, z_prime, optim_index=1)
        optimizer_discriminator.zero_grad()
        discriminator_loss.backward()
        train_loss_discriminator += discriminator_loss.item()
        optimizer_discriminator.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t vae loss: {:.6f}, dicriminator loss: {:.6f}'.format(
                epoch, batch_idx * len(data1), dataset_size,
                100. * batch_idx / len(train_loaders[0]),
                vae_loss.item(), discriminator_loss.item()
            ))

    avg_loss_vae = train_loss_vae / dataset_size * args.batch_size
    avg_loss_discriminator = train_loss_discriminator / dataset_size * args.batch_size
    print('=====> Epoch: {} Average loss: vae loss {:.4f}, discriminator loss {:.4f}'.format(
        epoch, avg_loss_vae, avg_loss_discriminator
    ))

    return {"vae": avg_loss_vae, "discriminator": avg_loss_discriminator}


def test(model, test_loaders, epoch, args, device, img_size, res_dir):
    """FactorVAE testing process"""
    model.eval()
    test_loss_vae = 0
    test_loss_discriminator = 0
    dataset_size = len(test_loaders[0].dataset)
    test_loader = zip(*[enumerate(l) for l in test_loaders])

    with torch.no_grad():
        for (i, (data1, _)), (_, (data2, _)) in test_loader:
            data1 = data1.to(device)
            data2 = data2.to(device)

            # test vae
            decoded, encoded, z = model(data1.view(-1, img_size[0]*img_size[1]))
            test_loss_vae += model.loss_function(
                decoded, data1.view(-1, img_size[0]*img_size[1]), encoded, z, optim_index=0).item()

            # test discriminator
            z_prime = model(data2.view(-1, img_size[0]*img_size[1]), no_dec=True)
            test_loss_discriminator += model.loss_function(z, z_prime, optim_index=1)

            if i == 0:
                recon_batch = model.reconstruct(
                    data1.view(-1, img_size[0]*img_size[1]))
                n = min(data1.size(0), 8)
                comparison = torch.cat([data1[:n], recon_batch.view(
                    args.batch_size, 1, img_size[0], img_size[1])[:n]])
                save_image(comparison.cpu(), res_dir +
                           '/reconstruction_'+str(epoch)+'.png', nrow=n)

    test_loss_vae = test_loss_vae / dataset_size * args.batch_size
    test_loss_discriminator = test_loss_discriminator / dataset_size * args.batch_size
    print('=====> Test set loss: vae {:.4f}, discriminator {:.4f}'.format(test_loss_vae, test_loss_discriminator))

    return {"vae": test_loss_vae, "discriminator": test_loss_discriminator}


def main(args):
    """main procedure"""
    # get configuration
    device = global_conf["device"]
    img_size = global_conf["image_size"]
    data_dir = global_conf["data_dir"]
    res_dir = global_conf["res_dir"]
    save_dir = global_conf["checkpoints_dir"]

    # prepare data
    train_loaders, test_loaders = prepare_data(args, dir_path=data_dir)

    # prepare model
    model = FactorVAE(img_size[0]*img_size[1], args.n_hidden, args.dim_z,
                    img_size[0]*img_size[1], args.gamma)
    optimizer_vae = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_discriminator = optim.Adam(model.discriminator.parameters(), lr=args.lr)
    optimizers = {"vae": optimizer_vae, "discriminator": optimizer_discriminator}

    # train and test
    losses_vae = []
    losses_discriminator = []
    for epoch in range(1, args.epochs+1):
        avg_loss = train(model, train_loaders, epoch,
                         optimizers, args, device, img_size)
        losses_vae.append(avg_loss["vae"])
        losses_discriminator.append(avg_loss["discriminator"])
        test(model, test_loaders, epoch, args, device, img_size, res_dir)
        with torch.no_grad():
            sample = model.sample(64, device).cpu()
            save_image(sample.view(
                64, 1, img_size[0], img_size[1]), res_dir+'/sample_'+str(epoch)+'.png')

    # plot train losses
    plot_losses({"vae": losses_vae, "discriminator": losses_discriminator}, res_dir+'/loss.png')

    # save the model and related params
    if args.save:
        save_dir = os.path.join(save_dir, 'mnist')
        save_(model, save_dir, args, global_conf, comment=args.tag)


if __name__ == '__main__':
    args = parse_args()
    configuration(args)

    main(args)
