import os
import torch
import argparse
from torch import optim
from torchvision.utils import save_image
import matplotlib
import matplotlib.pyplot as plt

from joint_vae import ConvJointVAE

import sys
sys.path.append(".")
sys.path.append("..")
from utils import save_, plot_losses
from load_data import prepare_data_mnist


global_conf = {}


def parse_args():
    """parse command line arguments"""
    desc = "Implementation of Joint VAE based on pytorch, using MNIST dataset"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='batch size for training (default 64)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between logging training status (default 10)')
    parser.add_argument('--temperature', type=float, default=0.67, metavar='TEMP',
                        help='temperature for gumbel softmax distribution')
    parser.add_argument('--disc-dims', nargs='+', type=int, default=[10], metavar='DDs',
                        help='dimensions of discrete latent variables')
    parser.add_argument('--disc-cap', nargs=4, type=float, default=[0., 5., 2e-4, 30.],
                        metavar=('DCap_min', 'DCap_max', 'DCap_delta', 'DCap_gamma'),
                        help='capacity for discrete channels (min, max, delta_per_step, gamma)')
    parser.add_argument('--cont-cap', nargs=4, type=float, default=[0., 5., 2e-4, 30.],
                        metavar=('CCap_min', 'CCap_max', 'CCap_delta', 'CCap_gamma'),
                        help='capacity for continuous channels (min, max, delta_per_step, gamma)')
    parser.add_argument('--channels', nargs='+', type=int, default=[32, 32, 64, 64], metavar='N',
                        help='number of channels of hidden layers (default [32 32 64 64])')
    parser.add_argument('--hidden-size', type=int, default=256, metavar='N',
                        help='number of hidden units in MLP layer before latent layer (default 256)')
    parser.add_argument('--dim-z', type=int, default=20, metavar='N',
                        help='dimension of latent space (total) (default 20)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate of optimizer (default 1e-3)')
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

    global_conf['device'] = torch.device("cuda" if args.cuda else "cpu")
    global_conf['image_size'] = (1, 32, 32)
    global_conf['data_dir'] = os.path.join(os.path.dirname(__file__), '../data')
    global_conf['res_dir'] = os.path.join(os.path.dirname(__file__), './results')
    global_conf['checkpoints_dir'] = os.path.join(os.path.dirname(__file__), './checkpoints')


def prepare_data(args, dir_path, shuffle=True):
    """prepare data for training and testing"""
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    image_size = global_conf['image_size']
    resize = None
    if len(image_size) == 2:
        resize = image_size
    elif len(image_size) > 2:
        resize = (image_size[-2], image_size[-1])
    else:
        raise RuntimeError("config image size is illegal")

    train_loader = prepare_data_mnist(
        args.batch_size, dir_path, train=True, shuffle=shuffle,
        resize=resize, **kwargs)
    test_loader = prepare_data_mnist(
        args.batch_size, dir_path, train=False, shuffle=shuffle,
        resize=resize, **kwargs)

    return train_loader, test_loader


def train(model, train_loader, epoch, optimizer, args, device):
    """VAE training process"""
    model.train()
    train_loss = {}

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()
        res = model(data)
        loss_dict = model.loss_function(*res, data)

        # record all the losses
        for k in loss_dict.keys():
            if k in train_loss:
                train_loss[k] += loss_dict[k].item()
            else:
                train_loss[k] = loss_dict[k].item()

        loss = loss_dict["loss"] # final loss
        loss.backward()
        optimizer.step()

        # the updates after each train step
        model.update_step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: \n{}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss_dict
            ))

    avg_loss = {}
    for k in train_loss.keys():
        avg_loss[k] = train_loss[k] / len(train_loader.dataset) * args.batch_size
    print('=====> Epoch: {} Average loss: \n{}'.format(
        epoch, avg_loss
    ))

    return avg_loss


def test(model, test_loader, epoch, args, device, res_dir):
    """VAE testing process"""
    model.eval()
    test_loss = {}

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            res = model(data)
            loss_dict = model.loss_function(*res, data)

            # record all the losses
            for k in loss_dict.keys():
                if k in test_loss:
                    test_loss[k] += loss_dict[k]
                else:
                    test_loss[k] = loss_dict[k]

            if i == 0:
                recon_batch = model.reconstruct(data)
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch[:n]])
                save_image(comparison.cpu(), res_dir +
                           '/reconstruction_'+str(epoch)+'.png', nrow=n)

    for k in test_loss:
        test_loss[k] = test_loss[k] / len(test_loader.dataset) * args.batch_size
    print('=====> Test set loss: \n{}'.format(test_loss))


def main(args):
    """main procedure"""
    # get configuration
    device = global_conf["device"]
    image_size = global_conf["image_size"]
    data_dir = global_conf["data_dir"]
    res_dir = global_conf["res_dir"]
    save_dir = global_conf["checkpoints_dir"]

    # prepare data
    train_loader, test_loader = prepare_data(args, dir_path=data_dir)

    # prepare model
    model = ConvJointVAE(
        args.temperature, args.disc_dims, args.disc_cap, args.cont_cap,
        image_size, args.channels, args.hidden_size, args.dim_z)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(model)

    # train and test
    losses = {}
    for epoch in range(1, args.epochs+1):
        avg_loss = train(model, train_loader, epoch,
                         optimizer, args, device)

        # record all the losses
        for k in avg_loss.keys():
            if k in losses:
                losses[k].append(avg_loss[k])
            else:
                losses[k] = [avg_loss[k]]

        test(model, test_loader, epoch, args, device, res_dir)
        with torch.no_grad():
            sample = model.sample(64, device).cpu()
            save_image(sample, res_dir+'/sample_'+str(epoch)+'.png')

    # plot train losses
    plot_losses(losses, res_dir+'/loss.png')

    # save the model and related params
    if args.save:
        save_dir = os.path.join(save_dir, 'mnist')
        save_(model, save_dir, args, global_conf, comment=args.tag,
              extra={"cont_cap": model.cont_cap_current,
                     "disc_cap": model.disc_cap_current})


if __name__ == '__main__':
    args = parse_args()
    configuration(args)

    main(args)
