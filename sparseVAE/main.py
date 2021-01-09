import os
import torch
import argparse
from torch import optim
from torchvision.utils import save_image
import matplotlib
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from sparse_vae import SparseVAE

import sys
sys.path.append("..")
from load_data import prepare_data_mnist


global_conf = {}


def parse_args():
    """parse command line arguments"""
    desc = "Implementation of Sparse VAE based on pytorch, using MNIST dataset"
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
    parser.add_argument('--hidden-sizes', nargs='+', type=int, default=[400], metavar='N',
                        help='numbers of hidden units for each layer in MLP (default [400])')
    parser.add_argument('--dim-z', type=int, default=100, metavar='N',
                        help='dimension of latent space (default 100)')
    parser.add_argument('--alpha', type=float, default=0.5, metavar=':math: `\\alpha`',
                        help='prior sparsity (0-1) (default 0.5)')
    parser.add_argument('--delta-c', type=float, default=0.001, metavar=':math: `\\delta c`',
                        help='increasing rate of c (updated per epoch) (default 0.001)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate of optimizer (default 1e-3)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.hidden_sizes = args.hidden_sizes if hasattr(args.hidden_sizes, '__len__') \
        else [args.hidden_sizes]

    return args


def configuration(args):
    """set global configuration for initialization"""
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    global_conf['device'] = torch.device("cuda" if args.cuda else "cpu")
    global_conf['image_size'] = (28, 28)
    global_conf['data_dir'] = os.path.join(os.path.dirname(__file__), '../data')
    global_conf['res_dir'] = os.path.join(os.path.dirname(__file__), './results')


def prepare_data(args, dir_path, shuffle=True):
    """prepare data for training and testing"""
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = prepare_data_mnist(
        args.batch_size, dir_path, train=True, shuffle=shuffle, **kwargs)
    test_loader = prepare_data_mnist(
        args.batch_size, dir_path, train=False, shuffle=shuffle, **kwargs)

    return train_loader, test_loader


def train(model, train_loader, epoch, optimizer, args, device, img_size, writer):
    """VAE training process"""
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        decoded, encoded = model(data.view(-1, img_size[0]*img_size[1]))
        losses = model.loss_function(
            decoded, data.view(-1, img_size[0]*img_size[1]), encoded)
        for l in losses.keys():
            writer.add_scalar(l, losses[l])
        loss = losses['loss']
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()
            ))

    avg_loss = train_loss / len(train_loader.dataset) * args.batch_size
    print('=====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, avg_loss
    ))

    return avg_loss


def test(model, test_loader, epoch, args, device, img_size, res_dir):
    """VAE testing process"""
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            decoded, encoded = model(data.view(-1, img_size[0]*img_size[1]))
            losses = model.loss_function(
                decoded, data.view(-1, img_size[0]*img_size[1]), encoded)
            test_loss += losses['loss'].item()

            if i == 0:
                recon_batch = model.reconstruct(
                    data.view(-1, img_size[0]*img_size[1]))
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(
                    args.batch_size, 1, img_size[0], img_size[1])[:n]])
                save_image(comparison.cpu(), res_dir +
                           '/reconstruction_'+str(epoch)+'.png', nrow=n)

    test_loss = test_loss / len(test_loader.dataset) * args.batch_size
    print('=====> Test set loss: {:.4f}'.format(test_loss))


def main(args):
    """main procedure"""
    # get configuration
    device = global_conf["device"]
    img_size = global_conf["image_size"]
    data_dir = global_conf["data_dir"]
    res_dir = global_conf["res_dir"]

    # prepare data
    train_loader, test_loader = prepare_data(args, dir_path=data_dir)

    # prepare model
    model = SparseVAE(img_size[0]*img_size[1], args.hidden_sizes,
                      args.dim_z, args.alpha)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # summary writer
    writer = SummaryWriter(comment='sparseVAE')
    writer.add_graph(model, torch.zeros(1, img_size[0]*img_size[1]))

    # train and test
    losses = []
    for epoch in range(1, args.epochs+1):
        loss = train(model, train_loader, epoch,
                     optimizer, args, device, img_size, writer)
        losses.append(loss)
        test(model, test_loader, epoch, args, device, img_size, res_dir)
        with torch.no_grad():
            sample = model.sample(64, device).cpu()
            save_image(sample.view(
                64, 1, img_size[0], img_size[1]), res_dir+'/sample_'+str(epoch)+'.png')
        
        # update c
        model.update_c(args.delta_c)

    # plot train losses
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.savefig(res_dir+'/loss.png')


if __name__ == '__main__':
    args = parse_args()
    configuration(args)

    main(args)
