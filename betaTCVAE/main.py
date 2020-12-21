import torch
import argparse
from torch import optim
from torchvision.utils import save_image
import matplotlib
import matplotlib.pyplot as plt

from beta_tcvae import BetaTCVAE

import sys
sys.path.append("..")
from load_data import prepare_data_mnist


global_conf = {}


def parse_args():
    """parse command line arguments"""
    desc = "Implementation of beta-TCVAE based on pytorch, using MNIST dataset"
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
    parser.add_argument('--alpha', type=float, default=1., metavar=':math: `\\alpha`',
                        help='alpha coefficient (on index-code MI term) of beta-TCVAE (default 1.)')
    parser.add_argument('--beta', type=float, default=6., metavar=':math: `\\beta`',
                        help='beta coefficient (on TC term) of beta-TCVAE (default 6.)')
    parser.add_argument('--gamma', type=float, default=1., metavar=':math: `\\gamma`',
                        help='gamma coefficient (on dimension wise KL term) of beta-TCVAE (default 1.)')
    parser.add_argument('--sampling', type=str, default='mws', metavar='S', choices=['mws', 'mss'],
                        help='sampling method applied in beta-TCVAE for computing q(z) (default "mws")')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args


def configuration(args):
    """set global configuration for initialization"""
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    global_conf['device'] = torch.device("cuda" if args.cuda else "cpu")
    global_conf['image_size'] = (28, 28)
    global_conf['data_dir'] = '../data'
    global_conf['res_dir'] = './results'


def prepare_data(args, dir_path, shuffle=True):
    """prepare data for training/testing"""
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = prepare_data_mnist(args.batch_size, dir_path, train=True, shuffle=shuffle, **kwargs)
    test_loader = prepare_data_mnist(args.batch_size, dir_path, train=False, shuffle=shuffle, **kwargs)

    return train_loader, test_loader


def train(model, train_loader, epoch, optimizer, args, device, img_size):
    """beta-TCVAE training process"""
    model.train()
    train_loss = 0
    dataset_size = len(train_loader.dataset)

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        decoded, encoded, z = model(data.view(-1, img_size[0]*img_size[1]))
        loss = model.loss_function(decoded, data, encoded, z, dataset_size=dataset_size)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx * len(data), dataset_size,
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)
            ))
        
    avg_loss = train_loss / len(train_loader.dataset)
    print('=====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, avg_loss
    ))

    return avg_loss


def test(model, test_loader, epoch, args, device, img_size, res_dir):
    """beta-TCVAE testing process"""
    model.eval()
    test_loss = 0
    dataset_size = len(test_loader.dataset)

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            decoded, encoded, z = model(data.view(-1, img_size[0]*img_size[1]))
            test_loss += model.loss_function(decoded, data, encoded, z, dataset_size=dataset_size).item()

            if i == 0:
                recon_batch = model.reconstruct(data.view(-1, img_size[0]*img_size[1]))
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch.view(args.batch_size, 1, img_size[0], img_size[1])[:n]])
                save_image(comparison.cpu(), res_dir+'/reconstruction_'+str(epoch)+'.png', nrow=n)
            
    test_loss /= len(test_loader.dataset)
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
    model = BetaTCVAE(img_size[0]*img_size[1], args.n_hidden, args.dim_z, img_size[0]*img_size[1],
                    args.alpha, args.beta, args.gamma, sampling=args.sampling)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # train and test
    losses = []
    for epoch in range(1, args.epochs+1):
        avg_loss = train(model, train_loader, epoch, optimizer, args, device, img_size)
        losses.append(avg_loss)
        test(model, test_loader, epoch, args, device, img_size, res_dir)
        with torch.no_grad():
            sample = model.sample(64, device).cpu()
            save_image(sample.view(64, 1, img_size[0], img_size[1]), res_dir+'/sample_'+str(epoch)+'.png')

    # plot train losses
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.savefig(res_dir+'/loss.png')


if __name__ == '__main__':
    args = parse_args()
    configuration(args)

    main(args)