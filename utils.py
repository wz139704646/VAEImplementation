import os
import time
import math
import torch
import torchvision
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from torchvision.utils import save_image


def create_mlp(dim_input, dim_hiddens, act_layer=nn.ReLU,
               act_args={"inplace": True}, norm=True):
    """create MLP network
    :param dim_input: dimension of the input layer
    :param dim_hiddens: a list of dimensions of the hidden layers
    :param act_layer: activation layer class
    :param act_args: dict args used to init act_layer (None or {} if no need)
    :param norm: whether to normalize (use BatchNorm1d)
    :return: a MLP network (include len(dim_hiddens) layers)
    """
    modules = []
    dim_in = dim_input
    act_args = act_args or {}
    # hidden layers
    for dim_h in dim_hiddens:
        modules.append(nn.Linear(dim_in, dim_h))
        if norm:
            modules.append(nn.BatchNorm1d(dim_h))
        modules.append(act_layer(**act_args))

        dim_in = dim_h

    return nn.Sequential(*modules)


def create_cnn2d(channel_in, channel_hiddens, kernel_size, stride, padding,
                 act_layer=nn.ReLU, act_args={"inplace": True}, norm=True):
    """create 2d CNN
    :param channel_in: number of input channels
    :param channel_hiddens: number of the input channels of the hidden layers
    :param kernel_size: param kernel_size for all the conv2d layers
    :param stride: param stride for all the Conv2d layers
    :param padding: param padding for all the Conv2d layers
    :param act_layer: activation layer class
    :param act_args: dict args used to init act_layer (None or {} if no need)
    :param norm: whether to normalize (use BatchNorm1d)
    :return: a 2d CNN network (include len(dim_hiddens) layers)
    """
    modules = []
    ch_in = channel_in
    act_args = act_args or {}
    # hidden layers
    for ch_h in channel_hiddens:
        modules.append(nn.Conv2d(ch_in, ch_h, kernel_size, stride, padding))
        if norm:
            modules.append(nn.BatchNorm2d(ch_h))
        modules.append(act_layer(**act_args))

        ch_in = ch_h

    return nn.Sequential(*modules)


def create_transpose_cnn2d(channel_in, channel_hiddens, kernel_size, stride,
                           padding, output_padding, act_layer=nn.ReLU,
                           act_args={"inplace": True}, norm=True):
    """create 2d transposed CNN
    :param channel_in: number of input channels
    :param channel_hiddens: number of the input channels of the hidden layers
    :param kernel_size: param kernel_size for all the conv2d layers
    :param stride: param stride for all the ConvTranspose2d layers
    :param padding: param padding for all the ConvTranspose2d layers
    :param output_padding: param out_padding for all the ConvTranspose2d layers
    :param act_layer: activation layer class
    :param act_args: dict args used to init act_layer (None or {} if no need)
    :param norm: whether to normalize (use BatchNorm1d)
    :return: a 2d CNN network (include len(dim_hiddens) layers)
    """
    modules = []
    ch_in = channel_in
    act_args = act_args or {}
    # hidden layers
    for ch_h in channel_hiddens:
        modules.append(nn.ConvTranspose2d(
            ch_in, ch_h, kernel_size, stride, padding, output_padding))
        if norm:
            modules.append(nn.BatchNorm2d(ch_h))
        modules.append(act_layer(**act_args))

        ch_in = ch_h

    return nn.Sequential(*modules)


def cal_cnn2d_shape(h_in, w_in, kernel_size, n_layers=1,
                    stride=1, padding=0, dilation=1):
    """calculate the output shape of cnns with input shape h_in x w_in
    :param n_layers: number of cnn2d layers with the same param
    """
    h_out, w_out = h_in, w_in
    for _ in range(n_layers):
        h_out = math.floor(
            (h_out + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
        w_out = math.floor(
            (w_out + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)

    return h_out, w_out


def save_(model, save_dir, args=None, config=None, extra=None, comment=None):
    """save the model accompany with its args and configuration
    :param model: the vae model object
    :param save_dir: directory to put the saved files
    :param args: the input arguments when training
    :param config: the config of the training
    :param extra: others to save
    :param comment: tag at the end of filename
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = "t{}".format(int(time.time()))
    if comment is not None:
        filename = "{}-{}".format(filename, comment)
    filename = os.path.join(save_dir, filename + '.pth.tar')

    torch.save({'train_args': args,
                'train_config': config,
                'state_dict': model.state_dict(),
                'extra': extra}, filename)


def latent_traversal_detailed(model, traversal_vals, img_size, save_dir,
                              tag='latent_traversal', base_input=None, num=10):
    """traverse the latent space for each dimension
    :param model: the vae model
    :param traversal_vals: the traversal values for each dimension
    :param tag: the tag string in the traversal results filenames
    :param base_input: the base original input for the model to encode,
                       default None (will sample num samples from latent
                       space under normal distribution)
    :param num: number of base samples (only work when base_input is None)
    """
    dim_z = model.dim_z
    if base_input is not None:
        codes = model.encode(base_input)[0]
    else:
        codes = torch.randn(num, dim_z)

    num_vals = traversal_vals.size(0)
    for i in range(codes.size(0)):
        # traversal for each code
        c = codes[i].clone()
        imgs = []
        for j in range(dim_z):
            # traverse each dimension
            cmat = torch.cat([c.clone().unsqueeze(0) for _ in range(num_vals)])
            cmat[:, j] = traversal_vals
            dec = model.decode(cmat)
            dec = dec.view(-1, *img_size)
            imgs.append(dec)

        save_image(torch.cat(imgs).cpu(),
                   os.path.join(save_dir, "{}_{}.png".format(tag, i)),
                   nrow=num_vals)


def plot_losses(losses, save_path=None, layout='v'):
    """plot the losses change over epoches
    :param losses: a dict mapping loss name to list of loss values (up to 9 categories)
    :param save_dir: if not None, save the fig at the path
    :param layout: currently 'v' and 'h' supported, specify how the subplot placed
    """
    cates = list(losses.keys())
    num = len(cates)
    assert num <= 9

    # default vertical layout
    nrow = 1
    ncol = 1
    if num >= 3:
        nrow = 3
        ncol = (num-1) // 3 + 1
    else:
        nrow = num
        ncol = 1

    if layout == 'v':
        # vertical layout
        pass
    elif layout == 'h':
        # horizontal layout
        nrow, ncol = ncol, nrow
    else:
        raise NotImplementedError

    fig = plt.figure(figsize=(12, 12))
    for i in range(num):
        idx = i + 1
        if layout == 'v':
            col = i // nrow # the subplot col no.
            row = i % nrow # the subplot row no.
            idx = row * ncol + col + 1
        elif layout == 'h':
            pass

        ax = fig.add_subplot(nrow, ncol, idx)
        ax.set_title(cates[i])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.plot(losses[cates[i]])

    if save_path is not None:
        plt.savefig(save_path)
