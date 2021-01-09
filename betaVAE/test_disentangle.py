import os
import torch
from torchvision.utils import save_image

from beta_vae import BetaVAE


def set_test_config():
    """set test configuration"""
    conf = {}
    conf['checkpoint_path'] = os.path.join(
        os.path.dirname(__file__),
        'checkpoints/mnist/t1610177574-epoch150_z15_beta4.pth.tar')
    conf['save_dir'] = os.path.join(
        os.path.dirname(__file__),
        'results/latent_traversal')
    conf['t_range'] = (-3, 3, 6 / 15)
    
    if not os.path.exists(conf['save_dir']):
        os.makedirs(conf['save_dir'])

    return conf


def latent_traversal(model, t_range, img_size, save_dir,
                     tag='latent_traversal', base_input=None, num=10):
    """traverse the latent space for each dimension
    :param model: the vae model
    :param t_range: the traversal range for each dimension,
                    (start, end, step)
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
    
    val_range = torch.arange(*t_range)
    num_vals = len(val_range)
    for i in range(codes.size(0)):
        # traversal for each code
        c = codes[i].clone()
        imgs = []
        for j in range(dim_z):
            # traverse each dimension
            cmat = torch.cat([c.clone().unsqueeze(0) for _ in range(num_vals)])
            cmat[:, j] = val_range
            dec = model.decode(cmat)
            dec = dec.view(-1, 1, img_size[0], img_size[1])
            imgs.append(dec)
        
        save_image(torch.cat(imgs).cpu(),
                   os.path.join(save_dir, "{}_{}.png".format(tag, i)),
                   nrow=num_vals)



if __name__ == '__main__':
    test_conf = set_test_config()
    checkpoint_path = test_conf['checkpoint_path']
    save_dir = test_conf['save_dir']
    t_range = test_conf['t_range']
    
    model_checkpoint = torch.load(checkpoint_path)
    train_args = model_checkpoint['train_args']
    train_conf = model_checkpoint['train_config']
    model_state_dict = model_checkpoint['state_dict']

    img_size = train_conf['image_size']
    model = BetaVAE(img_size[0]*img_size[1], train_args.n_hidden,
                    train_args.dim_z, img_size[0]*img_size[1], train_args.beta)
    model.load_state_dict(model_state_dict)

    # traversal
    latent_traversal(model, t_range, img_size, save_dir)
    