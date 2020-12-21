# beta Total Correlation Variational Auto-Encoder

使用 pytorch 实现，基于 MNIST 数据集。

## Paper

[Isolating Sources of Disentanglement in Variational Autoencoders](https://arxiv.org/abs/1802.04942)

## Usage

```shell
python main.py --help
```

output:

```
usage: main.py [-h] [--batch-size N] [--epochs N] [--no-cuda] [--seed S]
               [--log-interval N] [--n-hidden N] [--dim-z N] [--lr LR]
               [--alpha :math: `\alpha`] [--beta :math: `\beta`]
               [--gamma :math: `\gamma`] [--sampling S]

Implementation of beta-TCVAE based on pytorch, using MNIST dataset

optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        batch size for training (default 128)
  --epochs N            number of epochs to train (default 20)
  --no-cuda             disable CUDA training
  --seed S              random seed (default 1)
  --log-interval N      interval between logging training status (default 10)
  --n-hidden N          number of hidden units in MLP (default 400)
  --dim-z N             dimension of latent space (default 10)
  --lr LR               learning rate of optimizer (default 1e-3)
  --alpha :math: `\alpha`
                        alpha coefficient (on index-code MI term) of beta-
                        TCVAE (default 1.)
  --beta :math: `\beta`
                        beta coefficient (on TC term) of beta-TCVAE (default
                        6.)
  --gamma :math: `\gamma`
                        gamma coefficient (on dimension wise KL term) of beta-
                        TCVAE (default 1.)
  --sampling S          sampling method applied in beta-TCVAE for computing
                        q(z) (default "mws")
```

example:

```shell
python main.py --epochs 60 --batch-size 100 --alpha 0 --beta 5 --gamma 1 --sampling mss
```

## References

1. https://github.com/AntixK/PyTorch-VAE
2. https://github.com/rtqichen/beta-tcvae (official)

