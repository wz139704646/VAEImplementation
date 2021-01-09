# beta Variational Auto-Encoder

使用 pytorch 实现，基于 MNIST 数据集。

## Paper

[beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)

## Usage

```shell
python main.py --help
```

output:

```
usage: main.py [-h] [--batch-size N] [--epochs N] [--no-cuda] [--seed S]
               [--log-interval N] [--n-hidden N] [--dim-z N] [--lr LR]  
               [--beta :math: `\beta`] [--save] [--tag T]

Implementation of beta-VAE based on pytorch, using MNIST dataset

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
  --beta :math: `\beta`
                        beta coefficient of beta-VAE (default 3.)
  --save                save the model (under the checkpoints path (defined in
                        config))
  --tag T               tag string for the save model file name (default None
                        (no tag))
```

example:

```shell
python main.py --no-cuda --epochs 60 --batch-size 100 --beta 5
```

## References

1. https://github.com/AntixK/PyTorch-VAE
2. https://github.com/1Konny/Beta-VAE

