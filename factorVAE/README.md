# Factor Variational Auto-Encoder

使用 pytorch 实现，基于 MNIST 数据集。

## Paper

[Disentangling by Factorising](https://arxiv.org/pdf/1802.05983.pdf)

## Usage

```shell
python main.py --help
```

output:

```
usage: main.py [-h] [--batch-size N] [--epochs N] [--no-cuda] [--seed S]
               [--log-interval N] [--n-hidden N] [--dim-z N] [--lr LR]
               [--gamma :math: `\gamma`]

Implementation of FactorVAE based on pytorch, using MNIST dataset

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
  --gamma :math: `\gamma`
                        weight on tc loss term of FactorVAE (default 30.)
```

example:

```shell
python main.py --epochs 60 --batch-size 100 --gamma 10
```

## References

1. https://github.com/AntixK/PyTorch-VAE
2. https://github.com/1Konny/FactorVAE

