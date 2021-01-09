# Variational Sparse Coding

使用 pytorch 实现，基于 MNIST 数据集。

## Paper

[Variational Sparse Coding](https://openreview.net/forum?id=SkeJ6iR9Km)

## Usage

```shell
python main.py --help
```

output:

```
usage: main.py [-h] [--batch-size N] [--epochs N] [--no-cuda] [--seed S]
               [--log-interval N] [--hidden-sizes N [N ...]] [--dim-z N]
               [--alpha :math: `\alpha`] [--delta-c :math: `\delta c`]  
               [--lr LR]

Implementation of Sparse VAE based on pytorch, using MNIST dataset      

optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        batch size for training (default 64)
  --epochs N            number of epochs to train (default 20)
  --no-cuda             disable CUDA training
  --seed S              random seed (default 1)
  --log-interval N      interval between logging training status (default 10)
  --hidden-sizes N [N ...]
                        numbers of hidden units for each layer in MLP (default
                        [400])
  --dim-z N             dimension of latent space (default 100)
  --alpha :math: `\alpha`
                        prior sparsity (0-1) (default 0.5)
  --delta-c :math: `\delta c`
                        increasing rate of c (updated per epoch) (default
                        0.001)
  --lr LR               learning rate of optimizer (default 1e-3)
```

example:

```shell
python main.py --no-cuda --epochs 60 --dim-z 200 --batch-size 100 --alpha 0.2 --delta-c 0.01
```

## References

1. https://github.com/AntixK/PyTorch-VAE
2. https://github.com/Alfo5123/Variational-Sparse-Coding

