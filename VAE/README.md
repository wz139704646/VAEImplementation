# Variational Auto-Encoder

使用 pytorch 实现，基于 MNIST 数据集。

## Paper

[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

## Usage

```shell
python main.py --help
```

output:

```
usage: main.py [-h] [--batch-size N] [--epochs N] [--no-cuda] [--seed S]
               [--log-interval N] [--n-hidden N] [--dim-z N] [--lr LR]

Implementation of VAE based on pytorch

optional arguments:
  -h, --help        show this help message and exit
  --batch-size N    batch size for training (default 128)
  --epochs N        number of epochs to train (default 20)
  --no-cuda         disable CUDA training
  --seed S          random seed (default 1)
  --log-interval N  interval between logging training status (default 10)
  --n-hidden N      number of hidden units in MLP (default 400)
  --dim-z N         dimension of latent space (default 20)
  --lr LR           learning rate of optimizer (default 1e-3)
```

example:

```shell
python main.py --no-cuda --epochs 60 --batch-size 100
```

## References

1. https://github.com/pytorch/examples/tree/master/vae
2. https://github.com/hwalsuklee/tensorflow-mnist-VAE

