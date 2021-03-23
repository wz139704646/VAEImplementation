# Joint Variational Auto-Encoder

使用 pytorch 实现，基于 MNIST 数据集。

## Paper

[Learning Disentangled Joint Continuous and Discrete Representations](https://arxiv.org/abs/1804.00104)

## Usage

```shell
python main.py --help
```

output:

```
usage: main.py [-h] [--batch-size N] [--epochs N] [--no-cuda] [--seed S]
               [--log-interval N] [--temperature TEMP]
               [--disc-dims DDs [DDs ...]]
               [--disc-cap DCap_min DCap_max DCap_delta DCap_gamma]
               [--cont-cap CCap_min CCap_max CCap_delta CCap_gamma]
               [--channels N [N ...]] [--hidden-size N] [--dim-z N] [--lr LR]
               [--save] [--tag T]

Implementation of Joint VAE based on pytorch, using MNIST dataset

optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        batch size for training (default 64)
  --epochs N            number of epochs to train (default 20)
  --no-cuda             disable CUDA training
  --seed S              random seed (default 1)
  --log-interval N      interval between logging training status (default 10)
  --temperature TEMP    temperature for gumbel softmax distribution
  --disc-dims DDs [DDs ...]
                        dimensions of discrete latent variables
  --disc-cap DCap_min DCap_max DCap_delta DCap_gamma
                        capacity for discrete channels (min, max,
                        delta_per_step, gamma)
  --cont-cap CCap_min CCap_max CCap_delta CCap_gamma
                        capacity for continuous channels (min, max,
                        delta_per_step, gamma)
  --channels N [N ...]  number of channels of hidden layers (default [32 32 64
                        64])
  --hidden-size N       number of hidden units in MLP layer before latent
                        layer (default 256)
  --dim-z N             dimension of latent space (total) (default 20)
  --lr LR               learning rate of optimizer (default 1e-3)
  --save                save the model (under the checkpoints path (defined in
                        config))
  --tag T               tag string for the save model file name (default None
                        (no tag))
```

example:

```shell
python main.py --epochs 60 --batch-size 50 --disc-dims 10 --disc-cap 0. 5. 0.0002 30 --cont-cap 0. 5. 0.0002 30 --dim-z 20
```

## References

1. https://github.com/Schlumberger/joint-vae

