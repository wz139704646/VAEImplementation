import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


class CustomTensorDataset(Dataset):
    """Custom class for reading tensor datasets"""

    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]
    
    def __len__(self):
        return self.data_tensor.size(0)


def prepare_data_dsprites(batch_size, data_dir, shuffle=True, **kwargs):
    """load dsprites data (since the data are download via url, no train or test option)
    :param data_dir: the root directory path
    :param kwargs: other arguments for the return dataloader
    """
    data_file = "dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    data_file = os.path.join(data_dir, data_file)

    # load dataset
    data = np.load(data_file, encoding='bytes')
    data = torch.from_numpy(data['imgs']).unsqueeze(1).float() # use image data
    dset = CustomTensorDataset(data)

    return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def prepare_data_mnist(batch_size, data_dir, train=True, shuffle=True, **kwargs):
    """load MNIST data via torchvision
    :param data_dir: the root directory path
    :param train: whether the data is used for training (false means testing)
    :param kwargs: other arguments for the return dataloader
    """

    return DataLoader(datasets.MNIST(data_dir, train=train, download=True,
                                    transform=transforms.ToTensor()),
                    batch_size=batch_size, shuffle=shuffle, **kwargs)
