import torch
import numpy as np
import torch.optim as optim
import torch.distributions as td
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset, Dataset
from torch.utils.data import ConcatDataset
import sklearn.datasets
import pickle

import medmnist
from medmnist import BreastMNIST


class SyntheticDataset(Dataset):
    def __init__(self, n=1000):
        dataset_init = sklearn.datasets.make_classification(n_samples=n, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
        self.data = torch.Tensor(dataset_init[0])
        self.targets = torch.Tensor(dataset_init[1])
        self.dataset = []
        for i in range(len(self.data)):
            datapoint = (self.data[i], int(self.targets[i].item()))
            self.dataset.append(datapoint)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    
def loaddataset(name_data):
    if name_data == 'binarymnist':
        
        # Load MNIST data
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train = datasets.MNIST(
                    'mnist-data/', train=True, download=True, transform=transform)
        test = datasets.MNIST(
                    'mnist-data/', train=False, download=True, transform=transform)
        # Binarize
        train.targets = (train.targets > 4).float()
        test.targets = (test.targets > 4).float()
 
    elif name_data == 'medmnist':
        data_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])]
        )

        # load the data
        download = True
        input_root = 'data'
        train_short = BreastMNIST(root=input_root, split='train', transform=data_transform, download=download)
        val = BreastMNIST(root=input_root, split='val', transform=data_transform, download=download)
        train = ConcatDataset([train_short, val])
        test = BreastMNIST(root=input_root, split='test', transform=data_transform, download=download)


    elif name_data == 'synthetic':
        n_train = 100
        n_test = 50000
        train = SyntheticDataset(n=n_train)
        test = SyntheticDataset(n=n_test)

    return train, test


# The below is adapted from https://github.com/mperezortiz/PBB
def loadbatches(train, test, loader_kargs, batch_size, perc_train=1.0, perc_val=.1):
    """Function to load the batches for the dataset

    Parameters
    ----------
    train : torch dataset object
        train split
    
    test : torch dataset object
        test split 

    loader_kargs : dictionary
        loader arguments
    
    batch_size : int
        size of the batch


    perc_train : float
        percentage of data used for training (set to 1.0 if not intending to do data scarcity experiments)

    """

    n_train = len(train)
    n_test = len(test)

    # reduce number of training/validation data if needed
    new_num_train = int(np.round((perc_train)*n_train))

    # split into training and "test" set (for computing bound)
    num_val = int(np.round((perc_val)*new_num_train))
    num_train = new_num_train - num_val
    # train_data, val_data = torch.utils.data.random_split(train, [num_train, num_val])
    
    # NEW
    indices_full = list(range(n_train))
    np.random.shuffle(indices_full)
    indices_train = list(np.random.choice(indices_full, new_num_train, replace=False))
    new_train = Subset(train,indices_train)
    train_data, val_data = torch.utils.data.random_split(new_train, [num_train, num_val])
    
    # print(f"train size {len(train_data)}")
    # print(f"val size {len(val_data)}")
    
    # create data loaders
    indices = list(range(num_train))
    np.random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True,  **loader_kargs)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, **loader_kargs)
    
#     with open('val.pkl', 'wb') as f:
#         pickle.dump(val_data, f)
    
    return train_loader, val_loader, test_loader
