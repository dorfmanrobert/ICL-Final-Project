import torch
import numpy as np
import torch.optim as optim
import torch.distributions as td
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from medmnist import BreastMNIST


def loaddataset(name):
    """Function to load the datasets (mnist and cifar10)

    Parameters
    ----------
    name : string
        name of the dataset ('mnist' or 'cifar10')

    """

    torch.manual_seed(7)
    if name == 'binarymnist':
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
        
    
    elif name == 'medmnist':
        data_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])]
        )

        # load the data
        download = True
        input_root = 'data'
        train = BreastMNIST(split='train', transform=data_transform, download=download)
        test = BreastMNIST(split='test', transform=data_transform, download=download)


    else:
        raise RuntimeError(f'Wrong dataset chosen {name}')

    return train, test


def loadbatches(train, test, loader_kargs, batch_size, perc_train=1.0, perc_prior=.5):
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

    prior : bool
        boolean indicating the use of a learnt prior (e.g. this would be False for a random prior)

    perc_train : float
        percentage of data used for training (set to 1.0 if not intending to do data scarcity experiments)

    perc_prior : float
        percentage of data to use for building the prior (1-perc_prior is used to estimate the risk)

    """

    ntrain = len(train)
    ntest = len(test)
    
    # reduce training data if needed
    new_num_train = int(np.round((perc_train)*ntrain))
    
    indices = list(range(new_num_train))
    split = int(np.round((perc_prior)*new_num_train))
    random_seed = 10
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    all_train_sampler = SubsetRandomSampler(indices)
    train_idx, valid_idx = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_idx)  # to sample first half of data
    valid_sampler = SubsetRandomSampler(valid_idx)  # to sample second half of data

    # Training splits for either training procedure
    train_loader_1 = torch.utils.data.DataLoader(train, batch_size=batch_size, sampler=train_sampler, shuffle=False)
    train_loader_2 = torch.utils.data.DataLoader(train, batch_size=batch_size, sampler=valid_sampler, shuffle=False)
    set_bound_1batch_1 = torch.utils.data.DataLoader(train, batch_size=len(valid_idx), sampler=valid_sampler, **loader_kargs)
    set_bound_1batch_2 = torch.utils.data.DataLoader(train, batch_size=len(train_idx), sampler=train_sampler, **loader_kargs)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, sampler=all_train_sampler, shuffle=False)
    train_loader_1batch = torch.utils.data.DataLoader(train, batch_size=ntrain, sampler=valid_sampler, **loader_kargs)

    test_loader = torch.utils.data.DataLoader(test, batch_size=ntest, shuffle=True, **loader_kargs)

    # train_loader_1 use to train prior_1 and posterior_2
    # train_loader_2 use to train prior_2 and posterior_1
    # set_bound_1batch_1 to evaluate bound for posterior_1 since posterior_1 trained on valid_sampler
    # set_bound_1batch_2 to evaluate bound for posterior_2 since posterior_2 trained on train_sampler=
    # train_loader is all training data used in training one posterior

    return train_loader_1batch, train_loader_1, train_loader_2, set_bound_1batch_1, set_bound_1batch_2, train_loader, test_loader

