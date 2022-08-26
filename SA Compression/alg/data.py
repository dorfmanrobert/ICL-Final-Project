import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset
from torchvision import datasets, transforms
import numpy as np
import medmnist
from medmnist.dataset import BreastMNIST
import pickle


def loadbatches(train, test, loader_kargs, batch_size=100, perc_train=1.0, num_supp_init=5, continuing=True):
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
    # initialize support and nonsupport indices
    n_train = len(train.data)
    n_test = len(test.data)
    
    # reduce number of training/validation data if needed
    new_num_train = int(np.round((perc_train)*n_train))
    indices_full = list(range(n_train))
    # select uniformly random subset of size new_num_train*n_train (keeps class balance in theory)
    np.random.shuffle(indices_full)
    indices = list(np.random.choice(indices_full, new_num_train, replace=False))
    print(len(indices))

    # if continuing loop, upload the indices
    if continuing == True:
        # pickle indices to files
        with open('supp_indx.pkl', 'rb') as f:
            supp_indx = pickle.load(f)
        with open('nonsupp_indx.pkl', 'rb') as f:
            nonsupp_indx = pickle.load(f)
    # otherwise use the random initialized indices
    else:
        supp_indx = indices[:num_supp_init]
        nonsupp_indx = indices[num_supp_init:]

    supp_sampler = SubsetRandomSampler(supp_indx) 
    supp_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, sampler=supp_sampler, shuffle=False, **loader_kargs)
    #NEW
    nonsupp = Subset(train,nonsupp_indx)
    nonsupp_loader = torch.utils.data.DataLoader(nonsupp, batch_size=len(nonsupp_indx), shuffle=False, **loader_kargs)
    
    #ORIG
    # nonsupp_sampler = SubsetRandomSampler(nonsupp_indx)
    # nonsupp_loader = torch.utils.data.DataLoader(train, batch_size=len(nonsupp_indx), sampler=nonsupp_sampler, shuffle=False, **loader_kargs)
    #nonsupp_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, sampler=nonsupp_sampler, shuffle=False)

    # all_train_sampler = SubsetRandomSampler(indices)
    train_loader = torch.utils.data.DataLoader(train, batch_size=n_train, shuffle=True, **loader_kargs)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True, **loader_kargs)

    return supp_loader, nonsupp_loader, supp_indx, nonsupp_indx, train_loader, test_loader



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
        train = BreastMNIST(root=input_root, split='train', transform=data_transform, download=download)
        test = BreastMNIST(root=input_root, split='test', transform=data_transform, download=download)


    return train, test