import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset
from torchvision import datasets, transforms
import numpy as np
import medmnist
from medmnist.dataset import BreastMNIST
import pickle
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

        # Load MedMNIST data
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

    nonsupp = Subset(train,nonsupp_indx)
    nonsupp_loader = torch.utils.data.DataLoader(nonsupp, batch_size=len(nonsupp_indx), shuffle=False, **loader_kargs)
 
    # all_train_sampler = SubsetRandomSampler(indices)
    train_loader = torch.utils.data.DataLoader(train, batch_size=n_train, shuffle=True, **loader_kargs)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True, **loader_kargs)

    return supp_loader, nonsupp_loader, supp_indx, nonsupp_indx, train_loader, test_loader

