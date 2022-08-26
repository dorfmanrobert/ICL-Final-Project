import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange
import random

from sgd.networks import NNet4l, ConvNet2l, trainNNet, testNNet
from sgd.bounds import binomial, chernoff

from sgd.data import loadbatches, loaddataset
import pickle


def runexp(learning_rate, momentum, batch_size=250, delta=0.025, layers=9, train_epochs=100,
verbose=True, device='cuda', dropout_prob=0.2, perc_train=1.0, perc_val=.14, verbose_test=True, model_type='fcn', name_data='binarymnist', pmin=1e-4):
    """Run an experiment with PAC-Bayes inspired training objectives

    Parameters
    ----------
    name_data : string
        name of the dataset to use (check data file for more info)

    objective : string
        training objective to use

    model : string
        could be cnn or fcn
    
    pmin : float
        minimum probability to clamp the output of the cross entropy loss
    
    learning_rate : float
        learning rate hyperparameter used for the optimiser

    momentum : float
        momentum hyperparameter used for the optimiser

    delta : float
        confidence parameter for the risk certificate
    
    delta_test : float
        confidence parameter for chernoff bound


    train_epochs : int
        numer of training epochs for training

    verbose : bool
        whether to print metrics during training

    device : string
        device the code will run in (e.g. 'cuda')

    dropout_prob : float
        probability of an element to be zeroed.

    perc_train : float
        percentage of train data to use for the entire experiment (can be used to run
        experiments with reduced datasets to test small data scenarios)
    
    verbose_test : bool
        whether to print test and risk certificate stats during training epochs

    batch_size : int
        batch size for experiments
    """
 


    loader_kargs = {'num_workers': 1,
                    'pin_memory': True} if torch.cuda.is_available() else {}



    train, test = loaddataset(name_data)
    
        
    with open('train.pkl', 'wb') as f:
        pickle.dump(train, f)

    # initialize NN 
    if model_type=='fcn':
        net = NNet4l(dropout_prob=dropout_prob, device=device).to(device)
    elif model_type == 'cnn':
        net = ConvNet2l(dropout_prob=dropout_prob, device=device).to(device)

    # load data
    train_loader, val_loader, test_loader = loadbatches(train, test, loader_kargs, batch_size, perc_train=perc_train, perc_val=perc_val)

    # train
    #wandb.watch(net)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in trange(train_epochs):
        trainNNet(net, optimizer, epoch, train_loader, device=device, verbose=verbose)
    
    with open('net.pkl', 'wb') as f:
        pickle.dump(net,f)
    with open('test.pkl', 'wb') as f:
        pickle.dump(test,f)
    print("****Training complete****")

    # compute validation and test error 
    val_err = testNNet(net, val_loader, device=device, verbose=False)
    test_err = testNNet(net, test_loader, device=device, verbose=False)

    # compute risk certificates
    chernoff_bound = chernoff(net, val_loader, delta, device=device)
    binomial_bound = binomial(net, val_loader, delta, device=device)

    print(f"***Final results***") 
    print(f"validation error {val_err}")
    print(f"chernoff bound {chernoff_bound}")
    print(f"binomial bound {binomial_bound}")
    print(f"test error {test_err}")

    return binomial_bound, chernoff_bound, test_err