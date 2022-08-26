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


from pnn.networks import NNet4l, ProbNNet4l, trainNNet, testNNet,trainPNNet, computeRiskCertificates, testStochastic
from pnn.bounds import PBBobj

from pnn.data import loadbatches, loaddataset




def trainprior(train_loader_1, train_loader_2, bound_n_size=30000, delta=.025, delta_test=.01, mc_samples=10000, kl_penalty=1, classes=2, train_method='original', rho_prior=.03, learning_rate_prior=.001, momentum_prior=.95, prior_epochs=100, prior_train_opt='det', prior_train_method='one', prior_dist='gaussian', objective='invkl', dropout_prob=.2, verbose=True, device='cuda'):
    # Initialize NN for prior
    if prior_train_opt == 'det':
        net01 = NNet4l(dropout_prob=dropout_prob, device=device).to(device)
        if prior_train_method == 'two':
            net02 = NNet4l(dropout_prob=dropout_prob, device=device).to(device)
        elif prior_train_method == 'one':
            net02 = None

    elif prior_train_opt == 'pb':
        net01 = ProbNNet4l(rho_prior, prior_dist=prior_dist, device=device, train_method=train_method).to(device)
        if prior_train_method == 'two':
            net02 = ProbNNet4l(rho_prior, prior_dist=prior_dist, device=device, train_method=train_method).to(device)
        elif prior_train_method=='one':
            net02 = None
        bound0 = PBBobj(classes=classes, delta=delta, delta_test=delta_test, mc_samples=mc_samples, kl_penalty=kl_penalty, device=device, train_n=bound_n_size, bound_n=bound_n_size, objective=objective, prior_dist=prior_dist, train_method=train_method, prior_train_method=prior_train_method)

    # Train prior
    optimizer_01 = optim.SGD(net01.parameters(), lr=learning_rate_prior, momentum=momentum_prior)
    if prior_train_opt == 'det':
        if prior_train_method == 'two':
            optimizer_02 = optim.SGD(net02.parameters(), lr=learning_rate_prior, momentum=momentum_prior)
        for epoch in trange(prior_epochs):
            trainNNet(net01, optimizer_01, epoch, train_loader_1, device=device, verbose=verbose)
            if prior_train_method == 'two':
                trainNNet(net02, optimizer_02, epoch, train_loader_2, device=device, verbose=verbose)

    elif prior_train_opt == 'pb':
        if prior_train_method == 'two':
            optimizer_02 = optim.SGD(net02.parameters(), lr=learning_rate_prior, momentum=momentum_prior)
        for epoch in trange(prior_epochs):
            trainPNNet(net01, optimizer_01, bound0, epoch, train_loader_1, verbose=verbose)
            if prior_train_method == 'two':
                trainPNNet(net02, optimizer_02, bound0, epoch, train_loader_1, verbose=verbose)

    return net01, net02


def trainposterior(net01, net02, train_loader_1batch, train_loader_1, train_loader_2, set_bound_1batch_1, set_bound_1batch_2, train_loader, test_loader, bound_n_size=30000, posterior_n_size=30000, delta=.025, delta_test=.01, mc_samples=10000, kl_penalty=1, classes=2, train_method='original', rho_prior=.03, learning_rate=.001, momentum=.95, train_epochs=100, prior_train_opt='det', prior_train_method='one', prior_dist='gaussian', objective='invkl', verbose=True, verbose_test=False, device='cuda', toolarge=False):
    # Initialize posterior PNN
    net_1 = ProbNNet4l(rho_prior, prior_dist=prior_dist, device=device, init_net=net01, train_method=train_method, prior_train_opt=prior_train_opt).to(device)
    if prior_train_method=='two':
        net_2 = ProbNNet4l(rho_prior, prior_dist=prior_dist, device=device, init_net=net02, train_method=train_method, prior_train_opt=prior_train_opt).to(device)
    elif prior_train_method=='one':
        net_2 = None

    bound = PBBobj(classes=classes, delta=delta, delta_test=delta_test, mc_samples=mc_samples, kl_penalty=kl_penalty, device=device, train_n=posterior_n_size, bound_n=bound_n_size, objective=objective, prior_dist=prior_dist, train_method=train_method, prior_train_method=prior_train_method)
    
   # Run training of posterior
    optimizer_1 = optim.SGD(net_1.parameters(), lr=learning_rate, momentum=momentum)
    if prior_train_method == 'two':
        optimizer_2 = optim.SGD(net_2.parameters(), lr=learning_rate, momentum=momentum)
    
    for epoch in trange(train_epochs):
        if prior_train_method == 'two': 
            # Train on flipped halves the priors were trained on 
            trainPNNet(net_1, optimizer_1, bound, epoch, train_loader_2, verbose)
            trainPNNet(net_2, optimizer_2, bound, epoch, train_loader_1, verbose)
        elif prior_train_method == 'one':
            # Train on whole dataset
            trainPNNet(net_1, optimizer_1, bound, epoch, train_loader, verbose)
            
        if verbose_test and epoch % 20 == 0: 
            if prior_train_method=="two":
                avg_net_1 = avgnets(net01, net_1, net_2, rho_prior).to(device)
                avg_net_2 = avgnets(net02, net_1, net_2, rho_prior).to(device)
            elif prior_train_method=="one":
                avg_net_1 = None
                avg_net_2 = None
                
#             train_obj_1, train_obj_2, ub_risk_01, kl, err_01_train = computeRiskCertificates(avg_net_1, avg_net_2, net_1, net_2, bound, toolarge=toolarge, device=device, train_loader_1=train_loader_1, train_loader_2=train_loader_2, set_bound_1batch_1=set_bound_1batch_1, set_bound_1batch_2=set_bound_1batch_2)
            ub_risk_01, kl, err_01_train = computeRiskCertificates(avg_net_1, avg_net_2, net_1, net_2, bound, toolarge=toolarge, device=device, train_loader_1=train_loader_1, train_loader_2=train_loader_2, set_bound_1batch_1=set_bound_1batch_1, set_bound_1batch_2=set_bound_1batch_2)
            
            stch_err_1 = testStochastic(net_1, test_loader, bound, device=device)
            if prior_train_method=='two':
                # just average the stochastic test errors for now may NEED TO AVERAGE NETWORKS BEFPRE EVALUATION???
                stch_err_2 = testStochastic(net_2, test_loader, bound, device=device)
                stch_err = (stch_err_1 + stch_err_2) / 2
            elif prior_train_method == 'one':
                stch_err = stch_err_1

            print(f"***Checkpoint results***")         
            print(f"ub_risk_01, kl, loss_01 train, stch_test_error")
            print(f" {ub_risk_01 :.5f}, {kl :.5f}, {err_01_train :.5f}, {stch_err :.5f}")

    if prior_train_method=="two":
            avg_net_1 = avgnets(net01, net_1, net_2, rho_prior).to(device)
            avg_net_2 = avgnets(net02, net_1, net_2, rho_prior).to(device)
    elif prior_train_method=="one":
            avg_net_1 = None
            avg_net_2 = None
            
    # train_obj_1, train_obj_2, ub_risk_01, kl, err_01_train = computeRiskCertificates(avg_net_1, avg_net_2, net_1, net_2, bound, toolarge=toolarge, device=device, train_loader_1=train_loader_1, train_loader_2=train_loader_2, set_bound_1batch_1=set_bound_1batch_1, set_bound_1batch_2=set_bound_1batch_2)
    ub_risk_01, kl, err_01_train = computeRiskCertificates(avg_net_1, avg_net_2, net_1, net_2, bound, toolarge=toolarge, device=device, train_loader_1=train_loader_1, train_loader_2=train_loader_2, set_bound_1batch_1=set_bound_1batch_1, set_bound_1batch_2=set_bound_1batch_2)

    if prior_train_method=='two':
        # just average the stochastic test errors for now may NEED TO AVERAGE NETWORKS BEFPRE EVALUATION???
        # stch_err_1 = testStochastic(net_1, test_loader, bound, device=device)
        # stch_err_2 = testStochastic(net_2, test_loader, bound, device=device)
        # stch_err = (stch_err_1 + stch_err_2) / 2  # test set error

        stch_err = testStochastic(avg_net_1, test_loader, bound, device=device)

    elif prior_train_method == 'one':
        stch_err = testStochastic(net_1, test_loader, bound, device=device)

    if verbose:
        print(f"***Final results***") 
        print(f" ub_risk_01{ub_risk_01 :.5f}, kl{kl :.5f}, loss_01_train{err_01_train :.5f}, test_error{stch_err :.5f}")

    return ub_risk_01, stch_err


def runexp(sigma_prior, learning_rate, momentum, objective='quad',
learning_rate_prior=0.01, momentum_prior=0.95, delta=0.025, layers=9, delta_test=0.01, mc_samples=1000, kl_penalty=1, train_epochs=100, prior_dist='gaussian', 
verbose=True, device='cuda', prior_epochs=1, dropout_prob=0.2, perc_train=1.0, verbose_test=True, 
perc_prior=0.5, batch_size=250, train_method='original', model_type='fcn', prior_train_method='one', prior_train_opt='det', name_data='binarymnist'):
    """Run an experiment with PAC-Bayes inspired training objectives

    Parameters
    ----------
    name_data : string
        name of the dataset to use (check data file for more info)

    objective : string
        training objective to use
    
    model : string
        could be cnn or fcn
    
    sigma_prior : float
        scale hyperparameter for the prior
    
    pmin : float
        minimum probability to clamp the output of the cross entropy loss
    
    learning_rate : float
        learning rate hyperparameter used for the optimiser

    momentum : float
        momentum hyperparameter used for the optimiser

    learning_rate_prior : float
        learning rate used in the optimiser for learning the prior (only
        applicable if prior is learnt)

    momentum_prior : float
        momentum used in the optimiser for learning the prior (only
        applicable if prior is learnt)
    
    delta : float
        confidence parameter for the risk certificate
    
    layers : int
        integer indicating the number of layers (applicable for CIFAR-10, 
        to choose between 9, 13 and 15)
    
    delta_test : float
        confidence parameter for chernoff bound

    mc_samples : int
        number of monte carlo samples for estimating the risk certificate
        (set to 1000 by default as it is more computationally efficient, 
        although larger values lead to tighter risk certificates)

    kl_penalty : float
        penalty for the kl coefficient in the training objective

    train_epochs : int
        numer of training epochs for training

    prior_dist : string
        type of prior and posterior distribution (can be gaussian or laplace)

    verbose : bool
        whether to print metrics during training

    device : string
        device the code will run in (e.g. 'cuda')

    prior_epochs : int
        number of epochs used for learning the prior (not applicable if prior is rand)

    dropout_prob : float
        probability of an element to be zeroed.

    perc_train : float
        percentage of train data to use for the entire experiment (can be used to run
        experiments with reduced datasets to test small data scenarios)
    
    verbose_test : bool
        whether to print test and risk certificate stats during training epochs

    perc_prior : float
        percentage of data to be used to learn the prior

    batch_size : int
        batch size for experiments
    """

    # this makes the initialised prior the same for all bounds
    torch.manual_seed(10)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loader_kargs = {'num_workers': 1,
                    'pin_memory': True} if torch.cuda.is_available() else {}
    
    train, test = loaddataset(name_data)
    rho_prior = math.log(math.exp(sigma_prior)-1.0)

    # Load data
    train_loader_1batch, train_loader_1, train_loader_2, set_bound_1batch_1, set_bound_1batch_2, train_loader, test_loader = loadbatches(train, test, loader_kargs, batch_size, perc_train=perc_train, perc_prior=perc_prior)

    # Sizes of data subsets
    if prior_train_method == 'two':
        posterior_n_size = len(train_loader.dataset) * perc_prior 
        bound_n_size = len(train_loader.dataset) * perc_prior 
    else:
        posterior_n_size = len(train_loader.dataset)                  # number of data points used to train the posterior (i.e. all training data)
        bound_n_size = len(train_loader.dataset) * perc_prior         # number of data points used to compute the risk certificate (i.e. all train - num used in prior)
    
    toolarge = False

    # Run prior training
    net01, net02 = trainprior(train_loader_1, train_loader_2, bound_n_size=bound_n_size, delta=delta, delta_test=delta_test, mc_samples=mc_samples, kl_penalty=kl_penalty, train_method=train_method, rho_prior=rho_prior, learning_rate_prior=learning_rate_prior, momentum_prior=momentum_prior, prior_epochs=prior_epochs, prior_train_opt=prior_train_opt, prior_train_method=prior_train_method, prior_dist=prior_dist, objective=objective, dropout_prob=dropout_prob, verbose=verbose, device=device)
    
    # Run posterior training
    ub_risk_01, stch_err = trainposterior(net01, net02, train_loader_1batch, train_loader_1, train_loader_2, set_bound_1batch_1, set_bound_1batch_2, train_loader, test_loader, bound_n_size=bound_n_size, posterior_n_size=posterior_n_size, delta=delta, delta_test=delta_test, mc_samples=mc_samples, kl_penalty=kl_penalty, train_method=train_method, rho_prior=rho_prior, learning_rate=learning_rate, momentum=momentum, train_epochs=train_epochs, prior_train_opt=prior_train_opt, prior_train_method=prior_train_method, prior_dist=prior_dist, objective=objective, verbose=verbose, verbose_test=verbose_test, device=device, toolarge=toolarge)

    return ub_risk_01, stch_err

def avgnets(prior_net, net1, net2, rho_prior, model_type='fcn'):
    if model_type == 'fcn':
        avgnet = ProbNNet4l(rho_prior, init_net=prior_net)
        # Need to set rho not sigma, convert 
        # sigma = log(exp(rho)+1)
        # exp(sigma) = exp(rho) + 1
        # log(exp(sigma) - 1) = rho
    elif model_type == 'cnn':
        avgnet = ProbConvNet2l(rho_prior, init_net=prior_net)

    if net2 != none:
        avgnet.l1.weight.mu = nn.Parameter((net1.l1.weight.mu + net2.l1.weight.mu) / 2)
        avgnet.l1.weight.rho = nn.Parameter(torch.log(torch.exp(torch.sqrt((net1.l1.weight.sigma**2 + net2.l1.weight.sigma**2) / 4)) - 1))
        avgnet.l1.bias.mu = nn.Parameter((net1.l1.bias.mu + net2.l1.bias.mu) / 2)
        avgnet.l1.bias.rho = nn.Parameter(torch.log(torch.exp(torch.sqrt((net1.l1.bias.sigma**2 + net2.l1.bias.sigma**2) / 4)) - 1))

        avgnet.l2.weight.mu = nn.Parameter((net1.l2.weight.mu + net2.l2.weight.mu) / 2)
        avgnet.l2.weight.rho = nn.Parameter(torch.log(torch.exp(torch.sqrt((net1.l2.weight.sigma**2 + net2.l2.weight.sigma**2) / 4)) - 1))
        avgnet.l2.bias.mu = nn.Parameter((net1.l2.bias.mu + net2.l2.bias.mu) / 2)
        avgnet.l2.bias.rho = nn.Parameter(torch.log(torch.exp(torch.sqrt((net1.l2.bias.sigma**2 + net2.l2.bias.sigma**2) / 4)) - 1))

        avgnet.l3.weight.mu = nn.Parameter((net1.l3.weight.mu + net2.l3.weight.mu) / 2)
        avgnet.l3.weight.rho = nn.Parameter(torch.log(torch.exp(torch.sqrt((net1.l3.weight.sigma**2 + net2.l3.weight.sigma**2) / 4)) - 1))
        avgnet.l3.bias.mu = nn.Parameter((net1.l3.bias.mu + net2.l3.bias.mu) / 2)
        avgnet.l3.bias.rho = nn.Parameter(torch.log(torch.exp(torch.sqrt((net1.l3.bias.sigma**2 + net2.l3.bias.sigma**2) / 4)) - 1))

        avgnet.l4.weight.mu = nn.Parameter((net1.l4.weight.mu + net2.l4.weight.mu) / 2)
        avgnet.l4.weight.rho = nn.Parameter(torch.log(torch.exp(torch.sqrt((net1.l4.weight.sigma**2 + net2.l4.weight.sigma**2) / 4)) - 1))
        avgnet.l4.bias.mu = nn.Parameter((net1.l4.bias.mu + net2.l4.bias.mu) / 2)
        avgnet.l4.bias.rho = nn.Parameter(torch.log(torch.exp(torch.sqrt((net1.l4.bias.sigma**2 + net2.l4.bias.sigma**2) / 4)) - 1))
        return avgnet
    
    else:
        return None