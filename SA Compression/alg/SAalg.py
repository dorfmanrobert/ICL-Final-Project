import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset
import numpy as np
from math import comb
import sympy
from sympy import *
from tqdm import tqdm, trange
import pickle
from alg.models import NNet4l, trainNNet, testNNet

def SAbound(k, N, bet):
    m = np.linspace(k, N, num=N-1-k+1) 
    aux1 = np.sum(np.triu(np.log(np.ones((N-1-k+1,1))*m), 1), 1)
    aux2 = np.sum(np.triu(np.log(np.ones((N-1-k+1,1))*(m-k)), 1), 1)
    aux3 = np.sum(np.triu(np.log(np.ones((N-1-k+1,1))*N), 1), 1)
    aux4 = np.sum(np.triu(np.log(np.ones((N-1-k+1,1))*(N-k)), 1), 1)
    coeffs1 = aux2 - aux1
    coeffs2 = aux4 - aux3
    coeffs = coeffs1 - coeffs2
    t1 = 0
    t2 = 1 
    while t2 - t1 > 1e-10:
        t = (t1 + t2) / 2
        val = 1 - (bet/(N)*np.sum(np.exp(coeffs - (N - np.squeeze(np.matrix(m).H))*np.log(t))))
        if val > 0:
            t2 = t
        else:
            t1 = t
    eps = 1 - t1
    return eps
    

def find_max_loss(net, test_loader, device='cuda'):
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            losses = F.nll_loss(outputs, target, reduction='none')
    max_loss_index = torch.argmax(losses)
    max_loss = torch.max(losses)
    return max_loss_index.item(), max_loss.item()


def check_condition(C, net, test_loader, device='cuda'):
    # with torch.no_grad():
    #     for data, target in test_loader:
    #         data, target = data.to(device), target.to(device)
    #         outputs = net(data)
    #         losses = F.nll_loss(outputs, target, reduction='none')
    max_loss_indx, max_loss = find_max_loss(net, test_loader)
    print(f"previous max loss{C}")
    print(f"new max loss {max_loss}")
    print(max_loss <= C)
    if max_loss <= C:
        print("TERMINATING!!")
        return True
    else:
        return False
    
    
def changeComp(C, nets, max_nonsupp_losses, n_test, test_loader_comp, device='cuda'):
    # compute p change compression
    # pass one datapoint at a time
    print(len(test_loader_comp.dataset))
    num_misclass = 0
    for data, target in test_loader_comp:
        data, target = data.to(device), target.to(device)
        
        i = 0
        # check if is the max loss for any net
        for i in range(len(nets)):
            # for all but the last net check if new point has max loss 
            if i < len(nets) - 1:
                net = nets[i]
                output = net(data)
                loss = F.nll_loss(output, target)
                max_nonsupp_loss = max_nonsupp_losses[i]
                if loss > max_nonsupp_loss:
                    num_misclass = num_misclass + 1
                    break
            else:
                # for last net check if greater than epsilon
                last_net = nets[len(nets)-1]
                output = last_net(data)
                loss = F.nll_loss(output, target)
                if loss > C:
                    num_misclass = num_misclass + 1
        
    p_change_comp = num_misclass / n_test
   
    return p_change_comp
    


def SAalg(C, train, supp_loader, nonsupp_loader, supp_indx, nonsupp_indx, train_loader, test_loader, loader_kargs, learning_rate=.001, momentum=.9, batch_size=250, train_epochs=100, dropout_prob=.2, device='cuda', verbose=False, continuing=True):


    # run SA alg, additing worst points to support set
    counter = 0
    net = NNet4l(dropout_prob=dropout_prob, device=device).to(device)
    
       
    if continuing == True:
        with open('condition.pkl', 'rb') as f:
            condition = pickle.load(f)
        with open('net.pkl', 'rb') as f:
            net = pickle.load(f)
        with open('nets.pkl', 'rb') as f:
            nets = pickle.load(f)
        print(f"nets{len(nets)}")
        with open('maxnonsupplosses.pkl', 'rb') as f:
            max_nonsupp_losses = pickle.load(f)
        print(f"max_nonsupp_losses{len(max_nonsupp_losses)}")
    else:
        condition = False
        net = NNet4l(dropout_prob=dropout_prob, device=device).to(device)
        nets = [net]
        max_nonsupp_losses= []

    while condition == False:
        
        counter = counter + 1
        # remove max value found from not supp set and add to supp set
        max_loss_indx,max_loss = find_max_loss(net, nonsupp_loader, device=device)  # finds index of nonsupp that has max value
        
        max_nonsupp_losses.append(max_loss)
        with open('maxnonsupplosses.pkl', 'wb') as f:
            pickle.dump(max_nonsupp_losses, f)
            
            
        change_indx = nonsupp_indx[max_loss_indx]
        supp_indx.append(change_indx)
        nonsupp_indx.remove(change_indx)

        # create new data loaders for next NN training
        supp_sampler = SubsetRandomSampler(supp_indx) 
        supp_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, sampler=supp_sampler, shuffle=False, **loader_kargs)
        
        # # ORIGINAL
        # nonsupp_sampler = SubsetRandomSampler(nonsupp_indx)
        # nonsupp_loader = torch.utils.data.DataLoader(train, batch_size=len(nonsupp_indx), sampler=nonsupp_sampler, shuffle=False, **loader_kargs)
        
        #NEW
        nonsupp = Subset(train,nonsupp_indx)
        nonsupp_loader = torch.utils.data.DataLoader(nonsupp, batch_size=len(nonsupp_indx), shuffle=False, **loader_kargs)
    
        # pickle indices to files
        with open('supp_indx.pkl', 'wb') as f:
            pickle.dump(supp_indx, f)
        with open('nonsupp_indx.pkl', 'wb') as f:
            pickle.dump(nonsupp_indx, f)
        
        # run full training on support set
        net = NNet4l(dropout_prob=dropout_prob, device=device).to(device)
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
        for epoch in trange(train_epochs):
            trainNNet(net, optimizer, epoch, supp_loader, device=device, verbose=verbose)
        
             
        with open('net.pkl', 'wb') as f:
            pickle.dump(net, f)
                        
        nets.append(net)
        with open('nets.pkl', 'wb') as f:
            pickle.dump(nets, f)
   

        # update C 
        # _, max_loss = find_max_loss(net, supp_loader, device=device)
        # C = max_loss
        # with open('C.pkl', 'wb') as f:
        #     pickle.dump(C, f)

        # check if condition satisfied
        condition = check_condition(C, net, nonsupp_loader)

        # check if moved all n datapoints, then break 
        if len(supp_indx) == len(supp_indx) + len(nonsupp_indx):
            condition = True
            
        with open('condition.pkl', 'wb') as f:
            pickle.dump(condition,f)

        if verbose and counter % 20 == 0:
            print(f"***CHECKPOINT***")
            print(f"supp indx length {len(supp_indx)}")
            print(f"nonsupp indx length {len(nonsupp_indx)}")
            print(f"epsilon{C}")


    
    return net, nets, max_nonsupp_losses, supp_indx, nonsupp_indx, supp_loader, nonsupp_loader, C