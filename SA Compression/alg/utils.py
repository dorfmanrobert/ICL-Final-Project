import torch
import pickle
from alg.models import testNNet
from alg.SAalg import SAbound, SAalg, changeComp
from alg.data import loaddataset, loadbatches

def SArunexp(g, C, name_data='binarymnist', delta=.025, learning_rate=.001, momentum=.9, batch_size=100, train_epochs=100, dropout_prob=.2, perc_train=1, num_supp_init=5, device='cuda', verbose=True, continuing=False):

    # if continuing == True:
    #     with open('C.pkl', 'rb') as f:
    #         C = pickle.load(f)
            
    loader_kargs = {'num_workers': 1,
                    'pin_memory': True,
                   'generator':g} if torch.cuda.is_available() else {}

    # load data
    train, test = loaddataset(name_data)
    
    # initialize support and nonsupport indices
    supp_loader, nonsupp_loader, supp_indx, nonsupp_indx, train_loader, test_loader = loadbatches(train, test, loader_kargs, batch_size, perc_train=perc_train, num_supp_init=num_supp_init, continuing=continuing)

    # run SA alg, additing worst points to support set
    net, nets, max_nonsupp_losses, supp_indx, nonsupp_indx, supp_loader, nonsupp_loader, C = SAalg(C, train, supp_loader, nonsupp_loader, supp_indx, nonsupp_indx, train_loader, test_loader, loader_kargs, learning_rate=learning_rate, momentum=momentum, batch_size=batch_size, train_epochs=train_epochs, dropout_prob=dropout_prob, continuing=continuing, device=device, verbose=verbose)
    print("***SA algorithm complete***")

    # compute generalization error bound for SA alg
    supp_num = len(supp_indx)
    nonsupp_num = len(nonsupp_indx)
    total_num = supp_num + nonsupp_num
    
    sa_ub = SAbound(supp_num, total_num, delta)
    p_misclass = testNNet(net, test_loader)
    p_change_comp = changeComp(C, nets, max_nonsupp_losses, len(test_loader.dataset), test_loader, device=device)
    
    # # Save results
    # results = [sa_ub, p_misclass, p_change_comp, C, supp_num, nonsupp_num]
    # with open('SAbound-pmisclass-pchangecompr-eps-supp-nonsupp.pkl', 'wb') as f:
    #     pickle.dump(results, f)
    
    return sa_ub, p_misclass, p_change_comp, C, supp_num, nonsupp_num
  
    
