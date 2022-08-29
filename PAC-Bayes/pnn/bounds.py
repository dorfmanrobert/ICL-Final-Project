import math
import numpy as np
import torch
import torch.distributions as td
from tqdm import tqdm, trange
import torch.nn.functional as F
from scipy.special import xlogy


# Inverted binary KL
# For scalars
def kl_bin(q, p):
    eps = 1e-6
    # make sure q, p in (0,1)
    q = sorted([eps, q, 1 - eps])[1]
    p = sorted([eps, p, 1 - eps])[1]
    kl = xlogy(q, q/p) + xlogy(1-q, (1-q)/(1-p))
    return sorted([0, kl, 1e6])[1]

def inv_kl(q, c, iter=15):
    # performs Newton's method with iter number of iterations
    eps = 1e-6
    # intialize estimate with pinsker ubber bound
    sup = q + np.sqrt(c/2)
    # perform Newton's method
    for i in range(iter):
        if sup >= 1:
          sup = 1 - eps
        h = kl_bin(q, sup) - c
        h_deriv = (1-q)/(1-sup) - q/sup
        sup = sup - h / h_deriv
    return min(sup, 1)

# For torch tensors
def kl_bin_torch(q, p):
    eps = 1e-6
    q = torch.clamp(q, min=eps, max=1-eps)
    p = torch.clamp(p, min=eps, max=1-eps)
    kl = torch.xlogy(q, q/p) + torch.xlogy(1-q, (1-q)/(1-p))
    return torch.clamp(kl, min=0, max=1e6)

def inv_kl_torch(q, c, iter=15, device='cuda', **kwargs):  # orig is 10 iterations
    eps = 1e-6
    sup = q + torch.sqrt(c/2)
    for i in range(iter):
        if torch.max(sup) >= 1:
            sup = torch.clamp(sup, 1 - eps)
        h = kl_bin_torch(q, sup) - c
        h_deriv = (1-q)/(1-sup) - q/sup
        sup = sup - h / h_deriv
    return torch.minimum(sup, torch.Tensor([1]).to(device))


# InvKL class is from https://github.com/eclerico/ CondGauss.
class InvKL(torch.autograd.Function):
  
  @staticmethod
  def forward(ctx, q, c):
    #out = inv_kl_clerico(q, c)
    out = inv_kl_torch(q, c)
    ctx.save_for_backward(q, out)
    return out
  
  @staticmethod
  def backward(ctx, grad_output):
    eps = 1e-6    #torch.finfo(torch.float32).eps
    q, out = ctx.saved_tensors
    grad_q = grad_c = None
    den = (1-q)/(1-out) - q/out
    den[den==0] = eps
    sign = den.sign()
    den = den.abs_().clamp_(min=eps)
    den *= sign
    grad_c = grad_output / den
    grad_q = (torch.log(torch.clamp((1-q)/(1-out), min=eps)) - torch.log(torch.clamp(q/out, min=eps))) * grad_c * grad_output
    return grad_q, grad_c, None #last None is for iter...

invkl = InvKL.apply


class PBBobj():
    """Class including all functionalities needed to train a NN with a PAC-Bayes inspired 
    training objective and evaluate the risk certificate at the end of training. 

    Parameters
    ----------
    objective : string
        training objective to be optimised (choices are fquad, flamb, fclassic or fbbb)
    
    pmin : float
        minimum probability to clamp to have a loss in [0,1]

    classes : int
        number of classes in the learning problem
    
    train_size : int
        n (number of training examples)

    delta : float
        confidence value for the training objective
    
    delta_test : float
        confidence value for the chernoff bound (used when computing the risk)

    mc_samples : int
        number of Monte Carlo samples when estimating the risk

    kl_penalty : float
        penalty for the kl coefficient in the training objective
    
    device : string
        Device the code will run in (e.g. 'cuda')

    """
    def __init__(self, pmin=1e-6, classes=2, delta=0.025,
    delta_test=0.01, mc_samples=1000, kl_penalty=1, device='cuda', train_n = 30000, bound_n=30000, objective='quad', prior_dist = 'gaussian', train_method='original', prior_train_method='one'):
        super().__init__()
        self.objective = objective
        self.pmin = pmin
        self.classes = classes
        self.device = device

        if prior_train_method == 'one':
            self.delta = delta
            self.delta_test = delta_test
        elif prior_train_method == 'two':
            self.delta = delta / 2
            self.delta_test = delta_test/2

        # self.delta_test = delta_test
        self.mc_samples = mc_samples
        self.kl_penalty = kl_penalty
        self.n_posterior = train_n
        self.n_bound = bound_n
        self.train_method = train_method
        self.prior_train_method = prior_train_method


    def compute_empirical_risk(self, outputs, targets, bounded=True):
        # compute negative log likelihood loss and bound it with pmin (if applicable)
        empirical_risk = F.nll_loss(outputs, targets)
        if bounded == True:
            empirical_risk = (1./(np.log(1./self.pmin))) * empirical_risk
        return empirical_risk

    def compute_losses(self, net, data, target, clamping=True):
        # compute both cross entropy and 01 loss
        # returns outputs of the network as well
        outputs = net(data, sample=True, clamping=True, pmin=self.pmin)
        loss_ce = self.compute_empirical_risk(outputs, target, clamping)
        pred = outputs.max(1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum().item()
        total = target.size(0)
        loss_01 = 1-(correct/total)
        return loss_ce, loss_01, outputs

    def bound(self, empirical_risk, kl, train_size):
        # compute training objectives
        if self.objective == 'quad':
            kl = kl * self.kl_penalty
            repeated_kl_ratio = torch.div(kl + np.log((2*np.sqrt(train_size))/self.delta), 2*train_size)
            first_term = torch.sqrt(empirical_risk + repeated_kl_ratio)
            second_term = torch.sqrt(repeated_kl_ratio)
            train_obj = torch.pow(first_term + second_term, 2)
        elif self.objective == 'invkl':
            kl = kl * self.kl_penalty
            kl_term = torch.add(kl, np.log((2*np.sqrt(train_size)) / self.delta))
            train_obj = invkl(empirical_risk, torch.div(kl_term, train_size))
        else:
            raise RuntimeError(f'Wrong objective {self.objective}')
        return train_obj

    def mcsampling(self, net, input, target, batches=True, data_loader=None, clamping=True):
        # compute empirical risk with Monte Carlo sampling
        error = 0.0
        cross_entropy = 0.0
        if batches:
            for batch_id, (data_batch, target_batch) in enumerate(tqdm(data_loader)):
                data_batch, target_batch = data_batch.to(
                    self.device), target_batch.to(self.device)
                cross_entropy_mc = 0.0
                error_mc = 0.0
                for i in range(self.mc_samples):
                    loss_ce, loss_01, _ = self.compute_losses(net, data_batch, target_batch, clamping)
                    cross_entropy_mc += loss_ce
                    error_mc += loss_01
                # we average cross-entropy and 0-1 error over all MC samples
                cross_entropy += cross_entropy_mc/self.mc_samples
                error += error_mc/self.mc_samples
            # we average cross-entropy and 0-1 error over all batches
            cross_entropy /= batch_id
            error /= batch_id
        else:
            cross_entropy_mc = 0.0
            error_mc = 0.0
            for i in range(self.mc_samples):
                loss_ce, loss_01, _ = self.compute_losses(net, input, target, clamping)
                cross_entropy_mc += loss_ce
                error_mc += loss_01
                # we average cross-entropy and 0-1 error over all MC samples
            cross_entropy += cross_entropy_mc/self.mc_samples
            error += error_mc/self.mc_samples
        return cross_entropy, error

    def train_obj(self, net, input, target, clamping=True):
        # compute train objective and return all metrics
        outputs = torch.zeros(target.size(0), self.classes).to(self.device)
        kl = net.compute_kl()
        loss_ce, loss_01, outputs = self.compute_losses(net, input, target, clamping)

        if self.train_method == 'original':
            train_obj = self.bound(loss_ce, kl, self.n_posterior)

        # code for CondGauss adapted from https://github.com/eclerico/ CondGauss
        elif self.train_method == 'conditional':
            # 2) compute closed form conditional expected 0-1 loss
            # sample from last layer and compute output of hidden layers post activation
            X = net.hidden_out(input)
            last_weight_mean = net.l4.weight.mu
            last_bias_mean = net.l4.bias.mu
            # compute conditional mean of output
            M = F.linear(X, last_weight_mean, last_bias_mean)
            # compute conditional variance of output
            last_weight_std = net.l4.weight.sigma
            last_bias_std = net.l4.bias.sigma if net.l4.bias.rho is not None else None
            V = F.linear(X**2, torch.square(last_weight_std), torch.square(last_bias_std))
            # compute unbiased estimated of gaussian 0-1 loss
            Z = (M[..., 0] - M[..., 1])/torch.clamp(torch.sqrt(V.sum(-1)), min=1e-6)
            loss_gauss_zero = (1-torch.erf(Z/2**.5))/2
            loss_gauss_one = (1-torch.erf(-Z/2**.5))/2
            empirical_risk = ((1-target)*loss_gauss_zero + target*loss_gauss_one).mean()  
            train_obj = self.bound(empirical_risk, kl, self.n_posterior)

        return train_obj, kl/self.n_posterior, outputs, loss_01

    
    def compute_final_stats_risk(self, net_1, net_2, avgnet1=None, avgnet2=None, input_1=None, target_1=None, input_2=None, target_2=None, data_loader_1=None, data_loader_2=None, clamping=True):
        
        # compute kl
        # for two posterior training method
        if self.prior_train_method == 'two':
            kl_1 = avgnet1.compute_kl()#.cpu()
            kl_2 = avgnet2.compute_kl()#.cpu()
            kl = (kl_1 + kl_2) / 2 
        # for one posterior training method 
        else:
            kl =  net_1.compute_kl().cpu()
                  
        # Reduce number
        if data_loader_1 != None:
            # for two posterior training method
            if self.prior_train_method == 'two':
                error_ce_1, error_01_1 = self.mcsampling(avgnet1, input_1, target_1, batches=True, data_loader=data_loader_1, clamping=True)
                error_ce_2, error_01_2= self.mcsampling(avgnet1, input_2, target_2, batches=True, data_loader=data_loader_2, clamping=True)
            else:
                error_ce_1, error_01_1 = self.mcsampling(net_1, input_1, target_1, batches=True, data_loader=data_loader_1, clamping=True)
                
        # Don't reduce number
        else:
            # for two posterior training method
            if self.prior_train_method == 'two':
                error_ce_1, error_01_1 = self.mcsampling(avgnet1, input_1, target_1, batches=False, clamping=True)
                error_ce_2, error_01_2 = self.mcsampling(avgnet1, input_2, target_2, batches=False, clamping=True)
            else:
                error_ce_1, error_01_1 = self.mcsampling(net_1, input_1, target_1, batches=False, clamping=True)         

        # 1) compute ubber bound (first kl-1) on MC estimate of expected empirical risk holding w prob 1 - delta_test 
        # for both posterior training methods
        empirical_risk_ce_1 = inv_kl(error_ce_1.item(), np.log(2/self.delta_test)/self.mc_samples)
        empirical_risk_01_1 = inv_kl(error_01_1, np.log(2/self.delta_test)/self.mc_samples)
        # for two posterior training method
        if self.prior_train_method == 'two':
            empirical_risk_ce_2 = inv_kl(error_ce_2.item(), np.log(2/self.delta_test)/self.mc_samples)
            empirical_risk_01_2 = inv_kl(error_01_2, np.log(2/self.delta_test)/self.mc_samples)
            empirical_risk_01 = (empirical_risk_01_1 + empirical_risk_01_2) / 2  # average the empirical risk estimates
        # for one posterior training method
        else:
            empirical_risk_01 = empirical_risk_01_1

        # 2) compute ubber bound (second kl-1) on expected risk holding w prob 1 - delta
        #risk_ce = inv_kl_torch_riv(empirical_risk_ce, (kl + np.log((2 *np.sqrt(self.n_bound))/self.delta_test))/self.n_bound)
        risk_01 = inv_kl(empirical_risk_01, (kl + np.log((2 *np.sqrt(self.n_bound))/self.delta))/self.n_bound)

        # return train_obj_1, train_obj_2, kl/self.n_bound, empirical_risk_01, risk_01 
        return kl/self.n_bound, empirical_risk_01, risk_01 



