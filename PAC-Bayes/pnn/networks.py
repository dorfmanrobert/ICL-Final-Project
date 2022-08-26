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


def output_transform(x, clamping=True, pmin=1e-6):
    """Computes the log softmax and clamps the values using the
    min probability given by pmin.

    Parameters
    ----------
    x : tensor
        output of the network

    clamping : bool
        whether to clamp the output probabilities

    pmin : float
        threshold of probabilities to clamp.
    """
    # lower bound output prob
    output = F.log_softmax(x, dim=1)
    if clamping:
        output = torch.clamp(output, np.log(pmin))
    return output


# Deterministic Networks
class Linear(nn.Module):
    """Implementation of a Linear layer (reimplemented to use
    truncated normal as initialisation for fair comparison purposes)

    Parameters
    ----------
    in_features : int
        Number of input features for the layer

    out_features : int
        Number of output features for the layer

    device : string
        Device the code will run in (e.g. 'cuda')

    """
    def __init__(self, in_features, out_features, device='cuda'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        sigma_weights = 1/np.sqrt(in_features)

        # same initialisation as before for the prob layer
        self.weight = nn.Parameter(trunc_normal_(torch.Tensor(
            out_features, in_features), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(
            out_features), requires_grad=True)

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        return F.linear(input, weight, bias)


class NNet4l(nn.Module):
    """Implementation of a standard Neural Network with 4 layers and dropout
    (used for the experiments on MNIST so it assumes a specific input size and
    number of classes)

    Parameters
    ----------
    dropout_prob : float
        probability of an element to be zeroed.

    device : string
        Device the code will run in (e.g. 'cuda')

    """
    def __init__(self, dropout_prob=0.0, device='cuda'):
        super().__init__()
        self.l1 = Linear(28*28, 600, device)
        self.l2 = Linear(600, 600, device)
        self.l3 = Linear(600, 600, device)
        self.l4 = Linear(600, 2, device)
        self.d = nn.Dropout(dropout_prob)

    def forward(self, x):
        # forward pass for the network
        x = x.view(-1, 28*28)
        x = self.d(self.l1(x))
        x = F.relu(x)
        x = self.d(self.l2(x))
        x = F.relu(x)
        x = self.d(self.l3(x))
        x = F.relu(x)
        x = output_transform(self.l4(x), clamping=False)
        return x

class ConvNet2l(nn.Module):
    """Implementation of a standard Neural Network with 4 layers and dropout
    (used for the experiments on MNIST so it assumes a specific input size and
    number of classes)

    Parameters
    ----------
    dropout_prob : float
        probability of an element to be zeroed.

    device : string
        Device the code will run in (e.g. 'cuda')

    """
    def __init__(self, dropout_prob=0.0, device='cuda', prior_dist = 'gaussian'):
        super().__init__()
        self.prior_dist = prior_dist
        self.l1 = nn.Conv2d(1, 32, 3, device=device)
        self.l2 = nn.Conv2d(32, 64, 3, device=device)
        self.l3 = Linear(9216, 128, device=device)
        self.l4 = Linear(128, 2, device=device)
        self.d = nn.Dropout(dropout_prob)



    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1) # flatten
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return x
    
def trainNNet(net, optimizer, epoch, train_loader, device='cuda', verbose=False):
    """Train function for a standard NN (including CNN)

    Parameters
    ----------
    net : NNet/CNNet object
        Network object to train

    optimizer : optim object
        Optimizer to use (e.g. SGD/Adam)

    epoch : int
        Current training epoch

    train_loader: DataLoader object
        Train loader to use for training

    device : string
        Device the code will run in (e.g. 'cuda')

    verbose: bool
        Whether to print training metrics

    """
    # train and report training metrics
    net.train()
    total, correct, avgloss = 0.0, 0.0, 0.0
    for batch_id, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.flatten().to(device)
        net.zero_grad()
        outputs = net(data)
        loss = F.nll_loss(outputs, target)
        pred = torch.max(outputs, -1)[1]

        loss.backward()
        optimizer.step()

        correct += (pred==target).float().sum().item()
        total += target.size(0)
        avgloss = avgloss + loss.detach()

    # show the average loss and KL during the epoch
    if verbose:
        print(f"-Epoch {epoch :.5f}, Train loss: {avgloss/batch_id :.5f}, Train err:  {1-(correct/total):.5f}")


def testNNet(net, test_loader, device='cuda', verbose=True):
    """Test function for a standard NN (including CNN)

    Parameters
    ----------
    net : NNet/CNNet object
        Network object to train

    test_loader: DataLoader object
        Test data loader

    device : string
        Device the code will run in (e.g. 'cuda')

    verbose: bool
        Whether to print test metrics

    """
    net.eval()
    correct, total = 0, 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.flatten().to(device)
            outputs = net(data)
            loss = F.nll_loss(outputs, target)
            pred = torch.max(outputs, -1)[1]

            correct += (pred==target).float().sum().item()
            total += target.size(0)

    if verbose:
        print(f"-Prior: Test loss: {loss :.5f}, Test err:  {1-(correct/total):.5f}")

    return 1-(correct/total)



# Probabilistic Networks
class Gaussian(nn.Module):
    """Implementation of a Gaussian random variable, using softplus for
    the standard deviation and with implementation of sampling and KL
    divergence computation.

    Parameters
    ----------
    mu : Tensor of floats
        Centers of the Gaussian.

    rho : Tensor of floats
        Scale parameter of the Gaussian (to be transformed to std
        via the softplus function)

    device : string
        Device the code will run in (e.g. 'cuda')

    fixed : bool
        Boolean indicating whether the Gaussian is supposed to be fixed
        or learnt.

    """

    def __init__(self, mu, rho, device='cuda', fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)
        self.device = device

    @property
    def sigma(self):
        # Computation of standard deviation:
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. Specifically, we use sigma = log(exp(rho)+1)
        return torch.log(1 + torch.exp(self.rho))
        

    def sample(self):
        # Return a sample from the Gaussian distribution
        epsilon = torch.randn(self.sigma.size()).to(self.device)
        return self.mu + self.sigma * epsilon

    def compute_kl(self, other):
        # Compute KL divergence between two Gaussians (self and other)
        # (refer to the paper)
        # b is the variance of priors
        b1 = torch.pow(self.sigma, 2)
        b0 = torch.pow(other.sigma, 2)

        term1 = torch.log(torch.div(b0, b1))
        term2 = torch.div(
            torch.pow(self.mu - other.mu, 2), b0)
        term3 = torch.div(b1, b0)
        kl_div = (torch.mul(term1 + term2 + term3 - 1, 0.5)).sum()
        return kl_div
 

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used works best if :math:`\text{mean}` is
    near the center of the interval.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
    
    
   
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Fill tensor with uniform values from [l, u]
        tensor.uniform_(l, u)

        # Use inverse cdf transform from normal distribution
        tensor.mul_(2)
        tensor.sub_(1)

        # Ensure that the values are strictly between -1 and 1 for erfinv
        eps = torch.finfo(tensor.dtype).eps
        tensor.clamp_(min=-(1. - eps), max=(1. - eps))
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp one last time to ensure it's still in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

# Linear architecture
class ProbLinear(nn.Module):
    """Implementation of a Probabilistic Linear layer.

    Parameters
    ----------
    in_features : int
        Number of input features for the layer

    out_features : int
        Number of output features for the layer

    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_layer : Linear object
        Linear layer object used to initialise the prior

    init_prior : string
        string that indicates the way to initialise the prior:
        *"weights" = initialise with init_layer
        *"zeros" = initialise with zeros and rho prior
        *"random" = initialise with random weights and rho prior
        *""

    """

    def __init__(self, in_features, out_features, rho_prior, prior_dist='gaussian', device='cuda', init_prior='weights', init_layer=False, init_layer_prior=None, prior_train_opt='det'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        sigma_weights = 1/np.sqrt(in_features)

        if init_layer:
            if prior_train_opt == 'pb':
                weights_mu_init = init_layer.weight.mu
                bias_mu_init = init_layer.bias.mu
                weights_rho_init = init_layer.weight.rho
                bias_rho_init = init_layer.bias.rho
            elif prior_train_opt == 'det':
                weights_mu_init = init_layer.weight
                bias_mu_init = init_layer.bias
                weights_rho_init = torch.ones(out_features, in_features) * rho_prior
                bias_rho_init = torch.ones(out_features) * rho_prior
        else:
            # Initialise distribution means using truncated normal
            weights_mu_init = trunc_normal_(torch.Tensor(out_features, in_features), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_init = torch.zeros(out_features)

            weights_rho_init = torch.ones(out_features, in_features) * rho_prior
            bias_rho_init = torch.ones(out_features) * rho_prior


        if init_prior == 'zeros':
            bias_mu_prior = torch.zeros(out_features) 
            weights_mu_prior = torch.zeros(out_features, in_features)
        elif init_prior == 'random':
            weights_mu_prior = trunc_normal_(torch.Tensor(
                out_features, in_features), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_prior = torch.zeros(out_features) 
        elif init_prior == 'weights': 
            if init_layer_prior:
                weights_mu_prior = init_layer_prior.weight
                bias_mu_prior = init_layer_prior.bias
            else:
                # otherwise initialise to posterior weights
                weights_mu_prior = weights_mu_init
                bias_mu_prior = bias_mu_init
        else: 
            raise RuntimeError(f'Wrong type of prior initialisation!')

        if prior_dist == 'gaussian':
            dist = Gaussian
        elif prior_dist == 'laplace':
            dist = Laplace
        else:
            raise RuntimeError(f'Wrong prior_dist {prior_dist}')

        self.bias = dist(bias_mu_init.clone(), bias_rho_init.clone(), device=device, fixed=False)
        self.weight = dist(weights_mu_init.clone(), weights_rho_init.clone(), device=device, fixed=False)
        self.weight_prior = dist(weights_mu_prior.clone(), weights_rho_init.clone(), device=device, fixed=True)
        self.bias_prior = dist(bias_mu_prior.clone(), bias_rho_init.clone(), device=device, fixed=True)
        self.kl_div = 0
 
    def forward(self, input, sample=False):
        if self.training or sample:
            # during training we sample from the model distribution
            # sample = True can also be set during testing if we
            # want to use the stochastic/ensemble predictors
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            # otherwise we use the posterior mean
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(self.weight_prior) + self.bias.compute_kl(self.bias_prior)

        return F.linear(input, weight, bias)


class ProbNNet4l(nn.Module):
    """Implementation of a Probabilistic Neural Network with 4 layers
    (used for the experiments on MNIST so it assumes a specific input size and
    number of classes)

    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_net : NNet object
        Network object used to initialise the prior

    """

    def __init__(self, rho_prior, device='cuda', init_net=None,  prior_dist='gaussian', train_method='original', prior_train_opt='det'):
        super().__init__()
        self.train_method = train_method
        self.l1 = ProbLinear(28*28, 600, rho_prior, prior_dist=prior_dist, device=device, prior_train_opt=prior_train_opt, init_layer=init_net.l1 if init_net is not None else None)
        self.l2 = ProbLinear(600, 600, rho_prior, prior_dist=prior_dist, device=device, prior_train_opt=prior_train_opt, init_layer=init_net.l2 if init_net is not None else None)
        self.l3 = ProbLinear(600, 600, rho_prior, prior_dist=prior_dist, device=device, prior_train_opt=prior_train_opt, init_layer=init_net.l3 if init_net is not None else None)
        self.l4 = ProbLinear(600, 2, rho_prior, prior_dist=prior_dist, device=device, prior_train_opt=prior_train_opt, init_layer=init_net.l4 if init_net is not None else None)
 
    def forward(self, x, sample=False, clamping=True, pmin=1e-6):
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.relu(self.l3(x, sample))
        if self.train_method == 'original':
            x = output_transform(self.l4(x, sample), clamping, pmin)
        elif self.train_method == 'conditional':
            x = F.relu(self.l4(x, sample))
        return x
    
    def hidden_out(self, x, sample=False):
        #Return the output of the last hidden layer
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.relu(self.l3(x, sample))
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.l1.kl_div + self.l2.kl_div + self.l3.kl_div + self.l4.kl_div


# Convolutional architecture
class ProbConv2d(nn.Module):
    """Implementation of a Probabilistic Convolutional layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels for the layer

    out_channels : int
        Number of output channels for the layer

    kernel_size : int
        size of the convolutional kernel

    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    stride : int
        Stride of the convolution

    padding: int
        Zero-padding added to both sides of the input

    dilation: int
        Spacing between kernel elements

    init_layer : Linear object
        Linear layer object used to initialise the prior

    """

    def __init__(self, in_channels, out_channels, kernel_size, rho_prior, prior_dist='gaussian',
                 device='cuda', stride=1, padding=0, dilation=1, init_prior='weights', init_layer=False, init_layer_prior=None, prior_train_opt='det'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1



        # Compute and set sigma for the truncated gaussian of weights
        in_features = self.in_channels
        for k in self.kernel_size:
            in_features *= k
        sigma_weights = 1/np.sqrt(in_features)


        if init_layer:
            # Initialize with other networks parameters
            if prior_train_opt == 'pb':
                weights_mu_init = init_layer.weight.mu
                bias_mu_init = init_layer.bias.mu
                weights_rho_init = init_layer.weight.rho
                bias_rho_init = init_layer.bias.rho
            elif prior_train_opt == 'det':
                weights_mu_init = init_layer.weight
                bias_mu_init = init_layer.bias
                weights_rho_init = torch.ones(out_channels, in_channels, *self.kernel_size) * rho_prior
                bias_rho_init = torch.ones(out_channels) * rho_prior
        else:
            # Initialize with truncated normal
            weights_mu_init = trunc_normal_(torch.Tensor(out_channels, in_channels, *self.kernel_size), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_init = torch.zeros(out_channels)
            weights_rho_init = torch.ones(out_channels, in_channels, *self.kernel_size) * rho_prior
            bias_rho_init = torch.ones(out_channels) * rho_prior

        if init_prior == 'zeros':
            bias_mu_prior = torch.zeros(out_channels) 
            weights_mu_prior = torch.zeros(out_channels, in_features)
        elif init_prior == 'random':
            weights_mu_prior = trunc_normal_(torch.Tensor(
                out_channels, in_channels, *self.kernel_size), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_prior = torch.zeros(out_channels) 
        elif init_prior == 'weights': 
            if init_layer_prior:
                weights_mu_prior = init_layer_prior.weight
                bias_mu_prior = init_layer_prior.bias
            else:
                # otherwise initialise to posterior weights
                weights_mu_prior = weights_mu_init
                bias_mu_prior = bias_mu_init
        else: 
            raise RuntimeError(f'Wrong type of prior initialisation!')

        if prior_dist == 'gaussian':
            dist = Gaussian
        elif prior_dist == 'laplace':
            dist = Laplace
        else:
            raise RuntimeError(f'Wrong prior_dist {prior_dist}')

        self.weight = dist(weights_mu_init.clone(),weights_rho_init.clone(), device=device, fixed=False)
        self.bias = dist(bias_mu_init.clone(), bias_rho_init.clone(), device=device, fixed=False)
        self.weight_prior = dist(weights_mu_prior.clone(), weights_rho_init.clone(), device=device, fixed=True)
        self.bias_prior = dist(bias_mu_prior.clone(), bias_rho_init.clone(), device=device, fixed=True)

        self.kl_div = 0

    def forward(self, input, sample=False):
        if self.training or sample:
            # during training we sample from the model distribution
            # sample = True can also be set during testing if we
            # want to use the stochastic/ensemble predictors
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            # otherwise we use the posterior mean
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(
                self.weight_prior) + self.bias.compute_kl(self.bias_prior)

        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    
class ProbConvNet2l(nn.Module):
    """Implementation of a Probabilistic Neural Network with 4 layers
    (used for the experiments on MNIST so it assumes a specific input size and
    number of classes)

    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_net : NNet object
        Network object used to initialise the prior

    """

    def __init__(self, rho_prior, device='cuda', init_net=None,  prior_dist='gaussian', train_method='original', prior_train_opt='det'):
        super().__init__()
        self.train_method = train_method
        self.l1 = ProbConv2d(1, 32, 3, rho_prior, prior_dist=prior_dist,
                             device=device, prior_train_opt=prior_train_opt, init_layer=init_net.l1 if init_net is not None else None)
        self.l2 = ProbConv2d(32, 64, 3, rho_prior, prior_dist=prior_dist,
                             device=device, prior_train_opt=prior_train_opt, init_layer=init_net.l2 if init_net is not None else None)
        self.l3 = ProbLinear(9216, 128, rho_prior, prior_dist=prior_dist,
                             device=device, prior_train_opt=prior_train_opt, init_layer=init_net.l3 if init_net is not None else None)
        self.l4 = ProbLinear(128, 2, rho_prior, prior_dist=prior_dist,
                            device=device, prior_train_opt=prior_train_opt, init_layer=init_net.l4 if init_net is not None else None)


    def forward(self, x, sample=False, clamping=True, pmin=1e-6):
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1) # flatten
        x = F.relu(self.l3(x))
        if self.train_method=='original':
            x = output_transform(self.l4(x), clamping, pmin)
        elif self.train_method== 'conditional':
            x = F.relu(self.l4(x, sample))
        return x

    def hidden_out(self, x, sample=False):
        #Return the output of the last hidden layer
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1) # flatten
        x = F.relu(self.l3(x))
        return x
 
    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.l1.kl_div + self.l2.kl_div + self.l3.kl_div + self.l4.kl_div

    
def trainPNNet(net, optimizer, pbobj, epoch, train_loader, verbose=True):
    """Train function for a probabilistic NN (including CNN) trained with the CondLap algorithm

    Parameters
    ----------
    net : ProbNNet/ProbCNNet object
        Network object to train

    optimizer : optim object
        Optimizer to use (e.g. SGD/Adam)

    pbobj : pbobj object
        PAC-Bayes inspired training objective to use for training

    epoch : int
        Current training epoch

    train_loader: DataLoader object
        Train loader to use for training

    device : string
        Device the code will run in (e.g. 'cuda')

    verbose: bool
        Whether to print test metrics

    """
    net.train()
    # variables that keep information about the results of optimising the bound
    avgerr, avgbound, avgkl, avgloss = 0.0, 0.0, 0.0, 0.0

    for batch_id, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(pbobj.device), target.flatten().to(pbobj.device)
        net.zero_grad()
        bound, kl, _, err = pbobj.train_obj(net, data, target)

        bound.backward()
        optimizer.step()

        avgbound += bound.item()
        avgkl += kl
        avgerr += err
    
    if verbose:
        # show the average of the metrics during the epoch
        print(
            f"-Batch average epoch {epoch :.0f} results, Avg train obj: {avgbound/batch_id :.5f}, KL/n: {avgkl/batch_id :.5f}, Avg train 0-1 error:  {avgerr/batch_id :.5f}")
  

def testStochastic(net, test_loader, pbobj, device='cuda'):
    """Test function for the stochastic predictor using a PNN

    Parameters
    ----------
    net : PNNet/PCNNet object
        Network object to test

    test_loader: DataLoader object
        Test data loader

    pbobj : pbobj object
        PAC-Bayes inspired training objective used during training

    device : string
        Device the code will run in (e.g. 'cuda')

    """
    # compute mean test accuracy
    net.eval()
    correct, total = 0, 0.0
    outputs = torch.zeros(test_loader.batch_size, int(pbobj.classes - 1)).to(device)

    with torch.no_grad():
        for batch_id, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.to(device), target.flatten().to(device)
            outputs = net(data, sample=True)
            pred = pred = torch.max(outputs, -1)[1]
   
            correct += (pred==target).float().sum().item()
            total += target.size(0)

    return 1-(correct/total)


def computeRiskCertificates(avgnet1, avgnet2, net1, net2, pbobj, toolarge=False, device='cuda',train_loader_1=None, train_loader_2=None, set_bound_1batch_1=None, set_bound_1batch_2=None):
    
    """Function to compute risk certificates and other statistics at the end of training

    Parameters
    ----------
    net : PNNet/PCNNet object
        Network object to test

    toolarge: bool
        Whether the dataset is too large to fit in memory (computation done in batches otherwise)

    pbobj : pbobj object
        PAC-Bayes inspired training objective used during training

    device : string
        Device the code will run in (e.g. 'cuda')

    train_loader: DataLoader object
        Data loader for computing the risk certificate (multiple batches, used if toolarge=True)

    whole_train: DataLoader object
        Data loader for computing the risk certificate (one unique batch, used if toolarge=False)

    """
    net1.eval()
    # Double prior train method
    if net2 != None:
        net2.eval()
        with torch.no_grad():
            if toolarge:
                train_obj_1, train_obj_2, kl, err_01_train, ub_risk_01 = pbobj.compute_final_stats_risk(net1, net2, avgnet1=avgnet1, avgnet2=avgnet2, data_loader=train_loader_1)
            else:
                # a bit hacky, we load the whole dataset to compute the bound
                for data, target in set_bound_1batch_1:
                    data_1, target_1 = data.to(device), target.flatten().to(device)
                for data, target in set_bound_1batch_2:
                    data_2, target_2 = data.to(device), target.flatten().to(device)
                # train_obj_1, train_obj_2, kl, err_01_train, ub_risk_01 = pbobj.compute_final_stats_risk(net1, net2, avgnet1=avgnet1, avgnet2=avgnet2, input_1=data_1, target_1=target_1, input_2=data_2, target_2 = target_2)
                kl, err_01_train, ub_risk_01 = pbobj.compute_final_stats_risk(net1, net2, avgnet1=avgnet1, avgnet2=avgnet2, input_1=data_1, target_1=target_1, input_2=data_2, target_2 = target_2)
    
    # Original prior train method
    else:
        with torch.no_grad():
            if toolarge:
                train_obj_1, train_obj_2, kl, err_01_train, ub_risk_01 = pbobj.compute_final_stats_risk(net1, net2, data_loader=train_loader_1)
            else:
                # a bit hacky, we load the whole dataset to compute the bound
                for data, target in set_bound_1batch_1:
                    data_1, target_1 = data.to(device), target.flatten().to(device)
                    # train_obj_1, train_obj_2, kl, err_01_train, ub_risk_01 = pbobj.compute_final_stats_risk(net1, net2, input_1=data_1, target_1=target_1, input_2=None, target_2=None)
                    kl, err_01_train, ub_risk_01 = pbobj.compute_final_stats_risk(net1, net2, input_1=data_1, target_1=target_1, input_2=None, target_2=None)


    # return train_obj_1, train_obj_2, ub_risk_01, kl, err_01_train
    return ub_risk_01, kl, err_01_train