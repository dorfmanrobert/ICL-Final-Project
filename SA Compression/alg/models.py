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


# Most of the network implementation below is from https://github.com/mperezortiz/PBB

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    """Fills the input Tensor with values drawn from a truncated
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
        x = F.log_softmax(self.l4(x), dim=1)
        return x

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
        x = F.log_softmax(self.l4(x), dim=1)
        return x

def trainNNet(net, optimizer, epoch, train_loader, device='cuda', verbose=False, prior_dist = 'gaussian'):
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
    for batch_id, (data, target) in enumerate(tqdm(train_loader, disable=True)):
        data, target = data.to(device), target.to(device)
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
    # if verbose:
    #     print(
    #         f"-Epoch {epoch :.5f}, Train loss: {avgloss/batch_id :.5f}, Train err:  {1-(correct/total):.5f}")
    
    return 1-(correct/total)

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
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            
            loss = F.nll_loss(outputs, target)

            pred = torch.max(outputs, -1)[1]
            correct += (pred==target).float().sum().item()
            total += target.size(0)
    if verbose:
        print(f"Test loss: {loss :.5f}, Test err:  {1-(correct/total):.5f}")
    
    return 1-(correct/total)
