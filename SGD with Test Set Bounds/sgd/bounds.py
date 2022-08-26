import math
import numpy as np
from scipy.special import xlogy

from sgd.networks import testNNet

# Compute test set bounds
# Chernoff bound
def kl_bin(q, p):
    eps = 1e-6
    # make sure q, p in (0,1)
    q = sorted([eps, q, 1 - eps])[1]
    p = sorted([eps, p, 1 - eps])[1]
    return xlogy(q, q/p) + xlogy(1-q, (1-q)/(1-p))

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

def chernoff(net, test_loader, delta, device='cuda'):
    test_err = testNNet(net, test_loader, device=device, verbose=False)
    n_test = len(test_loader.dataset)
    kl_ub = np.log(1 / delta) / n_test
    chernoff_ub = inv_kl(test_err, kl_ub)
    return chernoff_ub


# Binomial bound
# Code adapted from https://github.com/cambridge-mlg/pac-bayes-tightness-small-data
def inv_binomial(n, k, delta):
    eps = 1e-6
    k = np.round(k).astype(int)
    p = np.linspace(eps, 1 - eps, num=10_000)
    log_terms = []
    for i in range(0, k + 1):
        log_comb = math.lgamma(n + 1) - math.lgamma(n - i + 1) - math.lgamma(i + 1) 
        log_pmf = log_comb + i * np.log(p) + (n - i) * np.log(1 - p)
        log_terms.append(log_pmf)
    valid = np.log(np.sum(np.exp(np.array(log_terms)), axis=0)) <= np.log(delta)
    binomial_ub = p[np.argmax(valid)]
    return binomial_ub

def binomial(net, test_loader, delta, device='cuda'):
    test_err = testNNet(net, test_loader, device=device, verbose=False)
    n_test = len(test_loader.dataset)
    binomial_ub = inv_binomial(n_test, n_test*test_err, delta)
    return binomial_ub