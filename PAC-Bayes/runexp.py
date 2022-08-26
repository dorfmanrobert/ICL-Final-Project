import torch
import optuna
import logging
import sys
import pickle

from pnn.utils import runexp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {torch.cuda.is_available()}')

# Fixed hyperparameters
BATCH_SIZE = 250 
DELTA = 0.025
DELTA_TEST = 0.01            # confidence parameter for computing final bound

PRIOR_DIST = 'gaussian'      # distribution used in prior/posterior

OBJECTIVE = 'invkl'          # training bound used :'invkl', 'quad', ...
TRAIN_METHOD = 'original'    # conditional 0-1 or with cross entropy: 'conditional' or 'original'
PRIOR_TRAIN_METHOD = 'one'   # 'one' (one prior, one posterior) or 'two' (two priors, two posteriors)
PRIOR_TRAIN_OPT = 'det'      # 'det' (train NN with SGD) or 'pb' (train PNN w PAC-Bayes objective)

MODEL_TYPE = 'fcn'           # type of model trained: 'fcn' or 'cnn'
NAME_DATA = 'binarymnist'

PERC_TRAIN = 1             # percent to lower the number of training data
PERC_PRIOR = .5              # percent of training data used to train prior


TRAIN_EPOCHS = 100
PRIOR_EPOCHS = 100

MC_SAMPLES = 10000


LEARNING_RATE = .001
MOMENTUM = .9

LEARNING_RATE_PRIOR = .01
MOMENTUM_PRIOR = .95
DROPOUT_PROB = .2

SIGMAPRIOR = .01

risk_certificate, test_loss01 = runexp(SIGMAPRIOR, LEARNING_RATE, MOMENTUM, prior_train_opt=PRIOR_TRAIN_OPT, prior_train_method=PRIOR_TRAIN_METHOD, train_method=TRAIN_METHOD, model_type=MODEL_TYPE, objective=OBJECTIVE, prior_dist=PRIOR_DIST, learning_rate_prior=LEARNING_RATE_PRIOR, momentum_prior=MOMENTUM_PRIOR, delta=DELTA, delta_test=DELTA_TEST, mc_samples=MC_SAMPLES, train_epochs=TRAIN_EPOCHS, prior_epochs=PRIOR_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE, perc_train=PERC_TRAIN, perc_prior=PERC_PRIOR, verbose=False, verbose_test=False, dropout_prob=DROPOUT_PROB, name_data=NAME_DATA)


# with open('cert-loss-200prior-500posterior-onepri.pkl', 'wb') as f:
#     pickle.dump([risk_certificate,test_loss01] ,f)
    
print(risk_certificate, test_loss01)



