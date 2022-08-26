import torch
import logging
import sys
import numpy as np
import random
import pickle
from sgd.utils import runexp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {torch.cuda.is_available()}')
random_seed = 10
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
random.seed(random_seed)
np.random.seed(random_seed)
g = torch.Generator(device='cpu').manual_seed(random_seed)



BATCH_SIZE = 64       # 250 for binarymnnist, 8 for 
TRAIN_EPOCHS = 200            # best of 100,200 for binarymnist    300 best of 100,200,300,400 for medmnist, 
DELTA = .035

MODEL_TYPE = 'fcn'           # 'fcn' or 'cnn'
NAME_DATA = 'binarymnist'    # binarymnist or medmninst

PMIN = 1e-6

PERC_TRAIN = 5/60
PERC_VAL = .2

VERBOSE = False


LEARNING_RATE = .01
MOMENTUM = .95
DROPOUT_PROB = .2

binomial, chernoff, test_loss01 = runexp(LEARNING_RATE, MOMENTUM, model_type=MODEL_TYPE, train_epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE, perc_train=PERC_TRAIN, perc_val = PERC_VAL, verbose=VERBOSE, dropout_prob=DROPOUT_PROB, name_data=NAME_DATA, pmin=PMIN)

results = [binomial, chernoff, test_loss01]
with open('bin-cher-testloss01.pkl', 'wb') as f:
    pickle.dump(results,f)
    
print("***Results***")
print(f"binomial{binomial}")
print(f"chernoff{chernoff}")
print(f"test_loss01{test_loss01}")



