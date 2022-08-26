import torch
import numpy as np
import random
from alg.utils import SArunexp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {torch.cuda.is_available()}')

random_seed = 10
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
random.seed(random_seed)
np.random.seed(random_seed)
g = torch.Generator(device='cpu').manual_seed(random_seed)

BATCH_SIZE = 64 # 250
TRAIN_EPOCHS = 200
DELTA = 0.035
MODEL_TYPE = 'fcn'           
NAME_DATA = 'binarymnist'

LEARNING_RATE = 0.01 
MOMENTUM = .95 
DROPOUT_PROB = .2

PERC_TRAIN = 5 / 60
NUM_SUPP_INIT = 0

VERBOSE = True
CONT=True

C = .69314718 # initial loss value we stop at


sa_ub, p_misclass, p_change_comp, C, supp_num, nonsupp_num = SArunexp(g, C, name_data=NAME_DATA, delta=DELTA, learning_rate=LEARNING_RATE, momentum=MOMENTUM, batch_size=BATCH_SIZE, train_epochs=TRAIN_EPOCHS, dropout_prob=DROPOUT_PROB, perc_train=PERC_TRAIN, num_supp_init=NUM_SUPP_INIT, device=DEVICE, verbose=VERBOSE, continuing=CONT)


print("***FINAL RESULTS***")
print(f"SA bound {sa_ub}")
print(f"p_change_comp{p_change_comp}")
print(f"p_misclass{p_misclass}")
print(f"epsilon{C}")
print(f"number support points{supp_num}")
print(f"number non support points{nonsupp_num}")

