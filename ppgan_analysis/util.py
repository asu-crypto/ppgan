import random
import os
import torch
import numpy as np

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.urandom(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    next_seed = torch.tensor(seed)
    global_seed = torch.tensor(seed)