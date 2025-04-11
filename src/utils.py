import os
import random

import numpy as np
import torch

from torch.backends import cudnn



def set_seed(seed):
    """Set seed in every way possible."""
    print(f"setting seeds to {seed}")
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        print(f"setting cuda seeds to {seed}")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def test_cuda_seed():
    """Print some random results from various libraries."""
    print(f"python random float: {random.random()}")
    print(f"numpy random int: {np.random.randint(100)}")
    print(f"torch random tensor (cpu): {torch.FloatTensor(100).uniform_()}")
    print(f"torch random tensor (cuda): {torch.cuda.FloatTensor(100).uniform_()}")

