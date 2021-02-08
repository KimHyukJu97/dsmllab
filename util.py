import numpy as np
import os
import random
import torch

def version_update(logdir):
    # make logs folder
    savedir = logdir
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    # check save version
    version = len(os.listdir(savedir))

    # make save folder
    savedir = os.path.join(savedir, f"version{version}")
    os.mkdir(savedir)

    # make model checkpoint folder
    ckptdir = os.path.join(savedir, "checkpoint")
    os.mkdir(ckptdir)

    print(f"Version {version}")

    return savedir, ckptdir

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
