import pandas as pd
import os

import argparse
import json 
import pickle

from load import dataloader
from train import training
from util import version_update, seed_everything

import warnings
warnings.filterwarnings(action = "ignore")

seed_everything(42)

if __name__ == "__main__":
    # config
    parser = argparse.ArgumentParser(description = "Machine Learning Pipeline")
    parser.add_argument("--datadir", type = str, default = "./data", help = "Set data directory")
    parser.add_argument("--logdir", type = str, default = "./logs", help = "Set log directory")
    parser.add_argument("--seed", type = int, default = 42, help = "Set seed")
    parser.add_argument("--learning_rate", type = float, default = 0.0005, help = "Set learning rate")
    parser.add_argument("--batch", type = int, default = 30, help = "Number of batch size")
    parser.add_argument("--fold", type = int, default = 5, help = "Number of cross validation")
    parser.add_argument("--val_size", type = float, default = 0.2, help = "Set validation size")
    parser.add_argument("--mul_num", type = int, default = 3, help = "Set multiplication of rsampling")
    parser.add_argument("--method", type = int, default = 0, help = "Set resampling method")
    parser.add_argument("--pre_epochs", type = int, default = 5, help = "Number of pre-training epoch")
    parser.add_argument("--epochs", type = int, default = 15, help = "Number of training epoch")

    args = parser.parse_args()

    # set version and define save directory
    savedir, ckptdir = version_update(args.logdir)

    # save argument
    json.dump(vars(args), open(os.path.join(savedir, "arguments.json"), "w"))

    # 1. data load
    data, X_features, y_target = dataloader(datadir = args.datadir)

    # 2. training & predict test set
    results = training(data = data,
                       X_features = X_features, 
                       y_target = y_target,
                       seed = args.seed, 
                       learning_rate = args.learning_rate, 
                       batch = args.batch, 
                       fold = args.fold,
                       val_size = args.val_size,
                       mul_num = args.mul_num,
                       method = args.method,
                       pre_epochs = args.pre_epochs,
                       epochs = args.epochs)

    # 3. save results
    pickle.dump(results, open(os.path.join(savedir, f"results.pkl"), "wb"))
    print(results)
    