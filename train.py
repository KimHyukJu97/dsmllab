import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from preprocess import resampling
from model import Net

def training(data, X_features, y_target, seed, learning_rate, batch, fold, val_size, mul_num, method, pre_epochs, epochs):

    skf = StratifiedKFold(n_splits = fold, shuffle = True, random_state = seed)
    results = pd.DataFrame()
    i = 1

    for train_idx, test_idx in tqdm(skf.split(X_features, y_target)):

        # train, validation, test set split
        X_train = data.iloc[train_idx, :-1]
        y_train = data.iloc[train_idx,  -1]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = val_size, stratify = y_train, random_state = seed)
        X_test = data.iloc[test_idx, :-1]
        y_test = data.iloc[test_idx,  -1]

        # resampling
        X_rs, y_rs = resampling(X_train = X_train, y_train = y_train, mul_num = mul_num, method = method)

        # to tensor
        X_train = torch.FloatTensor(X_train.values)
        y_train = torch.LongTensor(y_train.values)
        X_rs = torch.FloatTensor(X_rs.values)
        y_rs = torch.LongTensor(y_rs.values)
        X_test = torch.FloatTensor(X_test.values)
        y_test = torch.LongTensor(y_test.values)

        train = TensorDataset(X_train, y_train)
        rs = TensorDataset(X_rs, y_rs)

        # model
        model = Net()

        # optimizer & criterion
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)
        criterion = nn.CrossEntropyLoss()

        # pre-training
        rs_loader = DataLoader(rs, batch_size = batch, shuffle = False)

        for epochs in range(pre_epochs):

            for X_rs, y_rs in rs_loader:

                optimizer.zero_grad()
                rs_output = model(X_rs)
                rs_loss = criterion(rs_output, y_rs)
                rs_loss.backward()
                optimizer.step()

        # training
        train_loader = DataLoader(train, batch_size = batch, shuffle = False)

        for epochs in range(epochs):

            for X_train, y_train in train_loader:

                optimizer.zero_grad()
                train_output = model(X_train)
                train_loss = criterion(train_output, y_train)
                train_loss.backward()
                optimizer.step()

        # evaluation
        output = model(X_test)
        _, preds = torch.max(output, 1)
        accuracy = torch.sum(preds == y_test) / len(X_test)
        results.loc["accuracy", i] = accuracy.detach().numpy()

        i += 1

    results["mean"] = np.mean(results, axis = 1)

    return results