import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def log_transformation(X = None):
    X_log = np.log1p(X)

    return X_log

def scaling(X = None, y = None):
    scaler = StandardScaler()
    scale_fit = scaler.fit_transform(X, y)
    X_scaled = pd.DataFrame(scale_fit, index = X.index, columns = X.columns)

    return X_scaled

def resampling(X_train = None, y_train = None, mul_num = None, method = 0):

    data = pd.concat([X_train, y_train], axis = 1)
    df_rs = pd.DataFrame(columns = data.columns)

    if method == 0:

        data_0 = data.loc[y_train == 0]
        data_1 = data.loc[y_train == 1]
        
        for i in range(len(data_0) * mul_num):
            sam = data_0.sample(2, random_state = i)
            gen = sam.mean(axis = 0)
            df_rs.loc[f"LUAD{i + 1}"] = gen

        for i in range(len(data_1) * mul_num):
            sam = data_1.sample(2, random_state = i)
            gen = sam.mean(axis = 0)
            df_rs.loc[f"LUSC{i + 1}"] = gen

    if method == 1:

        i = 0
        idx0 = 1
        idx1 = 1

        while len(df_rs) < len(data) * mul_num:

            sam = data.sample(5, random_state = i)
            gen = sam.mean(axis = 0)
            val = gen[y_train.name]

            if val < 1/3:
                gen[y_train.name] = 0.0
                df_rs.loc[f"LUAD{idx0}"] = gen
                idx0 += 1

            elif val > 2/3:
                gen[y_train.name] = 1.0
                df_rs.loc[f"LUSC{idx1}"] = gen
                idx1 += 1

            i += 1

    if method == 2:

        i = 0
        idx0 = 1
        idx1 = 1

        while len(df_rs) < len(data) * mul_num:

            sam = data.sample(3, random_state = i)
            gen = sam.mean(axis = 0)
            val = gen[y_train.name]

            if val < 0.5:
                gen[y_train.name] = 0.0
                df_rs.loc[f"LUAD{idx0}"] = gen
                idx0 += 1

            elif val > 0.5:
                gen[y_train.name] = 1.0
                df_rs.loc[f"LUSC{idx1}"] = gen
                idx1 += 1

            i += 1

    X_rs = df_rs.iloc[:, :-1]
    y_rs = df_rs.iloc[:,  -1]

    return X_rs, y_rs

    