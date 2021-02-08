import numpy as np
import pandas as pd
import os
import re

from preprocess import log_transformation, scaling

def dataloader(datadir):
    df = pd.read_csv(os.path.join(datadir, f"LUADLUSC_float32.tsv"), delimiter = "\t", index_col = 0).astype("float32")
    df_copy = df.copy()

    X = df_copy.iloc[:, :-1]
    y = df_copy.iloc[:,  -1]

    ### gene selection

    # external data load
    filename = os.path.join(datadir, f"c4.cgn.v7.2.entrez.gmt")
    data = []
    with open(filename, "r", encoding = "utf-8-sig") as f_input:
        for line in f_input:
            data.append(list(line.strip().split("\t")))
            
    cancer_gene = pd.DataFrame(data)
    cancer_gene.index = np.array(cancer_gene[0])
    cancer_gene.drop(columns = [0, 1], axis = 1, inplace = True)
    cancer_gene.columns = np.arange(len(cancer_gene.columns))

    # directly related gene
    drel_gene = []

    for i in range(len(cancer_gene)):
        symbol = cancer_gene.index[i].split("_")[1]
        drel_gene.append(symbol)

    selected_drel = []

    for i in range(len(X.columns)):
        symbol_name = X.columns[i].split("|")[-1]
        if symbol_name in drel_gene:
            selected_drel.append(X.columns[i])

    # indirectly related gene
    irel_gene = cancer_gene.values[cancer_gene.values != None]
    irel_gene = irel_gene.astype(np.int)
    irel_gene = list(set(irel_gene))

    selected_irel = []

    for i in range(len(X.columns)):
        entrez_num = int(re.findall("\d+", X.columns[i])[0])
        if entrez_num in irel_gene:
            selected_irel.append(X.columns[i])

    # cancer-related gene selection & data split
    selected_rel = list(set(selected_drel) | set(selected_irel))
    X_cancer = X[selected_rel]

    ### log transformation
    X_log = log_transformation(X = X_cancer)

    ### scaling
    X_scaled = scaling(X = X_log, y = y)

    data = pd.concat([X_scaled, y], axis = 1)
    X_features = data.iloc[:, :-1]
    y_target = data.iloc[:, -1]

    return data, X_features, y_target