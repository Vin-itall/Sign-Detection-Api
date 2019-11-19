import pandas
import numpy as np
from sklearn.externals.joblib import dump, load

X=[]
def preprocess(Dataframe):
    max_seq = 324
    idx = [i for i in range(2, 52)]
    del idx[2::3]
    no_of_features = len(idx)
    df = Dataframe
    # print(df)
    # print(Dataframe)
    arr = np.array(df.values)
    arr = arr[:, idx]
    X = arr.tolist()
    if len(X) < max_seq:
        start = len(X)
        for i in range(start, max_seq):
            X.append([0] * no_of_features)
    else:
        X = X[:max_seq]
    X = np.array(X)
    # Scale
    sc = load('std_scaler.bin')  # Make sure to have this file
    X = sc.transform(X)
    X = np.expand_dims(X, axis=2)
    return X