import glob
import json
import os
import random

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)


df = pd.read_csv('smiles_pld.csv')
voxels = df['npz'].values
ids = [ voxel.split("voxels_50/")[-1].split(".npz")[0] for voxel in voxels]

X = []
Y = []

for idx in df.index:

    pld = df['pld'][idx]
    Y.append(pld)

    npz= np.load(df['npz'][idx])
    X.append(npz['X'])

X = np.asarray(X).astype(np.float32)
X = np.delete(X,6,4)
Y = np.asarray(Y).astype(np.float32)
print(X.shape)
print(X[0].shape)
X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(X, Y, ids, test_size=0.2, random_state=SEED)
#
np.savez_compressed('./data/train_data.npz',X=X_train,Y=y_train,ID=p_train)
np.savez_compressed('./data/unseen_data.npz',X=X_test,Y=y_test,ID=p_test)
