import numpy as np
import pandas as pd
from sklearn.feature_selection import *

helmet_data = np.genfromtxt ('helmet.csv', delimiter=",")
face_data = np.genfromtxt ('face.csv', delimiter=",")

data = np.concatenate((helmet_data, face_data), 0)
np.random.shuffle(data) #shuffle the tuples

n_features = data.shape[1] - 9

X = data[:, 8:-1]
y = data[:, -1]

gain = mutual_info_classif(X, y, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)
m = np.ndarray((n_features, 2), dtype = object)
m[:,0] = int(n_features)
m[:,1] = float(n_features)
for i in np.arange(n_features):
	m[i, 0], m[i, 1] = 8+i, np.around(gain[i], decimals=4)

m = m[(-m[:, 1]).argsort()]
df = pd.DataFrame(m)
df.to_csv("InfoGain.csv", index=False, header=False)
