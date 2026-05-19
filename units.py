import numpy as np
import warnings
from mat4py import loadmat

warnings.filterwarnings('ignore')

def load_data(path):
    try:
        print(path)
        data = loadmat(path)
    except:
        data = np.load(path, allow_pickle=True)
    try:
        data = np.array(data['trandata'])
    except:
        X = np.array(data['X'])
        y = np.array(data['y'])
        if type(y[0]) == np.ndarray:
            yy = []
            for i in range(len(y)):
                yy.append(y[i][0])
            y = np.array(yy)
        return X, y

    X = data[:,0:-1]
    y = data[:,-1]
    if max(y) != 1:
        y -= min(y)
    if sum(y) > len(y) / 2:
        for i in range(len(y)):
            y[i] = 1 if y[i] == 0 else 0
    return X, y
