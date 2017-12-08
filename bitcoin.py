from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

df = pd.read_csv('Data.csv')
data = df.values
np.random.shuffle(data)
X = data[:,0:data.shape[1]-2]
Y = data[:,data.shape[1]-1]


#data matrix is days since 11/29, price at close
'''KFolds cross validation with 4 splits
    This may result in bias when unshuffled, as intuitively
    the bitcoin prices seem to be following a pattern that
    is repeated a few times.
    '''

num_splits = 4
kf = KFold(n_splits=num_splits)
KFold_train = np.array([[] for i in range(4)])
KFold_test = np.array([[] for i in range(4)])
i = 0
for train, test in kf.split(data):
    print(train)
    KFold_train[i] = [train]
    KFold_test[i] = [test]
    i += 1
X_train = []
Y_train = []
X_test = []
Y_test = []




