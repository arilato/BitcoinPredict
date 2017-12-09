from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import dataMerge
import models
import pandas as pd
import numpy as np

def MSE(y1, y2):
    mse = 0
    for i, j in zip(y1, y2):
        mse += (i-j)**2
    return mse**0.5

def grid_search(model, params, trainx, trainy):
    gsearch = GridSearchCV(estimator = model, param_grid = params)
    gsearch.fit(trainx, trainy)
    return gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_

scaler = dataMerge.mergeData(y_option=2)

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

'''num_splits = 4
kf = KFold(n_splits=num_splits)
KFold_train = []
KFold_test = []
for train, test in kf.split(data):
    KFold_train.append([train])
    KFold_test.append([test])'''
X_train = X[0:255]
Y_train = Y[0:255]
X_test = X[255:]
Y_test = Y[255:]


model, params = models.MLPR_model()
scores, best_params, best_score = grid_search(model ,params, X_train, Y_train)
model.set_params(**best_params)
model.fit(X_train,Y_train)
y_test_pred = model.predict(X_test)
print(MSE(scaler.inverse_transform(Y_test), scaler.inverse_transform(y_test_pred)))

print(best_score)
print(best_params)
