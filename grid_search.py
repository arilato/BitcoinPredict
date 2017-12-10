from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import dataMerge
import models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab

def MSE(y1, y2):
    mse = 0
    for i, j in zip(y1, y2):
        mse += (i-j)**2
    return mse**0.5

def grid_search(model, params, trainx, trainy):
    gsearch = GridSearchCV(estimator = model, param_grid = params)
    gsearch.fit(trainx, trainy)
    return gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_

def plot_grid_search(grid_scores, best_params, best_score):
    keylist = [key for key in best_params]
    
    for key in keylist:
        X = [] #values of the key
        Y = [] #mean when run on values of the key
        for i in grid_scores:
            flag = False
            for k in i[0]:
                if k != key and i[0][k] != best_params[k]:
                    flag = True
                    break
            if flag == False:
                X.append(i[0][key])
                Y.append(i[1])
        pylab.figure(1)
        x = range(len(X))
        pylab.xticks(x, X)
        pylab.plot(x, Y)
        pylab.xlabel('Values of ' + str(key))
        pylab.ylabel('Mean Accuracy')
        pylab.show()




def grid_search_init(classify, model_name):
    if classify == True:
        dataMerge.mergeData(y_option=1)
    else:
        scaler = dataMerge.mergeData(y_option=2)

    df = pd.read_csv('Data.csv')
    data = df.values
    #np.random.shuffle(data)
    X = data[:,0:data.shape[1]-2]
    Y = data[:,data.shape[1]-1]
    #predict price change based on past prelen days worth of data
    '''prelen = 7
    X = [np.hstack([X[i-3], X[i-2], X[i-1], X[i]]) for i in range(prelen, len(X))]
        Y = np.delete(Y, range(prelen), 0)'''

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
    X_train = X[0:510]
    Y_train = Y[0:510]
    X_test = X[510:]
    Y_test = Y[510:]

    if model_name == "MLPR":
        model, params = models.MLPR_model()
    if model_name == "GBR":
        model, params = models.GBR_model()
    if model_name == "GBC":
        model, params = models.GBC_model()
    if model_name == "MLPC":
        model, params = models.MLPC_model()

    scores, best_params, best_score = grid_search(model ,params, X_train, Y_train)
#plot_grid_search(scores, best_params, best_score)
    model.set_params(**best_params)
    model.fit(X_train,Y_train)
    if classify == True:
        scaler = 0
    return model, X_test, Y_test, scaler
