from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
import numpy as np

def GBC_model():
    params = {'n_estimators':range(10, 51, 10), 'learning_rate':np.linspace(0.001, 0.1, 5),
        'max_depth':range(1, 21, 4), 'min_samples_leaf':range(1, 21, 4)}
    model = GradientBoostingClassifier(min_samples_split=30,max_features='sqrt',
                                       subsample=0.8,random_state=189)
    return model, params

def GBR_model():
    params = {'n_estimators':range(10, 51, 10), 'learning_rate':np.linspace(0.001, 0.1, 5),
        'max_depth':range(1, 21, 4), 'min_samples_leaf':range(1, 21, 4)}
    model = GradientBoostingRegressor(min_samples_split=30,max_features='sqrt',
                                       subsample=0.8,random_state=189)
    return model, params

def SVM_model():
    model = SVC(random_state=189)
    params = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'degree':range(3, 6, 1),
        'C':np.linspace(0.01, 1, 10)}
    return model, params

def SVR_model():
    model = SVR(random_state=189)
    params = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'degree':range(3, 6, 1),
        'C':np.linspace(0.01, 1, 10)}
    return model, params

def MLPC_model():
    model = MLPClassifier(random_state=189)
    params = {'hidden_layer_sizes':[(75, 75, 75), (100, 100, 100), (75, 75, 75, 75), (100, 100, 100, 100)],
              'activation':['relu', 'tanh'], 'alpha':np.linspace(0.000001, 0.001, 10)}
    return model, params

def MLPR_model():
    model = MLPRegressor(random_state=189)
    params = {'hidden_layer_sizes':[(75, 75, 75), (100, 100, 100), (75, 75, 75, 75), (100, 100, 100, 100)],
        'activation':['relu', 'tanh'], 'alpha':np.linspace(0.000001, 0.001, 10)}
    return model, params

