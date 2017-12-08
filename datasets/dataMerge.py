import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

days_in_month = [0, 23, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]

def dateToHours(date): #start reference from 2016-12-08
    year = int(date[0:4])
    month = int(date[5:7])
    day = int(date[8:10])
    hour = int(date[11:13])
    if year == 2016: month -= 12
    tot_days = day + sum(days_in_month[0:month+1]) - 8
    return hour + tot_days * 24

def stackToX(file, X):
    df = pd.read_csv(file, header=None)
    tmp = df.values
    for i in range(len(tmp)):
        tmp[i][0] = dateToHours(tmp[i][0])
    toadd = []
    for i in range(len(tmp)):
        if len(toadd) == 365:
            break
        if tmp[i][0] >= X[len(toadd)][0]:
            toadd.append(tmp[i][1])
    return np.hstack([X, np.array([toadd]).T])


df = pd.read_csv('market-price.csv', header=None)
X = df.values
for i in range(len(X)):
    X[i][0] = dateToHours(X[i][0])
Y = [1 if X[i+1][1] > X[i][1] else -1 for i in range(len(X)-1)] #1 if price increase, -1 if price decrease

feature_files = ['bip-9-segwit.csv', 'cost-per-transaction-percent.csv', 'cost-per-transaction.csv',
                 'estimated-transaction-volume-usd.csv', 'estimated-transaction-volume.csv',
                 'hash-rate.csv', 'market-cap.csv', 'n-transactions-excluding-popular.csv',
                 'n-transactions-total.csv', 'n-transactions.csv', 'n-unique-addresses.csv',
                 'output-volume.csv', 'total-bitcoins.csv', 'trade-volume.csv',
                 'utxo-count.csv']
for i in feature_files:
    X = stackToX(i, X)

X = np.delete(X, len(X)-1, 0)
#standardize data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X = np.hstack([X, np.array([Y]).T])

df = pd.DataFrame(data=X)
df.to_csv('~/Academic/CS189/projects/Data.csv')









