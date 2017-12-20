import urllib.request
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor as GBC
import os
import gdax

#constants
SAVE_EVERY_N_MINUTES = 10
SAVE_PATH = "monitor_data_10_min_interval.csv"
public_client = gdax.PublicClient()
last_n = [5, 10, 30, 60]
FEATURES = [['avprice_last'+str(i), 'avvolume_last'+str(i), 'maxprice_last'+str(i),
            'maxvolume_last'+str(i), 'minprice_last'+str(i),'minvolume_last'+str(i),
            'pricevar_last'+str(i), 'volvar_last'+str(i)] for i in last_n]
FEATURES = np.array(FEATURES).flatten().tolist()
FEATURES.extend(['curprice', 'in_10'])


def transform_feature_helper(prices, volumes, last_n):
    return [sum(prices[0:last_n])/last_n,
            sum(volumes[0:last_n])/last_n,
            max(prices[0:last_n]),
            max(volumes[0:last_n]),
            min(prices[0:last_n]),
            min(volumes[0:last_n]),
            np.var(prices),
            np.var(volumes)]

def transform_features(prices, volumes):
    toret = np.hstack([transform_feature_helper(prices, volumes, i) for i in last_n])
    toret = np.hstack([toret, prices[-1]])
    return toret

def get_price():
    trades = public_client.get_product_trades(product_id='BTC-USD')[0:10]
    return sum([float(trades[i]['price']) for i in range(10)])/10

'''features - avprice, avvolume, maxprice, maxvolume, minprice, minvolume, price variance, volume variance
of last 5 minutes, last 10 minutes, last 30 minutes, last hour, curprice
total = 4 * 8 + 1 = 33

we want to predict the price in 10 minutes, and the price in 30 minutes
'''
xtrain, ytrain = [], []
prices = [0 for i in range(60)]
volumes = [0 for i in range(60)]
tick = 0

#retrieve old data
saved = pd.read_csv(SAVE_PATH).values[:,1:]
oldx, oldy = [], []
oldx.extend(saved[:,0:saved.shape[1]-1])
oldy.extend(saved[:,saved.shape[1]-1])
oldy = np.array(oldy)
oldy = oldy.reshape(-1, 1)
olddata = np.hstack([oldx, oldy])

while(1): #every minute, get last 10 trades, update data
    trades = public_client.get_product_trades(product_id='BTC-USD')
    trades = trades[0:min(10, len(trades))]
    del prices[0]
    del volumes[0]
    prices.append(sum([float(trades[i]['price']) for i in range(10)])/10)
    volumes.append(sum([float(trades[i]['size']) for i in range(10)])/10)
    xtrain.append(transform_features(prices, volumes))
    if (tick >= 70): #start generating y 60 minutes in
        ytrain.append([get_price()])
        if (tick % 10 == 1): #save
            print(np.array(xtrain[60:len(xtrain)-10]).shape)
            print(np.array(ytrain).shape)
            print(np.hstack([xtrain[60:len(xtrain)-10], ytrain]).shape)
            print(olddata.shape)
            df = pd.DataFrame(np.vstack([olddata, np.hstack([xtrain[60:len(xtrain)-10], ytrain])]), columns=FEATURES)
            df.to_csv(SAVE_PATH)
            print("Data saved!")
    price = get_price()
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ": ", price)
    tick += 1
    time.sleep(60)




