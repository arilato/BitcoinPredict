import grid_search
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataMerge

'''
Given a model, we will simulate investing 10,000$ into Bitcoin through Coinbase.
There is a 1.5 percent charge fee for instant buys and sells. Our strategy is as follows:
Binary Classification:
    On day i:
        If predict increase, invest 90% of assets
        If predict decrease, sell 100% of assets
Regression:
    On day i:
        p = prediction of price change as a ratio, p = (0, inf)
        Buy and sell such that the ratio of assets invested is max(0, 1-e^-(2(p-0.8))
This will act as a final 'validation stage'
'''
def simulate(model, classify):
    e = 2.71828
    model, X_test, Y_test, scaler = grid_search.grid_search_init(classify=classify, model_name=model)
    money = 10000 #starting money, units are USD
    invested = 0 #units are USD
    total_money = [10000]
    money_data = [10000]
    invested_data = [0]
    naive_invest = [10000]
    if classify == True: #we need actual values from regression
        scaler = dataMerge.mergeData(y_option=2)
        df = pd.read_csv('Data.csv')
        data = df.values
        Y_test = data[255:,data.shape[1]-1]

    

    for i, j in zip(X_test, Y_test):
        pred_change = model.predict(i.reshape(1,-1))
        if classify == False:
            pred_change = scaler.inverse_transform(pred_change)
            toinvest_ratio = 1-e**(-2*(pred_change-0.8))
            toinvest = money*toinvest_ratio/(1-toinvest_ratio)
        else:
            if pred_change == 1: toinvest = 0.9 * (invested+money)
            else:
                print(-1)
                toinvest = 0

        real_change = scaler.inverse_transform(j.reshape(1,-1))[0][0]
        money -= (toinvest - invested) * 0.985
        naive_invest.append(naive_invest[-1] * real_change)
        invested = toinvest
        invested *= real_change
        total_money.append(money+invested)
        if classify == False:
            money_data.extend(money)
        else:
            money_data.append(money)
        invested_data.append(invested)

    plt.title("Simulation of running tuned Gradient Boost Regressor model on Bitcoin validation dataset")
    plt.plot(total_money)
    plt.plot(money_data)
    plt.plot(invested_data)
    plt.plot(naive_invest)
    plt.legend(['total money', 'uninvested money', 'invested money', 'naive total money'], loc='upper left')
    plt.xlabel('Day')
    plt.ylabel('Total Money')
    plt.show()
    print(money+invested)

simulate('MLPR', False)
