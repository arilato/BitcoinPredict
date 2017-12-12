import urllib.request
import pandas as pd
import time
import datetime
import os

while(1):
    urllib.request.urlretrieve ("https://www.bitstamp.net/api/ticker/", "tick.json")
    df = pd.read_csv("tick.json", header=None)
    last = df.values[0][1].split()[1]
    last = float(last[1:len(last)-1])
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ": ", last)
    if last < 15000:
        while(1):
            os.system('say "It\'s under 15000"')
            time.sleep(15)
    if last > 18000:
        while(1):
            os.system('say "It\'s over 18000"')
            time.sleep(15)
    time.sleep(30)




