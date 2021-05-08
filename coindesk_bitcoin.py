# API coindesk
import requests
import json
import pandas as pd


# Feed in the YYYY-MM-DD format
def get_bitcoin_prices(start, end):
    url = 'https://api.coindesk.com/v1/bpi/historical/close.json'
    first = '?start=' + start
    last = '&end=' + end
    full_url = url + first + last
    print(full_url)
    r = requests.get(full_url)
    data = json.loads(r.text)
    prices = data['bpi']
    return prices

prices = get_bitcoin_prices('2016-01-01', '2021-01-01')

df = pd.DataFrame(list(prices.items()), columns=['Date', 'BPI'])
df = df.set_index('Date')
df.to_csv('bpi.csv', index=True)
