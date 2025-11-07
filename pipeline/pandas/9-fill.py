#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove the 'Weighted_Price' column
df = df.drop(columns=['Weighted_Price'])

# Fill missing Close values with the previous row's value
df['Close'].fillna(method='ffill', inplace=True)

# Fill missing High, Low, and Open values with the same row’s Close value
for col in ['High', 'Low', 'Open']:
    df[col].fillna(df['Close'], inplace=True)

# Fill missing Volume_(BTC) and Volume_(Currency) with 0
df[['Volume_(BTC)', 'Volume_(Currency)']] = df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0)

print(df.head())
print(df.tail())
