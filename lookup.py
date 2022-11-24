import pandas as pd

df = pd.read_csv('draft_1124.csv')

df = df.round(0).astype('int')

print(sum(df['0']))

df.to_csv('draft_1125.csv')
