import pandas as pd
import csv

column_names = ['open','high','low','close']
df1 = pd.read_csv('testing_data.csv', header = None, names = column_names)
column_names2 = ['action']
df2 = pd.read_csv('output.csv', header = None, names = column_names2)

value = pd.Series(df1['open'], index = df1.index)
print(df2)
money = 0
handle = 0
i=0
L = len(value)
print(df2.loc[3].item())
while i < L-1:
	if handle < -1 or handle > 1:
		print("error")
		break
	if df2.iloc[i].item() == 1:
		money -= value.iloc[i+1].item()
		handle += 1
	elif df2.iloc[i].item() == -1:
		money += value.iloc[i+1].item()
		handle -= 1
	i += 1
	print(money)
print(handle)
print(money)