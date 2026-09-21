import pandas as pd

df = pd.read_csv(r"D:\ML journey 2027\bank+marketing\bank\bank.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())