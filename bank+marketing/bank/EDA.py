import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"D:\ML journey 2027\bank+marketing\bank\bank.csv")
#Step 2: Initial Inspection
df.shape           # rows, columns
df.head(10)        # first 10 rows
df.tail()           
df.columns
df.dtypes           # data types of each column
df.info()           # nulls + dtypes + memory usage
df.describe()       # stats for numeric columns
df.describe(include='object')  # stats for categorical columns