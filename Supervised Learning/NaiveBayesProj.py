import pandas as pd

#Step 1: Load the files and check for missing values
read_data_train = pd.read_csv("train_80.csv")
read_data_test = pd.read_csv("test_20.csv")

print(read_data_train.isnull().sum().sort_values(ascending=False))

