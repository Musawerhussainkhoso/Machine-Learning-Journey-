import os 
import pandas as pd

#Step 1: Load the files and check for missing values
file_path = os.path.join(
    os.path.dirname(__file__),
    'train_80.csv'
)
read_data_train = pd.read_csv(file_path)

file_path = os.path.join(
    os.path.dirname(__file__),
    'test_20.csv'
)
read_data_test = pd.read_csv(file_path)



