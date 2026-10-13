import os 
import pandas as pd

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

# Step 1: Load the files and check for missing values
base_dir = os.path.dirname(os.path.abspath(__file__))

read_data_train = pd.read_csv(os.path.join(base_dir, 'train_80.csv'))
read_data_test = pd.read_csv(os.path.join(base_dir, 'test_20.csv'))
print("Training Data Info:")
print(read_data_train.info())
print("\nTest Data Info:")
print(read_data_test.info())
print("\nMissing values in training data:")
print(read_data_train.isnull().sum().sort_values(ascending=False))
print("\nMissing values in test data:")
print(read_data_test.isnull().sum().sort_values(ascending=False))
