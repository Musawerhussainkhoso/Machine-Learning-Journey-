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
'''
# Step: Drop the four columns with too many missing values
heavy = ['Sunshine', 'Evaporation', 'Cloud9am', 'Cloud3pm']

read_data_train = read_data_train.drop(columns=heavy)
read_data_test = read_data_test.drop(columns=heavy)

# Check: these columns should no longer exist
print("Train shape:", read_data_train.shape)
print("Test shape:", read_data_test.shape)
print(read_data_train.columns.tolist())

'''
# Step: Fill missing values with median 
num_cols = read_data_train.select_dtypes(include='number').columns

for c in num_cols:
    med = read_data_train[c].median()   # median sirf TRAIN se
    read_data_train[c] = read_data_train[c].fillna(med)
    read_data_test[c] = read_data_test[c].fillna(med)

print(read_data_train[['Sunshine', 'Evaporation', 'Cloud9am', 'Cloud3pm']].isnull().sum())

# Step: Drop rows where the target (RainTomorrow) is missing
read_data_train = read_data_train.dropna(subset=['RainTomorrow'])
read_data_test = read_data_test.dropna(subset=['RainTomorrow'])