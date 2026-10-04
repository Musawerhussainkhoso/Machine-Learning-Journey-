import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Load the Excel dataset
file_path = os.path.join(os.path.dirname(__file__), 'insurance_80_percent_train.xlsx')
data = pd.read_excel(file_path)
print(data.head())
print(data.tail())
print(data.info())