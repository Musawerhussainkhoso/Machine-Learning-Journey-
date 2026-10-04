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

#x = data[['age', 'bmi', 'children', 'sex', 'smoker', 'region']]
#y = data['charges']
#numerical features
num_features = ['age', 'bmi', 'children']

# 1 row x 3 columns ka canvas
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, col in zip(axes, num_features):
    sns.regplot(
        data=data,
        x=col,
        y='charges',
        scatter_kws={'alpha': 0.5},   # points thode transparent
        line_kws={'color': 'red'},    # best-fit line ka rang
        x_jitter=0.1 if col == 'children' else 0,  # children discrete hai, isliye halka jitter
        ax=ax
    )
    ax.set_title(f'{col} vs charges')

plt.tight_layout()
plt.show()