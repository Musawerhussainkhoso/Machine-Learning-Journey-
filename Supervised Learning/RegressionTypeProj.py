import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Dataset

file_path = os.path.join(
    os.path.dirname(__file__),
    'insurance_80_percent_train.xlsx'
)

data = pd.read_excel(file_path)

print(data.head())
print(data.tail())
print(data.info())

# 2. Visualize Numerical Features

num_features = ['age', 'bmi', 'children']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, col in zip(axes, num_features):

    sns.regplot(#Yeh regplot scatter plot banata hai aur sath mein best-fit line bhi khud laga deta hai.
        data=data,
        x=col,
        y='charges',
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'red'},
        x_jitter=0.1 if col == 'children' else 0,
        ax=ax
    )

    ax.set_title(f'{col} vs charges')

plt.tight_layout()#graphs ke labels ko aapas mein overlap hone se bachata hai.
plt.show()

# 3. Encode Categorical Features

data = pd.get_dummies(
    data,
    columns=['sex', 'smoker', 'region'],
    drop_first=True
)

# 4. Feature Engineering

data['bmi_smoker'] = data['bmi'] * data['smoker_yes']
data['bmi_obese'] = (data['bmi'] >= 30).astype(int)

# 5. Separate Features and Target

X = data.drop('charges', axis=1)

y = np.log1p(data['charges'])

# 6. Cross Validation
model = LinearRegression()

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring='r2'
)

# 7. Results
print("R² Scores:", scores)
print("Mean R²:", scores.mean())
print("Standard Deviation:", scores.std())