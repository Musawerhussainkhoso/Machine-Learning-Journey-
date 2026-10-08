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
data.info()

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

# get_dummies se PEHLE chalayein (jab sex, smoker, region text mein hon)
cat_features = ['sex', 'smoker', 'region']
palettes = ['Set2', 'Set1', 'viridis']

sns.set_style('whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, col, pal in zip(axes, cat_features, palettes):# zip() teen lists ko jod kar har round mein ek ek item deta hai: (graph, column, rang ka set)


    # Scatter points (category ke ird-gird phaile hue)
    sns.stripplot(# Individual observations
        data=data, x=col, y='charges',
        hue=col, palette=pal, legend=False,
        jitter=0.25, alpha=0.6, size=5, ax=ax
    )

    #Average charges
    sns.pointplot(#Ye har category ke average charges ko black diamond ke saath show karta hai.
        data=data, x=col, y='charges',
        linestyle='none', color='black', marker='D',
        errorbar=None, ax=ax
    )

    ax.set_title(f'{col} vs charges', fontsize=14, fontweight='bold')
    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel('Charges', fontsize=12)

plt.tight_layout()
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

# 6. Train/Test Split (20% test alag rakh liya)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)#fit() model ko train karta hai; ye test data par prediction nahi karta.
y_pred = model.predict(X_test)#Trained model ko naye input features do aur usse predictions lo.
print("Test R²:", r2_score(y_test, y_pred))

print(
    "Test MSE:",
    mean_squared_error(y_test, y_pred)
)