import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# --- FIX: correct delimiter + strip quotes ---
df = pd.read_csv(r"D:\ML journey 2027\bank+marketing\bank\bank.csv", sep=';')
df.columns = [c.strip().replace('"', '') for c in df.columns]
for c in df.select_dtypes(include='object').columns:
    df[c] = df[c].str.replace('"', '', regex=False)

print(df.columns.tolist())  # sanity check — should show 17 clean column names

# Step 2: Initial Inspection
print(df.shape)
print(df.head(10))
print(df.tail())
print(df.columns)
print(df.dtypes)
df.info()
print(df.describe())
print(df.describe(include='object'))

# Step 3: Check Missing Values
print(df.isnull().sum())
print(df.isnull().mean() * 100)

plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Value Heatmap")
plt.show()

# Step 5: Univariate Analysis
num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

for col in num_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    sns.histplot(df[col], kde=True, ax=axes[0])
    axes[0].set_title(f"Distribution of {col}")
    sns.boxplot(x=df[col], ax=axes[1])
    axes[1].set_title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()