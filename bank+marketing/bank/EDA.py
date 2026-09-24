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
#Step 6: Outlier Detection
for col in num_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")
'''
Step 7: Handle the pdays Sentinel Value (-1)
-1 in pdays doesn't mean "negative one day" — it's 
a placeholder meaning "never contacted before." 
Left as-is, it will skew your stats and corrupt anything downstream
 (correlation, regression, etc.):
'''
df['was_contacted_before'] = (df['pdays'] != -1).astype(int)
df['pdays_clean'] = df['pdays'].replace(-1, np.nan)   
'''
Step 8: Bivariate / Multivariate Analysis
Correlation heatmap (numeric features):
''' 
plt.figure(figsize=(9,7))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

#Numeric features vs target y:
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='y', y=col, data=df)
    plt.title(f"{col} vs y")
    plt.show()
#Categorical features vs target y:
cat_cols_no_target = ['job','marital','education','default','housing','loan','contact','month','poutcome']

for col in cat_cols_no_target:
    plt.figure(figsize=(8,4))
    sns.countplot(x=col, hue='y', data=df)
    plt.xticks(rotation=45)
    plt.title(f"{col} vs y")
    plt.tight_layout()
    plt.show()    