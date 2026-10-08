import os
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Insurance Analytics", layout="wide")

RED = '#B3123C'   # dashboard ka main rang

# ---------- 1. Data load ----------
@st.cache_data   # data baar baar load nahi hota, dashboard tez rehta hai
def load_data():
    path = os.path.join(os.path.dirname(__file__), 'insurance_80_percent_train.xlsx')
    df = pd.read_excel(path)
    df['age_group'] = pd.cut(df['age'], bins=[17, 30, 40, 50, 64],
                             labels=['18-30', '31-40', '41-50', '51-64'])
    return df

data = load_data()

# ---------- 2. Sidebar filters ----------
st.sidebar.header("Filters")
smoker = st.sidebar.multiselect("Smoker", data['smoker'].unique(),
                                default=list(data['smoker'].unique()))
region = st.sidebar.multiselect("Region", data['region'].unique(),
                                default=list(data['region'].unique()))
age_min, age_max = int(data['age'].min()), int(data['age'].max())
age_range = st.sidebar.slider("Age", age_min, age_max, (age_min, age_max))

df = data[
    data['smoker'].isin(smoker)
    & data['region'].isin(region)
    & data['age'].between(*age_range)
]

# ---------- 3. Title + KPI cards ----------
st.title("Insurance Charges Analysis")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Patients", f"{len(df):,}")
k2.metric("Avg Charges", f"${df['charges'].mean():,.0f}")
k3.metric("Avg BMI", f"{df['bmi'].mean():.1f}")
k4.metric("Smokers %", f"{(df['smoker'] == 'yes').mean() * 100:.1f}%")

# ---------- 4. Charts ----------
c1, c2 = st.columns(2)

# Age group vs average charges
age_avg = df.groupby('age_group', observed=True)['charges'].mean().reset_index()
fig1 = px.bar(age_avg, x='age_group', y='charges', title="Avg Charges by Age Group",
              color_discrete_sequence=[RED])
c1.plotly_chart(fig1, use_container_width=True)

# BMI vs charges (smoker ke rang ke saath)
fig2 = px.scatter(df, x='bmi', y='charges', color='smoker', opacity=0.6,
                  title="BMI vs Charges",
                  color_discrete_map={'yes': RED, 'no': '#95A5A6'})
c2.plotly_chart(fig2, use_container_width=True)

c3, c4 = st.columns(2)

# Smoker vs charges
fig3 = px.box(df, x='smoker', y='charges', color='smoker', title="Smoker vs Charges",
              color_discrete_map={'yes': RED, 'no': '#95A5A6'})
c3.plotly_chart(fig3, use_container_width=True)

# Region vs average charges
reg_avg = df.groupby('region')['charges'].mean().reset_index()
fig4 = px.bar(reg_avg, x='region', y='charges', title="Avg Charges by Region",
              color_discrete_sequence=[RED])
c4.plotly_chart(fig4, use_container_width=True)