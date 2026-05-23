import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore

st.title("🏡 California House Price Prediction Dashboard")
st.write("Adjust the neighborhood features to see the estimated median house price.")

# Load the same clean data process to train the dashboard model quickly
# Fixed: Using old decorator layout for compatibility
@st.cache
def load_and_train():
    df = pd.read_csv('housing.csv')
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    df = df.drop(columns=['ocean_proximity'])
    
    X = df.drop(columns=['median_house_value'])
    y = df['median_house_value']
    
    model = LinearRegression().fit(X, y)
    return model

model = load_and_train()

# Layout Sidebars for user configuration inputs
st.sidebar.header("🔧 Adjust Housing Features")
med_inc = st.sidebar.slider("Median Income (in tens of thousands)", 0.5, 15.0, 3.5)
age = st.sidebar.slider("Median House Age", 1, 52, 28)
rooms = st.sidebar.slider("Total Rooms in Block", 10, 40000, 2500)
bedrooms = st.sidebar.slider("Total Bedrooms in Block", 1, 6000, 500)
pop = st.sidebar.slider("Population in Block", 3, 35000, 1400)
households = st.sidebar.slider("Total Households", 1, 6000, 500)
lat = st.sidebar.slider("Latitude coordinate", 32.5, 42.0, 35.5)
lon = st.sidebar.slider("Longitude coordinate", -124.3, -114.3, -119.5)

# Input construction (Order must match the columns in X exactly)
user_input = np.array([[lon, lat, age, rooms, bedrooms, pop, households, med_inc]])
prediction = model.predict(user_input)[0]

# Display results clearly
st.subheader("📊 Model Price Estimate Output")
if prediction < 0:
    st.error("The selected combination is unrealistic for a calculated evaluation.")
else:
    st.metric(label="Estimated Median House Value", value=f"${prediction:,.2f}")