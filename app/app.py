import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    color: #1f4e79;
    text-align: center;
}

.stButton>button {
    width: 100%;
    background-color: #1f77b4;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
}

</style>
""", unsafe_allow_html=True)


st.title("🏠 House Price Prediction Dashboard")

st.write("Predict house prices using Linear Regression")


df = pd.read_csv("Housing.csv")


encoder = LabelEncoder()

categorical_columns = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea',
    'furnishingstatus'
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])


X = df.drop("price", axis=1)

y = df["price"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LinearRegression()

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


st.sidebar.header("Enter House Details")


area = st.sidebar.number_input(
    "Area",
    min_value=500,
    max_value=20000,
    value=5000
)

bedrooms = st.sidebar.slider(
    "Bedrooms",
    1,
    10,
    3
)

bathrooms = st.sidebar.slider(
    "Bathrooms",
    1,
    10,
    2
)

stories = st.sidebar.slider(
    "Stories",
    1,
    5,
    2
)

mainroad = st.sidebar.selectbox(
    "Main Road Access",
    ["Yes", "No"]
)

guestroom = st.sidebar.selectbox(
    "Guest Room",
    ["Yes", "No"]
)

basement = st.sidebar.selectbox(
    "Basement",
    ["Yes", "No"]
)

hotwaterheating = st.sidebar.selectbox(
    "Hot Water Heating",
    ["Yes", "No"]
)

airconditioning = st.sidebar.selectbox(
    "Air Conditioning",
    ["Yes", "No"]
)

parking = st.sidebar.slider(
    "Parking Spaces",
    0,
    5,
    1
)

prefarea = st.sidebar.selectbox(
    "Preferred Area",
    ["Yes", "No"]
)

furnishingstatus = st.sidebar.selectbox(
    "Furnishing Status",
    ["Furnished", "Semi-Furnished", "Unfurnished"]
)


mainroad = 1 if mainroad == "Yes" else 0

guestroom = 1 if guestroom == "Yes" else 0

basement = 1 if basement == "Yes" else 0

hotwaterheating = 1 if hotwaterheating == "Yes" else 0

airconditioning = 1 if airconditioning == "Yes" else 0

prefarea = 1 if prefarea == "Yes" else 0


if furnishingstatus == "Furnished":
    furnishingstatus = 0

elif furnishingstatus == "Semi-Furnished":
    furnishingstatus = 1

else:
    furnishingstatus = 2


if st.sidebar.button("Predict House Price"):

    input_data = np.array([[
        area,
        bedrooms,
        bathrooms,
        stories,
        mainroad,
        guestroom,
        basement,
        hotwaterheating,
        airconditioning,
        parking,
        prefarea,
        furnishingstatus
    ]])

    prediction = model.predict(input_data)

    st.success(f"🏡 Predicted House Price: ₹ {prediction[0]:,.2f}")


mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)


st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{mae:,.2f}")

col2.metric("MSE", f"{mse:,.2f}")

col3.metric("R2 Score", f"{r2:.2f}")


st.subheader("📁 Dataset Preview")

st.dataframe(df.head())


st.subheader("📈 House Price Distribution")

fig1, ax1 = plt.subplots(figsize=(8,5))

sns.histplot(df['price'], kde=True, ax=ax1)

ax1.set_title("House Price Distribution")

st.pyplot(fig1)


st.subheader("🔥 Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(12,8))

correlation = df.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    ax=ax2
)

ax2.set_title("Correlation Heatmap")

st.pyplot(fig2)


st.subheader("📌 Actual vs Predicted Prices")

fig3, ax3 = plt.subplots(figsize=(8,6))

ax3.scatter(y_test, y_pred)

ax3.set_xlabel("Actual Prices")

ax3.set_ylabel("Predicted Prices")

ax3.set_title("Actual vs Predicted Prices")

st.pyplot(fig3)


importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

importance = importance.sort_values(
    by="Coefficient",
    ascending=False
)


st.subheader("⭐ Feature Importance")

fig4, ax4 = plt.subplots(figsize=(10,6))

sns.barplot(
    x="Coefficient",
    y="Feature",
    data=importance,
    ax=ax4
)

ax4.set_title("Feature Importance")

st.pyplot(fig4)


st.subheader("📋 Actual vs Predicted Table")

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

st.dataframe(comparison.head(20))


import pickle

pickle.dump(model, open("linear_model.pkl", "wb"))


st.markdown("---")

st.markdown(
    "Developed using Streamlit and Linear Regression"
)