import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

st.title("🚗 MPG Prediction (Polynomial Regression)")

file = st.file_uploader("Upload Auto MPG CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    # Clean column names
    df.columns = df.columns.str.strip()

    st.write("📌 Columns in Dataset:")
    st.write(df.columns)

    st.subheader("🔧 Select Columns")

    # User selects columns manually
    target = st.selectbox("Select Target (MPG)", df.columns)

    feature1 = st.selectbox("Feature 1 (Displacement)", df.columns)
    feature2 = st.selectbox("Feature 2 (Horsepower)", df.columns)
    feature3 = st.selectbox("Feature 3 (Weight)", df.columns)
    feature4 = st.selectbox("Feature 4 (Acceleration)", df.columns)

    if st.button("Train Model"):
        try:
            X = df[[feature1, feature2, feature3, feature4]]
            y = df[target]

            # Handle missing values
            X = X.fillna(X.mean(numeric_only=True))
            y = y.fillna(y.mean())

            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Polynomial transformation
            poly = PolynomialFeatures(degree=2)
            X_train = poly.fit_transform(X_train)
            X_test = poly.transform(X_test)

            # Model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Prediction
            y_pred = model.predict(X_test)

            # Plot
            st.subheader("📉 Actual vs Predicted MPG")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

            # Store model for prediction
            st.session_state["model"] = model
            st.session_state["poly"] = poly
            st.session_state["features"] = [feature1, feature2, feature3, feature4]

            st.success("✅ Model trained successfully!")

        except Exception as e:
            st.error(f"Error: {e}")

    # Prediction section
    if "model" in st.session_state:
        st.subheader("🔮 Predict MPG")

        f1 = st.number_input("Feature 1 value")
        f2 = st.number_input("Feature 2 value")
        f3 = st.number_input("Feature 3 value")
        f4 = st.number_input("Feature 4 value")

        if st.button("Predict MPG"):
            data = np.array([[f1, f2, f3, f4]])
            data = st.session_state["poly"].transform(data)
            result = st.session_state["model"].predict(data)
            st.success(f"🚘 Predicted MPG: {result[0]:.2f}")

else:
    st.info("Please upload dataset to continue.")
