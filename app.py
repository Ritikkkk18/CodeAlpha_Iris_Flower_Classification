import streamlit as st
import numpy as np
import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# ================= MODEL (UNCHANGED LOGIC) =================
iris = load_iris()
model = LogisticRegression(max_iter=200).fit(iris.data, iris.target)

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)

# ================= HEADER =================
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🌸 Iris Flower Prediction App</h1>
    <p style='text-align: center;'>Predict the species of an Iris flower using Machine Learning</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ================= SIDEBAR =================
st.sidebar.header("🌼 Enter Flower Measurements")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width  = st.sidebar.slider("Sepal Width (cm)", 2.0, 5.0, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width  = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# ================= DISPLAY INPUT DATA =================
st.subheader("📊 Selected Input Values")
st.write(f"""
- **Sepal Length:** {sepal_length} cm  
- **Sepal Width:** {sepal_width} cm  
- **Petal Length:** {petal_length} cm  
- **Petal Width:** {petal_width} cm
""")

# ================= PREDICTION =================
if st.button("🔍 Predict Flower Species"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction_index = model.predict(input_data)[0]
    prediction = iris.target_names[prediction_index]

    probabilities = model.predict_proba(input_data)[0]
    confidence = np.max(probabilities) * 100

    st.success(f"🌺 **Predicted Iris Species: {prediction.upper()}**")
    st.info(f"📈 Prediction Confidence: **{confidence:.2f}%**")


