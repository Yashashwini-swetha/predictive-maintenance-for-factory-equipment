import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ============================
# Load trained CNN model
# ============================

cnn_model = load_model("model/cnn_weights.hdf5")

# ============================
# Load dataset and preprocess (same as training)
# ============================

data = pd.read_csv("Dataset/predictive_maintenance.csv")

encoder1 = LabelEncoder()
encoder2 = LabelEncoder()

data['Product_ID'] = encoder1.fit_transform(data['Product_ID'].astype(str))
data['Type'] = encoder2.fit_transform(data['Type'].astype(str))

data.drop(['UDI', 'Target', 'Failure_Type'], axis=1, inplace=True)

scaler = MinMaxScaler()
scaler.fit(data.values)

# ============================
# Streamlit UI
# ============================

st.set_page_config(page_title="Predictive Maintenance AI", layout="centered")

st.title("🏭 Predictive Maintenance System")
st.subheader("Factory Equipment Health Monitor")

st.markdown("### Enter Machine Sensor Values")

# NEW INPUTS
product_id = st.number_input("Product ID (numeric)", value=1)
product_type = st.selectbox("Product Type", ["Low", "Medium", "High"])

# Map product type to encoded value
type_map = {"Low":0, "Medium":1, "High":2}
type_encoded = type_map[product_type]

air = st.number_input("Air Temperature (K)", value=300.0)
process = st.number_input("Process Temperature (K)", value=310.0)
speed = st.number_input("Rotational Speed (RPM)", value=1500.0)
torque = st.number_input("Torque (Nm)", value=40.0)
wear = st.number_input("Tool Wear (min)", value=100.0)

# ============================
# Prediction
# ============================

if st.button("🔍 Predict Machine Health"):

    input_data = np.array([[product_id, type_encoded, air, process, speed, torque, wear]])
    input_data = scaler.transform(input_data)
    input_data = input_data.reshape(1, input_data.shape[1], 1, 1)

    prediction = cnn_model.predict(input_data)
    result = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.markdown("### 🔎 Prediction Result")

    if result == 0:
        st.success("✅ Machine is Healthy")
    else:
        st.error("🚨 Failure Detected! Maintenance Required")

    st.info(f"Prediction Confidence: {confidence:.2f}%")
