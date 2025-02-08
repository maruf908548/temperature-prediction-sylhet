import streamlit as st
import pandas as pd
import gdown
import joblib

# Set Page Configuration
st.set_page_config(page_title="Temperature Prediction(Sylhet)", layout="wide")

# Background Image (Replace with your image URL)
background_image = "https://images.pexels.com/photos/209831/pexels-photo-209831.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"


st.markdown(
    f"""
    <style>
    .stApp {{
        background: url({background_image}) no-repeat center center fixed;
        background-size: cover;
    }}
    .sidebar .sidebar-content {{
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
    }}
    .main-container {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
    }}
    .stMetric {{
        font-size: 1.5em;
        color: #FF5733;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<h1 style='text-align: center; color: white;'>🌡 Temperature Prediction(Sylhet)</h1>", unsafe_allow_html=True)


st.sidebar.header("🌍 User Input Parameters")

# Input Fields (User-friendly labels)
date = st.sidebar.text_input("📅 Date (MM/DD/YYYY)", "11/1/2012")
time = st.sidebar.text_input("⏰ Time (HH:MM)", "01:00")

SO2 = st.sidebar.number_input("🛢 SO2 Level", value=None)
NO = st.sidebar.number_input("🚗 NO Level", value=4.55)
NO2 = st.sidebar.number_input("🌫 NO2 Level", value=7.56)
NOX = st.sidebar.number_input("🛑 NOX Level", value=12.15)
CO = st.sidebar.number_input("🔥 CO Level", value=None)
CO_8hr = st.sidebar.number_input("⌛ CO 8hr Level", value=None)
O3 = st.sidebar.number_input("🌍 O3 Level", value=1.06)
O3_8hr = st.sidebar.number_input("⌛ O3 8hr Level", value=None)
PM2_5 = st.sidebar.number_input("💨 PM2.5 Level", value=27.87)
PM10 = st.sidebar.number_input("🌫 PM10 Level", value=31.03)
Wind_Speed = st.sidebar.number_input("💨 Wind Speed", value=3.58)
Wind_Dir = st.sidebar.number_input("🧭 Wind Direction", value=203.18)
Temperature = st.sidebar.number_input("🌡 Temperature", value=25.22)
RH = st.sidebar.number_input("💦 Relative Humidity", value=84.16)
Solar_Rad = st.sidebar.number_input("☀ Solar Radiation", value=7.11)
BP = st.sidebar.number_input("🌡 Barometric Pressure", value=1092.60)
Rain = st.sidebar.number_input("🌧 Rainfall", value=0.03)
V_Wind_Speed = st.sidebar.number_input("💨 Vertical Wind Speed", value=1.35)

# Collect Inputs into Dictionary
input_data = {
    "Date": date,
    "Time": time,
    "SO2": SO2,
    "NO": NO,
    "NO2": NO2,
    "NOX": NOX,
    "CO": CO,
    "CO 8hr": CO_8hr,
    "O3": O3,
    "O3 8hr": O3_8hr,
    "PM2.5": PM2_5,
    "PM10": PM10,
    "Wind Speed": Wind_Speed,
    "Wind Dir": Wind_Dir,
    "Temperature": Temperature,
    "RH": RH,
    "Solar Rad": Solar_Rad,
    "BP": BP,
    "Rain": Rain,
    "V Wind Speed": V_Wind_Speed,
}

# Convert to DataFrame
input_data_df = pd.DataFrame([input_data])

# Load Model

# Google Drive file ID

# file_id = "1bPUWSsw2s1mNuJ-RrrkpHKe36dP9P-KT" 
# output_path = "model_with_pipeline.pkl"

# Download the file
# gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

# model = joblib.load(output_path)

# model = joblib.load("../models/model_with_pipeline.pkl")
model = joblib.load("../models/model_with_pipeline.pkl")

# Prediction
result = model.predict(input_data_df)

# Display Data and Result
st.subheader("📊 User Input Data")
st.dataframe(input_data_df.transpose(), use_container_width=True)


st.subheader("🔮 Predicted Temperature")
st.metric("Predicted Temperature (°C)", f"{result[0]:,.2f}")

st.markdown("</div>", unsafe_allow_html=True)
