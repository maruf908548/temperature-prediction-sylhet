import joblib
import pandas as pd
from utils import to_string

#predicting the temperature from a sample data
sample_data = {
    'Date': '11/1/2012', 
    'Time': '01:00', 
    'SO2': None, 
    'NO': 4.55, 
    'NO2': 7.56, 
    'NOX': 12.15, 
    'CO': None, 
    'CO 8hr': None, 
    'O3': 1.06, 
    'O3 8hr': None, 
    'PM2.5': 27.87, 
    'PM10': 31.03, 
    'Wind Speed': 3.58, 
    'Wind Dir': 203.18, 
    'Temperature': 25.22, 
    'RH': 84.16, 
    'Solar Rad': 7.11, 
    'BP': 1092.60, 
    'Rain': 0.03, 
    'V Wind Speed': 1.35
}
sample_data_df = pd.DataFrame([sample_data])
model = joblib.load('models/model_with_pipeline.pkl')
result = model.predict(sample_data_df)
print(result[0])