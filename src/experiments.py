import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error 
import os
import joblib

# Load data
data = pd.read_csv('data/air_quality_index_dataset.csv')

# Drop SO2 and NOX features
data = data.drop(columns=['SO2', 'NOX'])

# Drop any rows where Temperature is missing
data = data.dropna(subset=['Temperature'])

# Split data
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42, )

# Save train and test set
os.makedirs('data', exist_ok=True)
train_set.to_csv('data/train.csv', index=False)
test_set.to_csv('data/test.csv', index=False)

# Load train set
train_set = pd.read_csv('data/train.csv')

# Split features and target
X_train = train_set.drop(columns=['Temperature'])
y_train = train_set['Temperature'].copy()

# Validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Drop NaN values in y_train
y_train = y_train.dropna()
y_val = y_val.dropna()

# Ensure X_train and y_train have the same length after dropping NaNs
valid_idx = y_train.index
X_train = X_train.loc[valid_idx]

valid_idx_val = y_val.index
X_val = X_val.loc[valid_idx_val]

# Identify numerical and categorical columns
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Impute and scale numerical columns
num_imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

X_val[num_cols] = num_imputer.transform(X_val[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])

# Impute and encode categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])

X_val[cat_cols] = cat_imputer.transform(X_val[cat_cols])
X_val[cat_cols] = encoder.transform(X_val[cat_cols])

# Fit the model
model = RandomForestRegressor(n_estimators=120,random_state=42)
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)

# Evaluate the model
rmse =root_mean_squared_error(y_val, y_pred)
print(f'RMSE: {rmse}')


# Save the model and preprocessors
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/random_forest.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoder, 'models/encoder.pkl')
joblib.dump(num_imputer, 'models/num_imputer.pkl')
joblib.dump(cat_imputer, 'models/cat_imputer.pkl')

