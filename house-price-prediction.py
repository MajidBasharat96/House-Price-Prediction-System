########################################################
# House Price Prediction System
########################################################

# Folder structure:
# house-price-prediction/
# ├── data/housing.csv
# ├── src/data_ingestion.py
# ├── src/feature_engineering.py
# ├── src/model_training.py
# ├── src/predict_api.py
# ├── requirements.txt
# ├── Dockerfile
# ├── README.md
# └── .gitignore

# src/data_ingestion.py
import pandas as pd

def load_data(file_path='data/housing.csv'):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    return df

# src/feature_engineering.py
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import pickle

def preprocess(df):
    cat_cols = df.select_dtypes(include=['object']).columns
    num_cols = df.select_dtypes(include=['int64','float64']).columns

    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    scaler = StandardScaler()

    df_cat = pd.DataFrame(encoder.fit_transform(df[cat_cols]), columns=encoder.get_feature_names_out(cat_cols))
    df_num = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

    df_processed = pd.concat([df_num, df_cat], axis=1)

    # Save transformers
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return df_processed

# src/model_training.py
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from feature_engineering import preprocess
from data_ingestion import load_data

# Load and preprocess
df = load_data()
df_processed = preprocess(df)
X = df_processed.drop('Price', axis=1)
y = df_processed['Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print('RMSE:', mean_squared_error(y_test, preds, squared=False))

# Save model
with open('house_price_model.pkl', 'wb') as f:
CMD ["uvicorn", "src.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
