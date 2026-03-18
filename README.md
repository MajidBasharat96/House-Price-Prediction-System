# House-Price-Prediction-System

# README.md content
"""
# House Price Prediction System

This project predicts real estate prices based on features like location, square footage, age, and amenities. It is a full end-to-end production-ready ML system.

## Features
- Data ingestion from Kaggle or scraped datasets
- Feature engineering and preprocessing
- Regression model training (XGBoost / LightGBM)
- FastAPI endpoint for price prediction
- Dockerized for cloud deployment (AWS/GCP)

## Folder Structure
```
house-price-prediction/
├── data/housing.csv
├── src/data_ingestion.py
├── src/feature_engineering.py
├── src/model_training.py
├── src/predict_api.py
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

## Quick Start
1. Install dependencies:
```
pip install -r requirements.txt
```
2. Train model:
```
python src/model_training.py
```
3. Run API:
```
uvicorn src.predict_api:app --host 0.0.0.0 --port 8000
```
4. Send POST request to `/predict_price` with property features in JSON.

## Docker Deployment
```
docker build -t house-price-prediction .
docker run -p 8000:8000 house-price-prediction
```
"""
