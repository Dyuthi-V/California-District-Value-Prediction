# California District Value Prediction

## Overview

A machine learning project that predicts the median house value of a California district using the California Housing Prices dataset.

## Features

The model uses:

- Location (longitude, latitude)
- Housing median age
- Total rooms and bedrooms
- Population
- Households
- Median income
- Ocean proximity

Additional features:
- Rooms per household
- Bedrooms per room
- Population per household

## Model

The project uses Linear Regression with:

- Median imputation
- StandardScaler
- One-Hot Encoding
- Train-test split (80/20)

The model is evaluated using:

- R² Score
- MAE
- RMSE
- MAPE

## Project Structure

```text
house-price-predictor/
├── data/
│   └── dataset.csv
├── model/
│   └── housepricing_model.pkl
├── src/
│   └── train.py
├── app.py
├── README.md
└── requirements.txt
```

## How to Run

### Install dependencies:
pip install -r requirements.txt

### Train the model:
python src/train.py

### Run the prediction application:
python app.py

## Technologies Used
Python, Pandas, NumPy, Scikit-learn, Joblib, Matplotlib, VS Code
