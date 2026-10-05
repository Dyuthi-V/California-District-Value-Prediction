# California District Value Prediction

## Overview

A Machine Learning project that predicts the median house value of a California district using the California Housing Prices dataset.

Dataset Link: https://www.kaggle.com/datasets/shravanbangera/californiahousing1990

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

The project uses Linear Regression with the following preprocessing techniques:

1. Missing numerical values are handled using median imputation.
2. Numerical features are standardized using `StandardScaler`.
3. The categorical `ocean_proximity` feature is converted into numerical values using `OneHotEncoder`.
4. All preprocessing steps and the machine learning model are combined using a Scikit-learn pipeline.

The model is evaluated using:

- **R² Score**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)**

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
## Data Visualization
One useful visualization of the project is the Actual vs Predicted House Values graph plotted using Matplotlib
The graph compares:
```text
Actual house values
        vs
Predicted house values
```
Points closer to the diagonal reference line indicate predictions that are closer to the actual values.

## How to Run

### Install dependencies:
```text
pip install -r requirements.txt
```

### Train the model:
```text
python src/train.py
```

### Run the prediction application:
```text
python app.py
```

## Technologies Used
Python, Pandas, NumPy, Scikit-learn, Joblib, Matplotlib, VS Code
