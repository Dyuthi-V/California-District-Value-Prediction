import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#loading the dataset

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent #__file__ means location of python file currently being executed
data = pd.read_csv(BASE_DIR / "data" / "dataset.csv")
print(data.head())

#checking for null values

print("missing values:")
print(data.isnull().sum())

#better features
#rooms per household = how many rooms there are on average per household
data["rooms_per_household"] = data["total_rooms"] / data["households"]

#bedrooms per room = proportion of rooms that are bedrooms
data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]

#population per household = average number of people per household
data["population_per_household"] = data["population"] / data["households"]


#seperate features and target
X=data[[
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household"
]]

y=data["median_house_value"]

#ocean_proximity is a categorical feature(not numeric) so we one hot encode it into numerical values

# Numerical columns
numeric_features = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household"
]

#categorical column
categorical_features = [
    "ocean_proximity"
]

#preprocessing the data
preprocessor=ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")), #imputes values for null values using median (instead of dropping the rows)
            ("scaler", StandardScaler())
        ]), numeric_features),

        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

#model pipeline
model=Pipeline( #put mulitple ML steps together-- preprocessing, training
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

#split training and testing data
X_train, X_test, y_train, y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("training samples:", len(X_train))
print("testing samples:", len(X_test))

#don't need the below code anymore as we used columntransformer while preprocessing the data above which handles both numeric and text features
"""preprocessing the data
scaler=StandardScaler() 
X_train_scaled=scaler.fit_transform(X_train) uses z-score formula, z=x-u/sigma
X_test_scaled= scaler.transform(X_test) does not calculate a new mean and std devn,
uses the same mean and std dev from the training set bcus otherwise testing data will have its own scaling"""

#train model
model.fit(X_train, y_train)
print("model training complete")

#predict model
y_pred=model.predict(X_test)

print('actual prices:')
print(y_test.values)

print('predicted prices:')
print(y_pred)

#graph of actual vs predicted prices
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Actual vs Predicted House Values")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)
plt.show()


# model performance
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print()
print("MODEL PERFORMANCE")
print("-----------------")

print("R² score:", r2)
print("R² percentage:", r2 * 100, "%")
print("This means your model explains approximately",
      round(r2 * 100, 2),
      "%", "of the variation in median house values in the test data.")
print("It does not mean the model is", round(r2 * 100, 2), "% accurate.")

print()

print("MAE:", round(mae, 2))
print("This means that, on average, your predictions are about $",
      round(mae, 2),
      "away from the actual values.")

print()

print("RMSE:", round(rmse, 2))
print("This is similar to MAE but penalizes large errors more heavily.")
print("It is useful for seeing whether your model sometimes makes very large mistakes.")

print()

print("MAPE:", round(mape, 2), "%")
print("This means the predictions are, on average, about",
      round(mape, 2),
      "%", "away from the actual values in percentage terms.")


#save the model
joblib.dump(model, "housepricing_model.pkl")
print("model saved successfully")
