import joblib
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(
    BASE_DIR / "model" / "housepricing_model.pkl"
)

longitude = float(input("Enter longitude: "))
latitude = float(input("Enter latitude: "))

housing_median_age = float(
    input("Enter housing median age: ")
)

total_rooms = float(
    input("Enter total rooms: ")
)

total_bedrooms = float(
    input("Enter total bedrooms: ")
)

population = float(
    input("Enter population: ")
)

households = float(
    input("Enter number of households: ")
)

median_income = float(
    input("Enter median income: ")
)

ocean_proximity = input(
    "Enter ocean proximity: "
)

new_house = pd.DataFrame({
    "longitude": [longitude],
    "latitude": [latitude],
    "housing_median_age": [housing_median_age],
    "total_rooms": [total_rooms],
    "total_bedrooms": [total_bedrooms],
    "population": [population],
    "households": [households],
    "median_income": [median_income],
    "ocean_proximity": [ocean_proximity]
})

# Calculate engineered features

new_house["rooms_per_household"] = (
    new_house["total_rooms"] / new_house["households"]
)

new_house["bedrooms_per_room"] = (
    new_house["total_bedrooms"] / new_house["total_rooms"]
)

new_house["population_per_household"] = (
    new_house["population"] / new_house["households"]
)

prediction = model.predict(new_house)

print("Predicted house price:", prediction[0])