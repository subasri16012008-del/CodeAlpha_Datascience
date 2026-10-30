
# TASK 3: CAR PRICE PREDICTION USING MACHINE LEARNING

# Step 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Step 2: Load dataset
df = pd.read_csv("car data.csv")

# Step 3: Display dataset
print("FIRST FIVE RECORDS")
print(df.head())

# Step 4: Dataset information
print("\nDATASET INFORMATION")
print(df.info())

# Step 5: Statistical description
print("\nSTATISTICAL DESCRIPTION")
print(df.describe())

# Step 6: Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Step 7: Remove duplicate records
df.drop_duplicates(inplace=True)

# Step 8: Handle missing target values
df["price"] = pd.to_numeric(
    df["price"], errors="coerce"
)

df.dropna(subset=["price"], inplace=True)

# Step 9: Feature engineering
# Extract car brand from car name

if "CarName" in df.columns:
    df["brand"] = df["CarName"].astype(str).str.split().str[0]

    # Remove original car name
    df.drop(columns=["CarName"], inplace=True)

# Step 10: Remove unnecessary ID columns
df.drop(
    columns=["car_ID"],
    errors="ignore",
    inplace=True
)

# Step 11: Separate input and target
X = df.drop("price", axis=1)
y = df["price"]

print("\nINPUT FEATURES")
print(X.head())

print("\nTARGET PRICE")
print(y.head())

# Step 12: Identify numerical and categorical columns
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Step 13: Create preprocessing pipelines
numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Step 14: Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

# Step 15: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Step 16: Create Random Forest Regression model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ]
)

# Step 17: Train model
model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# Step 18: Predict car prices
y_pred = model.predict(X_test)

print("\nACTUAL CAR PRICES")
print(y_test.head(10).values)

print("\nPREDICTED CAR PRICES")
print(y_pred[:10])

# Step 19: Evaluate model
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")

print("Mean Absolute Error:", round(mae, 2))

print("Mean Squared Error:", round(mse, 2))

print("Root Mean Squared Error:", round(rmse, 2))

print("R2 Score:", round(r2, 4))

# Step 20: Actual vs Predicted visualization
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Car Price")
plt.ylabel("Predicted Car Price")

plt.title("Actual vs Predicted Car Prices")

plt.tight_layout()
plt.show()

# Step 21: Prediction error visualization
errors = y_test.values - y_pred

plt.figure(figsize=(8, 5))

plt.hist(errors, bins=15)

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")

plt.title("Car Price Prediction Error")

plt.tight_layout()
plt.show()

# Step 22: Display price distribution
plt.figure(figsize=(8, 5))

sns.histplot(df["price"], kde=True)

plt.title("Distribution of Car Prices")

plt.xlabel("Car Price")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Step 23: Correlation analysis
numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)

plt.figure(figsize=(12, 8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Car Features")

plt.tight_layout()
plt.show()

# Step 24: Predict a new car price
# Use an existing test record as an example
sample_car = X_test.iloc[[0]]

predicted_price = model.predict(sample_car)

print("\nNEW CAR PRICE PREDICTION")

print("Car Details:")
print(sample_car.to_string(index=False))

print("\nPredicted Car Price:",
      round(predicted_price[0], 2))

# Step 25: Final result
print("\nCAR PRICE PREDICTION COMPLETED SUCCESSFULLY!")
