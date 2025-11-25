import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

print("Loading dataset...")

df = pd.read_csv("water_requirement_sa.csv")

# Remove ANN prediction column if present
if "Predicted water requirnment" in df.columns:
    df = df.drop(columns=["Predicted water requirnment"])

# ==========================================================
# SAFE CATEGORY MAPPING (avoids KeyError)
# ==========================================================
if "SOIL TYPE" in df.columns:
    df["SOIL TYPE"] = df["SOIL TYPE"].map({'DRY': 4, 'HUMID': 2, 'WET': 0})

if "REGION" in df.columns:
    df["REGION"] = df["REGION"].map({'DESERT': 3, 'SEMI ARID': 2, 'SEMI HUMID': 1, 'HUMID': 0})

if "TEMPERATURE" in df.columns:
    df["TEMPERATURE"] = df["TEMPERATURE"].map({'10-20': 15, '20-30': 25, '30-40': 35, '40-50': 45})

if "WEATHER CONDITION" in df.columns:
    df["WEATHER CONDITION"] = df["WEATHER CONDITION"].map({'RAINY': 0, 'WINDY': 1, 'NORMAL': 3, 'SUNNY': 4})

# ==========================================================
# Extract numeric columns
# ==========================================================
numericCols = list(df.select_dtypes(include=[np.number]).columns)

# Remove unwanted & TARGET columns
for col in ['Farmer_Age', 'Annual_Income', 'Water_Bill', 'WATER REQUIREMENT']:
    if col in numericCols:
        numericCols.remove(col)

if "WATER REQUIREMENT" not in df.columns:
    raise ValueError("WATER REQUIREMENT column missing from CSV!")

# X = all numeric features EXCEPT the target
X = df[numericCols].values
y = df["WATER REQUIREMENT"].values


print("Fitting distilled linear model...")

# ==========================================================
# Fit linear regression
# ==========================================================
linear_model = LinearRegression()
linear_model.fit(X, y)

print("\nDistilled Linear Model Coefficients:\n")
for col, coef in zip(numericCols[:-1], linear_model.coef_):
    print(f"{col} : {coef}")

print("\nIntercept:", linear_model.intercept_)

# ==========================================================
# Create `models/` directory if missing
# ==========================================================
if not os.path.exists("models"):
    os.makedirs("models")

# Save coefficients
np.savez(
    "models/water_model_params.npz",
    coef=linear_model.coef_,
    intercept=linear_model.intercept_,
    numeric_cols=np.array([
        "SOIL TYPE",
        "REGION",
        "TEMPERATURE",
        "WEATHER CONDITION",
        "Farm_Size",
        "Fertilizer_Usage"
    ], dtype=object)
)

print("\nSaved distilled model to models/water_model_params.npz successfully.")
