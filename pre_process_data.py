import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import difflib
import sys
import os
import re

# 1. Load dataset
csv_path = "water_requirement_sa.csv"
df = pd.read_csv(csv_path)

# 2. Separate features & target (update if needed)
target_col = "WATER REQUIREMENT"

# Robust check for target column
if target_col not in df.columns:
    cols_map = {c.lower().strip(): c for c in df.columns}
    key = target_col.lower().strip()
    if key in cols_map:
        real_name = cols_map[key]
        df = df.rename(columns={real_name: target_col})
        print(f"Auto-renamed column '{real_name}' -> '{target_col}'")
    else:
        close = difflib.get_close_matches(target_col, df.columns, n=5, cutoff=0.5)
        print(f"ERROR: Target column '{target_col}' not found in {csv_path}.")
        print("Available columns:", list(df.columns))
        if close:
            print("Close matches:", close)
            print("If one of these is the target, update target_col accordingly.")
        else:
            print("No close matches found. Open the CSV and confirm the header name.")
        sys.exit(1)

X = df.drop(columns=[target_col])
y = df[target_col]

# --- Begin added/changed block: robust numeric parsing & column inference ---
def parse_range_series(s: pd.Series) -> pd.Series:
    """Convert 'num-num' ranges to their midpoint, parse numbers, else NaN."""
    pattern = re.compile(r'^\s*([+-]?\d+(?:\.\d+)?)\s*-\s*([+-]?\d+(?:\.\d+)?)\s*$')
    def conv(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float)):
            return v
        sv = str(v).strip()
        m = pattern.match(sv)
        if m:
            return (float(m.group(1)) + float(m.group(2))) / 2.0
        try:
            return float(sv)
        except Exception:
            return np.nan
    return s.map(conv)

# Trim whitespace from column names and standardize casing for matching
original_columns = list(X.columns)
cols_map = {c.lower().strip(): c for c in original_columns}
# If user provided categorical names (mixed-case), try to map them; otherwise infer
user_categorical = [
    "CROP TYPE", "SOIL TYPE", "REGION", "WEATHER CONDITION",
    "Farm_Location", "Crop_Type"
]
mapped_cats = []
for name in user_categorical:
    key = name.lower().strip()
    if key in cols_map:
        mapped_cats.append(cols_map[key])
# If no mapped user categories found, infer by dtype/string content
if mapped_cats:
    categorical_cols = mapped_cats
else:
    # Try to infer categorical columns by dtype (object) or low unique count
    categorical_cols = []
    for c in X.columns:
        if X[c].dtype == object or X[c].nunique(dropna=True) < max(10, X.shape[0] * 0.05):
            categorical_cols.append(c)

# Attempt to coerce possible numeric-looking columns (including "40-50" ranges)
for c in list(X.columns):
    if c in categorical_cols:
        # still try to convert ranges inside categorical candidates (sometimes ranges are numeric)
        converted = parse_range_series(X[c])
        # if many non-NaN numeric conversions, keep the numeric conversion
        if converted.notna().sum() >= max(1, int(0.5 * len(converted))):
            X[c] = converted
        else:
            X[c] = X[c].astype(object)
    else:
        X[c] = parse_range_series(X[c])

# Recompute numeric / categorical after parsing
numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
categorical_cols = [c for c in X.columns if c not in numeric_cols]
print(f"Inferred numeric cols: {numeric_cols}")
print(f"Inferred categorical cols: {categorical_cols}")
# --- End added/changed block ---

# 3. Identify column types
# categorical_cols = [
#     "CROP TYPE", 
#     "SOIL TYPE", 
#     "REGION", 
#     "WEATHER CONDITION",
#     "Farm_Location", 
#     "Crop_Type"
# ]
# numeric_cols = [c for c in X.columns if c not in categorical_cols]


# 4. Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# 5. Build pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model trained successfully.")
print(f"Mean Squared Error: {mse:.3f}")
print(f"R² Score: {r2:.3f}")

# 9. Save trained model and params
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

# Save the full model pipeline
model_path = os.path.join(output_dir, "water_model_plain.pkl")
joblib.dump(model, model_path)

# Extract regression parameters for encryption step
reg = model.named_steps["regressor"]
coef = np.asarray(reg.coef_)
intercept = np.asarray(reg.intercept_)

# Extract preprocessing details
pre = model.named_steps["preprocessor"]
scaler = pre.named_transformers_["num"].named_steps["scaler"]
scaler_mean = np.asarray(scaler.mean_)
scaler_scale = np.asarray(scaler.scale_)

ohe = pre.named_transformers_["cat"].named_steps["encoder"]
ohe_categories = [np.asarray(c) for c in ohe.categories_]

# Feature lists — use the inferred lists from preprocessing (do not override)
# numeric_cols and categorical_cols were computed earlier after parsing
# Convert ohe_categories to an object array so np.savez can store variable-length arrays
ohe_categories_obj = np.array(ohe_categories, dtype=object)

# Ensure numeric_cols and categorical_cols are plain Python lists (inferred earlier)
# numeric_cols and categorical_cols were printed earlier; reuse those variables
numeric_cols_to_save = list(numeric_cols)
categorical_cols_to_save = list(categorical_cols)

# Save model parameters
params_path = os.path.join(output_dir, "water_model_params.npz")
np.savez(
    params_path,
    coef=coef,
    intercept=intercept,
    scaler_mean=scaler_mean,
    scaler_scale=scaler_scale,
    ohe_categories=ohe_categories_obj,
    numeric_cols=np.array(numeric_cols_to_save, dtype=object),
    categorical_cols=np.array(categorical_cols_to_save, dtype=object)
)

print(f"Model pipeline saved to: {model_path}")
print(f"Model parameters saved to: {params_path}")
print("All files are located in the 'models/' directory.")

# --- Begin added block: save useful outputs to Excel for inspection ---
excel_path = os.path.join(output_dir, "water_model_outputs.xlsx")

# Predictions sheet: X_test with true and predicted target
pred_df = X_test.reset_index(drop=True).copy()
pred_df["y_true"] = y_test.reset_index(drop=True)
pred_df["y_pred"] = y_pred

# Coefficients sheet: build feature names (numeric + one-hot)
try:
    # get ohe feature names (scikit-learn >= 1.0)
    ohe_feature_names = list(ohe.get_feature_names_out(categorical_cols_to_save))
except Exception:
    # fallback: manual construction
    ohe_feature_names = []
    for col, cats in zip(categorical_cols_to_save, ohe.categories_):
        for cat in cats:
            ohe_feature_names.append(f"{col}_{cat}")

feature_names = list(numeric_cols_to_save) + ohe_feature_names
coef_arr = np.asarray(coef).ravel()
# Ensure length matches (if mismatch, still save what we can)
min_len = min(len(feature_names), coef_arr.shape[0])
coef_df = pd.DataFrame({
    "feature": feature_names[:min_len],
    "coefficient": coef_arr[:min_len]
})

# Scaler params sheet
scaler_df = pd.DataFrame({
    "numeric_feature": numeric_cols_to_save,
    "scaler_mean": list(scaler_mean),
    "scaler_scale": list(scaler_scale)
})

# One-hot categories sheet
ohe_rows = []
for col, cats in zip(categorical_cols_to_save, ohe.categories_):
    ohe_rows.append({"categorical_column": col, "categories": ", ".join(map(str, cats))})
ohe_df = pd.DataFrame(ohe_rows)

# Model info (intercept)
try:
    intercept_val = float(np.asarray(intercept).ravel()[0])
except Exception:
    intercept_val = np.asarray(intercept).tolist()
model_info_df = pd.DataFrame([{"intercept": intercept_val, "num_features": len(feature_names)}])

# Write to Excel (requires openpyxl; pandas will pick an available engine)
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    pred_df.to_excel(writer, sheet_name="predictions", index=False)
    coef_df.to_excel(writer, sheet_name="coefficients", index=False)
    scaler_df.to_excel(writer, sheet_name="scaler_params", index=False)
    ohe_df.to_excel(writer, sheet_name="ohe_categories", index=False)
    model_info_df.to_excel(writer, sheet_name="model_info", index=False)

print(f"Excel outputs saved to: {excel_path}")

# --- Begin added block: diagnostics & full-dataset predictions ---
# Print basic counts so you can confirm where rows went
print(f"Original dataframe rows: {len(df)}")
print(f"Features (X) rows: {len(X)}")
print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

# Count NaNs introduced by parsing (per column)
nan_counts = X.isna().sum()
print("NaN counts per column (after parsing):")
print(nan_counts[nan_counts > 0].to_dict())

# Save predictions for the full dataset (predict on X; will raise if NaNs remain)
try:
    full_pred = model.predict(X)
    full_pred_df = X.reset_index(drop=True).copy()
    full_pred_df["y_pred_full"] = full_pred
    # If ground-truth exists, add it
    if target_col in df.columns:
        full_pred_df["y_true_full"] = df[target_col].reset_index(drop=True)
    full_excel_path = os.path.join(output_dir, "water_model_full_predictions.xlsx")
    with pd.ExcelWriter(full_excel_path, engine="openpyxl") as writer:
        full_pred_df.to_excel(writer, sheet_name="full_predictions", index=False)
    print(f"Full-dataset predictions saved to: {full_excel_path}")
except Exception as e:
    print("Could not predict on full X (likely NaNs present). Exception:", e)
    # Save rows with NaNs so you can inspect them
    nan_rows = X[X.isna().any(axis=1)].reset_index(drop=True)
    if len(nan_rows) > 0:
        nan_rows_path = os.path.join(output_dir, "rows_with_nans.xlsx")
        nan_rows.to_excel(nan_rows_path, index=False)
        print(f"Rows containing NaNs saved to: {nan_rows_path}")
# --- End added block ---
