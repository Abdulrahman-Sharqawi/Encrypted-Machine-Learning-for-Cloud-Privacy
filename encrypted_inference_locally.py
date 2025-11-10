import numpy as np
import pandas as pd
import joblib
import tenseal as ts
import re

# --- Helpers (match your training parser for numeric ranges like "40-50") ---
_range_re = re.compile(r'^\s*([+-]?\d+(?:\.\d+)?)\s*-\s*([+-]?\d+(?:\.\d+)?)\s*$')
def parse_range(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int, float)): return float(val)
    s = str(val).strip()
    m = _range_re.match(s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    try:
        return float(s)
    except:
        return np.nan

def coerce_numeric_series(s: pd.Series) -> pd.Series:
    return s.map(parse_range)

# --- Load artifacts ---
params = np.load("models/water_model_params.npz", allow_pickle=True)
coef           = params["coef"].ravel()
intercept      = float(np.asarray(params["intercept"]).ravel()[0])
scaler_mean    = params["scaler_mean"].astype(float)
scaler_scale   = params["scaler_scale"].astype(float)
ohe_categories = list(params["ohe_categories"])     # list of arrays
numeric_cols   = list(params["numeric_cols"])
categorical_cols = list(params["categorical_cols"])

pipe = joblib.load("models/water_model_plain.pkl")   # for plaintext comparison

# --- Load data and pick a sample ---
df = pd.read_csv("water_requirement_sa.csv")

# Ensure numeric columns match training parsing
for c in numeric_cols:
    if c in df.columns:
        df[c] = coerce_numeric_series(df[c])

# Choose a row deterministically (or change random_state)
sample = df.sample(1, random_state=7).iloc[0]

# --- Build the transformed feature vector exactly as in training ---
# 1) numeric: StandardScaler(X - mean) / scale
x_num = []
for c in numeric_cols:
    val = sample.get(c, np.nan)
    val = parse_range(val)
    if np.isnan(val):  # fallback to mean if missing
        i = numeric_cols.index(c)
        val = scaler_mean[i]
    i = numeric_cols.index(c)
    x_num.append( (val - scaler_mean[i]) / scaler_scale[i] )
x_num = np.array(x_num, dtype=float)

# 2) categorical: OneHot in training order
x_cat = []
for col, cats in zip(categorical_cols, ohe_categories):
    v = str(sample.get(col, ""))
    enc = np.zeros(len(cats), dtype=float)
    # match by string equality (as sklearn would map categories_)
    for idx, cat in enumerate(cats):
        if str(cat) == v:
            enc[idx] = 1.0
            break
    # if unseen category -> all zeros (handle_unknown="ignore")
    x_cat.append(enc)
x_cat = np.concatenate(x_cat) if len(x_cat) else np.array([], dtype=float)

# Concatenate numeric then categorical (ColumnTransformer order: "num" then "cat")
x_full = np.concatenate([x_num, x_cat]).ravel()

# Sanity check
assert x_full.shape[0] == coef.shape[0], f"Feature length {x_full.shape[0]} != coef length {coef.shape[0]}"

# --- TenSEAL CKKS context ---
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[40, 21, 40]
)
context.global_scale = 2 ** 30
context.generate_galois_keys()

# Encrypt feature vector
enc_x = ts.ckks_vector(context, x_full)

# Encrypted inference: w·x + b
enc_y = enc_x.dot(coef) + intercept

# Decrypt for verification
y_enc = enc_y.decrypt()[0]

# Plaintext prediction using the full pipeline
# Drop target column if present
target_candidates = ["WATER REQUIREMENT", "Water_Requirement", "water_requirement"]
X_plain = sample.to_frame().T.copy()
for t in target_candidates:
    if t in X_plain.columns:
        X_plain = X_plain.drop(columns=[t])
        break
y_plain = float(pipe.predict(X_plain)[0])

# numbers meanings 
data = {
    "Value": ["Encrypted prediction (decrypted)", "Plain prediction", "Absolute difference"],
    "Number": [round(y_enc, 6), round(y_plain, 6), round(abs(y_enc - y_plain), 6)],
    "Meaning": [
        "Prediction computed on encrypted data (then decrypted).",
        "Prediction from the normal model without encryption.",
        "Difference showing CKKS numerical approximation error."
    ]
}

# output as a DataFrame
print("\n--- Encrypted Inference Summary ---")
df_out = pd.DataFrame(data)

# Ensure pandas prints full column contents (no truncation) and wide output
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 200)

# Print full contents
print(df_out.to_string(index=False))

#print("Encrypted prediction (decrypted):", y_enc)
#print("Plain prediction:", y_plain)
#print("Absolute difference:", abs(y_enc - y_plain))
