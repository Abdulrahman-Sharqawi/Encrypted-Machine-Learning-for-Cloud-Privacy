# client.py  –  Encrypted inference client
import base64
import re
import numpy as np
import pandas as pd
import tenseal as ts
import requests

# ---------- 1. Load saved model parameters ----------
params = np.load("models/water_model_params.npz", allow_pickle=True)
coef             = params["coef"].ravel()
scaler_mean      = params["scaler_mean"].astype(float)
scaler_scale     = params["scaler_scale"].astype(float)
ohe_categories   = list(params["ohe_categories"])       # list of arrays
numeric_cols     = list(params["numeric_cols"])
categorical_cols = list(params["categorical_cols"])

# ---------- 2. Helper functions ----------
_range_re = re.compile(r'^\s*([+-]?\d+(?:\.\d+)?)\s*-\s*([+-]?\d+(?:\.\d+)?)\s*$')
def parse_range(v):
    if isinstance(v, (int, float)): return float(v)
    s = str(v).strip()
    m = _range_re.match(s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    try:
        return float(s)
    except:
        return np.nan

# ---------- 3. Prepare one input sample ----------
df = pd.read_csv("water_requirement_sa.csv")
sample = df.sample(1, random_state=42).iloc[0]

# Numeric features: normalize using saved scaler
x_num = []
for c in numeric_cols:
    val = parse_range(sample.get(c, np.nan))
    if np.isnan(val):
        val = scaler_mean[numeric_cols.index(c)]
    norm = (val - scaler_mean[numeric_cols.index(c)]) / scaler_scale[numeric_cols.index(c)]
    x_num.append(norm)
x_num = np.array(x_num, dtype=float)

# Categorical features: one-hot using stored categories
x_cat = []
for col, cats in zip(categorical_cols, ohe_categories):
    val = str(sample.get(col, ""))
    enc = np.zeros(len(cats))
    for i, cat in enumerate(cats):
        if str(cat) == val:
            enc[i] = 1.0
            break
    x_cat.append(enc)
x_cat = np.concatenate(x_cat) if len(x_cat) else np.array([])

# Combine numeric + categorical
x_full = np.concatenate([x_num, x_cat])
assert len(x_full) == len(coef), f"Length mismatch: vector={len(x_full)} vs coef={len(coef)}"

# ---------- 4. Encrypt vector ----------
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[40, 21, 40]
)
context.global_scale = 2 ** 30
context.generate_galois_keys()

enc_x = ts.ckks_vector(context, x_full)

# Serialize to Base64 for JSON transport
enc_x_b64 = base64.b64encode(enc_x.serialize()).decode("ascii")
context_b64 = base64.b64encode(context.serialize(save_secret_key=False)).decode("ascii")

# ---------- 5. Send to cloud server ----------
url = "http://10.13.96.52:5000/infer"   # replace with your cloud IP or domain
payload = {"enc_vector": enc_x_b64, "context": context_b64}

print("Sending encrypted request to server...")
resp = requests.post(url, json=payload)
result = resp.json()

# ---------- 6. Receive encrypted result and decrypt ----------
if "enc_result" in result:
    enc_result_b64 = result["enc_result"]
    enc_y = ts.ckks_vector_from(context, base64.b64decode(enc_result_b64))
    decrypted = enc_y.decrypt()[0]
    print(f"\nDecrypted prediction: {decrypted:.6f}")
else:
    print("\nError from server:", result)
