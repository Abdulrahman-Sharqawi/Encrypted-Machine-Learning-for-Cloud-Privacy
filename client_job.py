# client_job.py
import base64
import time
import uuid

import numpy as np
import pandas as pd
import requests
import tenseal as ts

# ---------- Server config ----------
SERVER_URL = "http://16.171.27.11:5000"
UPLOAD_JOB_URL = f"{SERVER_URL}/upload_job"
JOB_STATUS_URL = f"{SERVER_URL}/job_status"
DOWNLOAD_URL = f"{SERVER_URL}/download"

# ---------- Model params ----------
PARAMS_PATH = "models/water_model_params.npz"
DATA_PATH = "water_requirement_sa.csv"


# ---------- Feature mapping (simple numeric model) ----------
# This matches the simple distilled linear model (distill_ann_to_linear.py)
def map_soil(v):
    return {"DRY": 4, "HUMID": 2, "WET": 0}.get(v, 0)

def map_region(v):
    return {"DESERT": 3, "SEMI ARID": 2, "SEMI HUMID": 1, "HUMID": 0}.get(v, 0)

def map_temp(v):
    return {"10-20": 15, "20-30": 25, "30-40": 35, "40-50": 45}.get(v, 25)

def map_weather(v):
    return {"RAINY": 0, "WINDY": 1, "NORMAL": 3, "SUNNY": 4}.get(v, 1)


def build_feature_vector(row, col_name):
    """Build feature value matching the numeric_cols order."""
    if col_name == "SOIL TYPE":         return float(map_soil(row["SOIL TYPE"]))
    if col_name == "REGION":            return float(map_region(row["REGION"]))
    if col_name == "TEMPERATURE":       return float(map_temp(row["TEMPERATURE"]))
    if col_name == "WEATHER CONDITION": return float(map_weather(row["WEATHER CONDITION"]))
    if col_name in row:
        return float(row[col_name])
    raise ValueError(f"Unknown feature column: {col_name}")


def create_ckks_context():
    """
    Create CKKS context with secret key (kept on client),
    but we will send a public-only serialized version to the server.
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[40, 21, 40],
    )
    context.global_scale = 2 ** 30
    context.generate_galois_keys()
    return context


def encrypt_dataset(context, df, feature_order):
    """
    Encrypt all rows of df according to feature_order.
    Returns list of Base64-serialized CKKS ciphertexts.
    """
    enc_vectors = []
    for _, row in df.iterrows():
        x = np.array(
            [build_feature_vector(row, col) for col in feature_order],
            dtype=float
        )
        enc = ts.ckks_vector(context, x)
        enc_b64 = base64.b64encode(enc.serialize()).decode("ascii")
        enc_vectors.append(enc_b64)
    return enc_vectors


def submit_job(enc_vectors, context_public_b64):
    payload = {
        "enc_vectors": enc_vectors,
        "context": context_public_b64,
    }
    resp = requests.post(UPLOAD_JOB_URL, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Upload job failed: {resp.status_code} {resp.text}")
    data = resp.json()
    print("Job created:", data)
    return data["job_id"]


def poll_status(job_id, interval=2.0):
    """
    Poll /job_status until status == 'complete' or 'failed'.
    """
    url = f"{JOB_STATUS_URL}/{job_id}"
    while True:
        resp = requests.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"Status failed: {resp.status_code} {resp.text}")
        data = resp.json()
        status = data["status"]
        print(f"[{job_id}] status = {status}")
        logs = data.get("logs") or []
        for line in logs[-3:]:
            print("  ", line)
        if status in ("complete", "failed"):
            return data
        time.sleep(interval)


def download_encrypted_csv(job_id, out_path=None):
    url = f"{DOWNLOAD_URL}/{job_id}"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"Download failed: {resp.status_code} {resp.text}")
    if out_path is None:
        out_path = f"{job_id}_encrypted_predictions.csv"
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print("Encrypted predictions saved to:", out_path)
    return out_path


def decrypt_predictions_csv(context, enc_csv_path, out_plain_path=None):
    df_enc = pd.read_csv(enc_csv_path)
    preds = []
    for enc_b64 in df_enc["enc_prediction"]:
        enc_vec = ts.ckks_vector_from(context, base64.b64decode(enc_b64))
        val = enc_vec.decrypt()[0]
        scaled = 0.0016 * val      # <<— APPLY SCALING HERE
        preds.append(scaled)

    df_plain = pd.DataFrame({"Prediction": preds})
    if out_plain_path is None:
        out_plain_path = enc_csv_path.replace("_encrypted_", "_decrypted_")
    df_plain.to_csv(out_plain_path, index=False)
    print("Decrypted predictions saved to:", out_plain_path)
    return out_plain_path


if __name__ == "__main__":
    # 1) Load model params for feature order
    params = np.load(PARAMS_PATH, allow_pickle=True)
    numeric_cols = list(params["numeric_cols"])
    print("Using feature order:", numeric_cols)

    # 2) Load dataset
    df = pd.read_csv(DATA_PATH)
    print("Loaded dataset with", len(df), "rows")

    # 3) Create CKKS context (client keeps secret key)
    context = create_ckks_context()
    context_public_b64 = base64.b64encode(
        context.serialize(save_secret_key=False)
    ).decode("ascii")

    # 4) Encrypt all rows
    print("Encrypting dataset...")
    enc_vectors = encrypt_dataset(context, df, numeric_cols)
    print("Encrypted", len(enc_vectors), "rows")

    # 5) Submit job
    BATCH_SIZE = 20
    job_id = None

    for i in range(0, len(enc_vectors), BATCH_SIZE):
        batch = enc_vectors[i:i+BATCH_SIZE]
        print(f"Uploading batch {i//BATCH_SIZE + 1} with {len(batch)} items")

        payload = {
            "enc_vectors": batch,
            "context": context_public_b64,
            "job_id": job_id  # None for first batch, server returns ID
        }

        resp = requests.post(UPLOAD_JOB_URL, json=payload)
        data = resp.json()

        if resp.status_code != 200:
            print("Server error:", resp.text)
            raise SystemExit

        # First batch: get job_id
        if job_id is None:
            job_id = data["job_id"]
            print("Job created with ID:", job_id)
            
    # 6) Mark job as complete
    print("Sending complete_job request...")
    resp = requests.post(f"{SERVER_URL}/complete_job/{job_id}")
    print("Complete job response:", resp.json())

    # 6) Poll status until complete/failed
    info = poll_status(job_id, interval=2.0)
    if info["status"] != "complete":
        raise RuntimeError(f"Job did not complete successfully: {info}")

    # 7) Download encrypted CSV
    enc_csv_path = download_encrypted_csv(job_id)

    # 8) Decrypt predictions
    decrypt_predictions_csv(context, enc_csv_path)
