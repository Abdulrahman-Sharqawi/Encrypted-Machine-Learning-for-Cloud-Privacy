import os
import uuid
import json
import base64
from datetime import datetime

from flask import Flask, request, jsonify, send_file
import tenseal as ts
import numpy as np

# ---------------- CONFIG ----------------
JOBS_DIR = "jobs"
os.makedirs(JOBS_DIR, exist_ok=True)

MODEL_PARAMS_PATH = "models/water_model_params.npz"

# ---------------- LOAD MODEL ----------------
params = np.load(MODEL_PARAMS_PATH, allow_pickle=True)
COEF = params["coef"].ravel()
INTERCEPT = float(np.asarray(params["intercept"]).ravel()[0])

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 300MB upload limit


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def generate_job_id():
    return datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:6]


def paths(job_id):
    job_dir = os.path.join(JOBS_DIR, job_id)
    return {
        "dir": job_dir,
        "meta": os.path.join(job_dir, "meta.json"),
        "log": os.path.join(job_dir, "progress.log"),
        "enc_csv": os.path.join(job_dir, "encrypted_predictions.csv"),
    }


def log(job_id, msg):
    p = paths(job_id)
    with open(p["log"], "a", encoding="utf-8") as f:
        f.write(f"[{datetime.utcnow().isoformat()}] {msg}\n")


def save_meta(job_id, meta):
    p = paths(job_id)
    with open(p["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_meta(job_id):
    p = paths(job_id)
    if not os.path.exists(p["meta"]):
        return None
    with open(p["meta"], "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "OK"})


# ---------------------------------------------------------
@app.route("/upload_job", methods=["POST"])
def upload_job():
    try:
        data = request.get_json(force=True)

        enc_vectors_b64 = data["enc_vectors"]
        context_b64 = data["context"]
        job_id = data.get("job_id")

        if not isinstance(enc_vectors_b64, list) or len(enc_vectors_b64) == 0:
            return jsonify({"error": "enc_vectors must be non-empty list"}), 400

        # FIRST CHUNK
        if job_id is None:
            job_id = generate_job_id()
            p = paths(job_id)
            os.makedirs(p["dir"], exist_ok=True)

            meta = {
                "job_id": job_id,
                "status": "running",
                "total_chunks": 0,
                "total_vectors": 0,
                "created_at": datetime.utcnow().isoformat(),
            }
            save_meta(job_id, meta)

            with open(p["enc_csv"], "w", encoding="utf-8") as f:
                f.write("enc_prediction\n")

            log(job_id, "Job created. First chunk received.")

        # SUBSEQUENT CHUNKS
        p = paths(job_id)
        meta = load_meta(job_id)

        if meta is None:
            return jsonify({"error": "Invalid job_id"}), 400

        context = ts.context_from(base64.b64decode(context_b64))

        with open(p["enc_csv"], "a", encoding="utf-8") as f:
            for enc_vec_b64 in enc_vectors_b64:
                enc_x = ts.ckks_vector_from(context, base64.b64decode(enc_vec_b64))
                enc_y = enc_x.dot(COEF.tolist()) + INTERCEPT
                enc_y_b64 = base64.b64encode(enc_y.serialize()).decode("ascii")
                f.write(enc_y_b64 + "\n")

        meta["total_chunks"] += 1
        meta["total_vectors"] += len(enc_vectors_b64)
        save_meta(job_id, meta)

        log(job_id, f"Processed chunk #{meta['total_chunks']} ({len(enc_vectors_b64)} vectors).")

        return jsonify({
            "job_id": job_id,
            "status": "running",
            "chunks_received": meta["total_chunks"],
            "vectors_received": meta["total_vectors"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------------------------------------------------
@app.route("/complete_job/<job_id>", methods=["POST"])
def complete_job(job_id):
    meta = load_meta(job_id)
    if not meta:
        return jsonify({"error": "Job not found"}), 404

    meta["status"] = "complete"
    meta["completed_at"] = datetime.utcnow().isoformat()
    save_meta(job_id, meta)

    log(job_id, "Job completed.")
    return jsonify({"job_id": job_id, "status": "complete"})


# ---------------------------------------------------------
@app.route("/job_status/<job_id>", methods=["GET"])
def job_status(job_id):
    p = paths(job_id)
    meta = load_meta(job_id)
    if not meta:
        return jsonify({"error": "Job not found"}), 404

    logs = []
    if os.path.exists(p["log"]):
        with open(p["log"], "r", encoding="utf-8") as f:
            logs = [line.strip() for line in f.readlines()[-15:]]

    return jsonify({
        "job_id": job_id,
        "status": meta.get("status"),
        "chunks_received": meta.get("total_chunks"),
        "vectors_received": meta.get("total_vectors"),
        "logs": logs,
    })


# ---------------------------------------------------------
@app.route("/download/<job_id>", methods=["GET"])
def download(job_id):
    p = paths(job_id)
    meta = load_meta(job_id)

    if not meta:
        return jsonify({"error": "Job not found"}), 404

    if meta.get("status") != "complete":
        return jsonify({"error": "Job not complete"}), 400

    return send_file(p["enc_csv"], as_attachment=True,
                     download_name=f"{job_id}_encrypted_predictions.csv")


# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
