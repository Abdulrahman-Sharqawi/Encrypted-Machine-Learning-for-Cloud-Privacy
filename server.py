# Client (Local)                           Cloud Server
# -------------------                      --------------------------
# Load TenSEAL context + keys              Load model params (.npz)
# ↓                                         ↓
# Encrypt user input → ciphertext  ─────►  Receive ciphertext
#                                           Compute: enc_y = enc_x·w + b
# ↓                                         ↓
# Decrypt encrypted output ◄──────────────  Return ciphertext result
# ↓
# Display decrypted prediction
from flask import Flask, request, jsonify
import tenseal as ts
import numpy as np
import base64

app = Flask(__name__)

# Load model parameters once on startup
params = np.load("models/water_model_params.npz", allow_pickle=True)
coef = params["coef"].ravel()
intercept = float(np.asarray(params["intercept"]).ravel()[0])

@app.route("/infer", methods=["POST"])
def infer():
    try:
        data = request.get_json()
        # Expect Base64-encoded encrypted vector and TenSEAL context
        enc_vector_b64 = data["enc_vector"]
        context_b64 = data["context"]

        # Reconstruct context and ciphertext
        context = ts.context_from(base64.b64decode(data["context"]))
        enc_x = ts.ckks_vector_from(context, base64.b64decode(data["enc_vector"]))

        # Compute encrypted inference
        enc_y = enc_x.dot(coef) + intercept
        result_b64 = base64.b64encode(enc_y.serialize()).decode("ascii")
        return jsonify({"enc_result": result_b64})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
