# ai_model_poly.py
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pandas import DataFrame

class PolyNet(nn.Module):
    """
    Simple HE-friendly neural network:
      x -> Linear(D_in -> H) -> square activation -> Linear(H -> 1)
    Only uses operations we can reproduce with CKKS (dot/mul/add/square).
    """
    def __init__(self, D_in, H, D_out=1):
        super().__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)

    def forward(self, x):
        h = self.fc1(x)
        h = h * h  # polynomial activation: square
        y = self.fc2(h)
        return y


def load_data():
    df = pd.read_csv("water_requirement_sa.csv")

    # IMPORTANT: same mappings as client_job.py
    df["SOIL TYPE"]         = df["SOIL TYPE"].map({'DRY': 4, 'HUMID': 2, 'WET': 0})
    df["REGION"]            = df["REGION"].map({'DESERT': 3, 'SEMI ARID': 2, 'SEMI HUMID': 1, 'HUMID': 0})
    df["TEMPERATURE"]       = df["TEMPERATURE"].map({'10-20': 15, '20-30': 25, '30-40': 35, '40-50': 45})
    df["WEATHER CONDITION"] = df["WEATHER CONDITION"].map({'RAINY': 0, 'WINDY': 1, 'NORMAL': 3, 'SUNNY': 4})

    numeric_cols = [
        "SOIL TYPE",
        "REGION",
        "TEMPERATURE",
        "WEATHER CONDITION",
        "Farm_Size",
        "Fertilizer_Usage",
    ]
    target_col = "WATER REQUIREMENT"

    X_df = DataFrame(df, columns=numeric_cols)
    y_df = DataFrame(df, columns=[target_col])

    X = torch.tensor(X_df.values, dtype=torch.float32)
    y = torch.tensor(y_df.values, dtype=torch.float32)

    return X, y, numeric_cols


if __name__ == "__main__":
    start = time.time()

    X, y, numeric_cols = load_data()
    D_in = X.shape[1]
    H = 4  # hidden size – you can tune this
    model = PolyNet(D_in, H, 1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 500
    for epoch in range(epochs):
        y_pred = model(X)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, loss = {loss.item():.4f}")

    # Export weights for HE inference
    with torch.no_grad():
        W1 = model.fc1.weight.cpu().numpy()   # (H, D_in)
        b1 = model.fc1.bias.cpu().numpy()     # (H,)
        W2 = model.fc2.weight.cpu().numpy()   # (1, H)
        b2 = model.fc2.bias.cpu().numpy()     # (1,)

    import os
    os.makedirs("models", exist_ok=True)
    np.savez(
        "models/poly_net_params.npz",
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        numeric_cols=np.array(numeric_cols, dtype=object),
    )

    print("Saved HE-friendly PolyNet weights to models/poly_net_params.npz")
    print("Training time:", time.time() - start, "seconds")
