import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# project modules
from data.load_elliptic import load_elliptic
from baselines.train_xgb import train_xgb
from models.gnn.graphsage import GraphSAGE
from trainers.train_gnn import train_gnn
from evaluation.metrics import auc_pr, recall_at_k


# =========================================================
# CONFIG
# =========================================================
DATA_DIR = "/content/drive/MyDrive/ProjectPython/TGNN/Dataset/elliptic_bitcoin_dataset"   # <-- ganti jika beda
RESULT_PATH = "results/exp1.csv"

os.makedirs("results", exist_ok=True)


# =========================================================
# LOAD DATASET
# =========================================================
print("\n=== Loading Elliptic Dataset ===")
X, y, edge_index, timestep = load_elliptic(DATA_DIR)

# pakai hanya label known dulu
mask_known = y != -1
X = X[mask_known]
y = y[mask_known]
timestep = timestep[mask_known]

print("\nAfter removing unknown:")
print("Nodes:", len(y))
print("Fraud:", (y==1).sum(), "Normal:", (y==0).sum())


# =========================================================
# TIME SPLIT (REALISTIC)
# =========================================================
train_mask = timestep <= 34
val_mask   = (timestep > 34) & (timestep <= 40)
test_mask  = timestep > 40

print("\nSplit:")
print("Train:", train_mask.sum())
print("Val:", val_mask.sum())
print("Test:", test_mask.sum())


# =========================================================
# BASELINE — XGBOOST
# =========================================================
print("\n=== Training XGBoost Baseline ===")
model_xgb = train_xgb(X[train_mask], y[train_mask])

prob_xgb = model_xgb.predict_proba(X[test_mask])[:,1]

auc_xgb = auc_pr(y[test_mask], prob_xgb)
rec_xgb = recall_at_k(y[test_mask], prob_xgb)

print("XGBoost AUC-PR:", auc_xgb)
print("XGBoost Recall@5%:", rec_xgb)


# =========================================================
# GRAPH MODEL — GraphSAGE
# =========================================================
print("\n=== Training GraphSAGE ===")

data = Data(
    x=torch.tensor(X, dtype=torch.float),
    edge_index=torch.tensor(edge_index, dtype=torch.long),
    y=torch.tensor(y, dtype=torch.float)
)

train_mask_torch = torch.tensor(train_mask)

model = GraphSAGE(in_dim=X.shape[1])
model = train_gnn(model, data, train_mask_torch, epochs=50)

# inference
model.eval()
with torch.no_grad():
    logits = model(data)
    prob = torch.sigmoid(logits).cpu().numpy()

prob_test = prob[test_mask]

auc_gnn = auc_pr(y[test_mask], prob_test)
rec_gnn = recall_at_k(y[test_mask], prob_test)

print("GraphSAGE AUC-PR:", auc_gnn)
print("GraphSAGE Recall@5%:", rec_gnn)


# =========================================================
# SAVE RESULT
# =========================================================
df = pd.DataFrame({
    "model":["XGBoost","GraphSAGE"],
    "AUC_PR":[auc_xgb, auc_gnn],
    "Recall@5%":[rec_xgb, rec_gnn]
})

df.to_csv(RESULT_PATH, index=False)

print("\n=== FINAL RESULT ===")
print(df)
print("\nSaved to:", RESULT_PATH)
