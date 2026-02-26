import os
import numpy as np
import torch
from torch_geometric.data import Data

from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from data.load_elliptic import load_elliptic
from data.filter_graph import filter_graph_known_nodes
from data.split_time import time_based_split
from models.gnn.graphsage import GraphSAGE
from trainers.train_gnn import train_gnn
from evaluation.metrics import recall_at_k


# =========================================================
# 1. LOAD DATASET
# =========================================================
print("\n=== Loading Elliptic Dataset ===")

dataset_path = "/content/drive/MyDrive/ProjectPython/TGNN/Dataset/elliptic_bitcoin_dataset"

X, y, edge_index, timestep = load_elliptic(dataset_path)

print("Nodes:", len(X))
print("Features:", X.shape[1])
print("Edges:", edge_index.shape[1])
print("Fraud:", np.sum(y==1), "Normal:", np.sum(y==0), "Unknown:", np.sum(y==-1))


# =========================================================
# 2. REMOVE UNKNOWN + REINDEX GRAPH  ⭐ PENTING
# =========================================================
X, y, edge_index, timestep = filter_graph_known_nodes(X, y, edge_index, timestep)

print("\nAfter removing unknown:")
print("Nodes:", len(X))
print("Fraud:", np.sum(y==1), "Normal:", np.sum(y==0))


# =========================================================
# 3. TIME SPLIT
# =========================================================
train_mask, val_mask, test_mask = time_based_split(timestep)

print("\nSplit:")
print("Train:", train_mask.sum())
print("Val:", val_mask.sum())
print("Test:", test_mask.sum())


# =========================================================
# 4. BASELINE (XGBOOST)
# =========================================================
print("\n=== Training XGBoost Baseline ===")

model_xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="logloss"
)

model_xgb.fit(X[train_mask], y[train_mask])

probs = model_xgb.predict_proba(X[test_mask])[:,1]

auc_pr = average_precision_score(y[test_mask], probs)
rec5 = recall_at_k(y[test_mask], probs, k=0.05)

print("XGBoost AUC-PR:", auc_pr)
print("XGBoost Recall@5%:", rec5)


# =========================================================
# 5. BUILD GRAPH FOR GNN  ⭐ FIX EDGE INDEX DISINI
# =========================================================
print("\n=== Preparing Graph ===")

x_torch = torch.tensor(X, dtype=torch.float)
y_torch = torch.tensor(y, dtype=torch.long)
edge_index_torch = torch.tensor(edge_index, dtype=torch.long)

print("FINAL GRAPH CHECK")
print("Nodes:", x_torch.shape[0])
print("Max edge index:", edge_index_torch.max().item())

data = Data(x=x_torch, edge_index=edge_index_torch, y=y_torch)


train_mask_torch = torch.tensor(train_mask)
val_mask_torch = torch.tensor(val_mask)
test_mask_torch = torch.tensor(test_mask)


# =========================================================
# 6. TRAIN GRAPHSAGE
# =========================================================
print("\n=== Training GraphSAGE ===")

model = GraphSAGE(
    in_channels=X.shape[1],
    hidden_channels=128,
    out_channels=2
)

model = train_gnn(model, data, train_mask_torch, epochs=50)


# =========================================================
# 7. EVALUATION
# =========================================================
print("\n=== Evaluating GNN ===")

model.eval()
with torch.no_grad():
    out = model(data)
    prob = torch.softmax(out, dim=1)[:,1].cpu().numpy()

auc_pr_gnn = average_precision_score(y[test_mask], prob[test_mask])
rec5_gnn = recall_at_k(y[test_mask], prob[test_mask], k=0.05)

print("\n=== FINAL RESULT ===")
print("XGBoost Recall@5% :", rec5)
print("GraphSAGE Recall@5% :", rec5_gnn)
