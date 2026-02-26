import pandas as pd
import numpy as np
import os

def load_elliptic(data_dir):
    """
    Load Elliptic dataset from any directory (Google Drive compatible)
    """

    # ===== FILE PATH =====
    feat_path = os.path.join(data_dir, "elliptic_txs_features.csv")
    edge_path = os.path.join(data_dir, "elliptic_txs_edgelist.csv")
    class_path = os.path.join(data_dir, "elliptic_txs_classes.csv")

    print("Loading from:", data_dir)

    # ===== READ =====
    feat = pd.read_csv(feat_path, header=None)
    edges = pd.read_csv(edge_path)
    classes = pd.read_csv(class_path)

    classes.columns = ["txId", "class"]
    edges.columns = ["txId1", "txId2"]

    # ===== FEATURES =====
    tx_ids = feat[0].values
    timestep = feat[1].values
    X = feat.iloc[:, 2:].values.astype(np.float32)

    # ===== LABEL =====
    label_map = {"unknown": -1, "1": 1, "2": 0}
    y_map = classes.set_index("txId")["class"].astype(str).map(label_map)
    y = np.array([y_map.get(tx, -1) for tx in tx_ids], dtype=np.int64)

    # ===== NODE INDEX MAP =====
    id_map = {tx: i for i, tx in enumerate(tx_ids)}

    # ===== EDGE INDEX =====
    src, dst = [], []
    for a, b in zip(edges["txId1"], edges["txId2"]):
        if a in id_map and b in id_map:
            src.append(id_map[a])
            dst.append(id_map[b])

    edge_index = np.array([src, dst], dtype=np.int64)

    # ===== INFO =====
    print("Nodes:", X.shape[0])
    print("Features:", X.shape[1])
    print("Edges:", edge_index.shape[1])
    print("Fraud:", (y==1).sum(), "Normal:", (y==0).sum(), "Unknown:", (y==-1).sum())

    return X, y, edge_index, timestep
