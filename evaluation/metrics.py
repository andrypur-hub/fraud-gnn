import numpy as np
from sklearn.metrics import average_precision_score

def auc_pr(y_true, y_score):
    return average_precision_score(y_true, y_score)

def recall_at_k(y_true, y_score, k=0.05):
    n = int(len(y_score)*k)
    idx = np.argsort(-y_score)[:n]
    return y_true[idx].sum() / max(y_true.sum(),1)
