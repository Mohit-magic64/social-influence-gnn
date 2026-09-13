"""
evaluate.py
-----------
Metrics for a heavily imbalanced binary task.

Accuracy is deliberately absent. With a ~15% positive rate, "always predict 0" scores
85% and is worthless, so reporting accuracy would be actively misleading.

AUC-ROC is the headline number because it is threshold-free and the problem statement
asks for it. AUC-PR is reported alongside it because under class imbalance ROC-AUC
flatters models: the false-positive rate has a huge denominator (all the non-adopters),
so a model can rack up false positives without the ROC curve moving much. Precision-
recall has no such cushion. If you are asked one sharp question about metrics in the
interview, it will probably be this one.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def pick_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Choose the F1-maximising threshold ON VALIDATION ONLY, then freeze it for test."""
    order = np.argsort(-scores)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    fn = y.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    return float(scores[order][int(np.argmax(f1))])


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "auc_pr": float(average_precision_score(y_true, scores)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
    }


def aggregate(runs: list[dict]) -> dict:
    """mean +/- std across seeds. A single seed is an anecdote, not a result."""
    keys = [k for k in runs[0] if k != "threshold"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in runs], dtype=float)
        out[k] = float(vals.mean())
        out[k + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return out


def fmt_row(name: str, m: dict) -> str:
    return (f"| {name} | {m['auc_roc']:.4f} ± {m.get('auc_roc_std', 0):.4f} "
            f"| {m['auc_pr']:.4f} ± {m.get('auc_pr_std', 0):.4f} "
            f"| {m['f1']:.4f} ± {m.get('f1_std', 0):.4f} "
            f"| {m['precision']:.4f} | {m['recall']:.4f} |")
