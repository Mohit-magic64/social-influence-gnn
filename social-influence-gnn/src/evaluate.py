"""
Metrics for an imbalanced binary task.

Accuracy is deliberately absent. At a ~37% positive rate, "always predict 0" scores 63%
and is useless, so reporting accuracy would flatter every model here.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def pick_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """F1-maximising threshold. Call this on validation only."""
    order = np.argsort(-scores)
    y = y_true[order]
    # Sort by score descending, then sweep. Every prefix of this ordering is exactly the
    # prediction set you get at some threshold, so one pass covers all thresholds.
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    fn = y.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    # F1 written in terms of counts: 2TP / (2TP + FP + FN). The maximum() guards the
    # degenerate case where all three are zero.
    return float(scores[order][int(np.argmax(f1))])


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Threshold-free metrics plus the thresholded ones."""
    pred = (scores >= threshold).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "auc_pr": float(average_precision_score(y_true, scores)),
        # Both, and AUC-PR is the one to trust under imbalance. ROC's false positive
        # rate divides by every negative in the set, so a model can pile up false
        # positives without the ROC curve moving much. Precision has no such cushion.
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        # zero_division=0 rather than letting it warn: a model that predicts no
        # positives has precision 0, not an exception.
        "positive_rate": float(y_true.mean()),
        # Recorded per run. AUC-PR is only readable against the base rate, since a random
        # model scores the positive rate rather than 0.5.
        "threshold": float(threshold),
    }


def aggregate(runs: list[dict]) -> dict:
    """Mean and sd across seeds. One seed is an anecdote."""
    keys = [k for k in runs[0] if k != "threshold"]
    # Thresholds are not comparable across runs, so averaging them would be meaningless.
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in runs], dtype=float)
        out[k] = float(vals.mean())
        out[k + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        # ddof=1 is the sample standard deviation. With 3 seeds the difference from the
        # population version is not small.
    return out


def fmt_row(name: str, m: dict) -> str:
    """One markdown table row."""
    return (f"| {name} | {m['auc_roc']:.4f} ± {m.get('auc_roc_std', 0):.4f} "
            f"| {m['auc_pr']:.4f} ± {m.get('auc_pr_std', 0):.4f} "
            f"| {m['f1']:.4f} ± {m.get('f1_std', 0):.4f} "
            f"| {m['precision']:.4f} | {m['recall']:.4f} |")
