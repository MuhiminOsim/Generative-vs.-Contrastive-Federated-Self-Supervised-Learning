"""Utility functions for metrics and evaluation."""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def get_metrics(y_true, y_probs, threshold=0.5):
    """Calculate classification metrics from probabilities.
    
    Args:
        y_true: True binary labels
        y_probs: Predicted probabilities
        threshold: Classification threshold (default: 0.5)
        
    Returns:
        Dictionary of metrics
    """
    y_true = np.array(y_true).flatten()
    y_probs = np.array(y_probs).flatten()
    y_pred = (y_probs >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    metrics = {
        "auc": roc_auc_score(y_true, y_probs),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": float(tn) / (float(tn) + float(fp)) if (tn + fp) > 0 else 0.0,
        "sensitivity": float(tp) / (float(tp) + float(fn)) if (tp + fn) > 0 else 0.0,
    }
    
    return metrics


def find_optimal_threshold(y_true, y_probs):
    """Find optimal threshold using Youden's J statistic.
    
    Args:
        y_true: True binary labels
        y_probs: Predicted probabilities
        
    Returns:
        Optimal threshold value
    """
    y_true = np.array(y_true).flatten()
    y_probs = np.array(y_probs).flatten()
    
    thresholds = np.linspace(0, 1, 100)
    best_j = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            sensitivity = float(tp) / (float(tp) + float(fn)) if (tp + fn) > 0 else 0.0
            specificity = float(tn) / (float(tn) + float(fp)) if (tn + fp) > 0 else 0.0
            j = sensitivity + specificity - 1
            if j > best_j:
                best_j = j
                best_thresh = thresh
    
    return best_thresh
