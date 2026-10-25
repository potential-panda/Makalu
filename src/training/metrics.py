"""
Custom evaluation metrics for trading models.
"""

import numpy as np


def soft_trading_precision_fpr_metric(preds, train_data):
    """
    Stable version of trading metric using probabilities (soft metrics).
    Better suited for early stopping as it is continuous and less noisy.
    """
    labels = train_data.get_label()
    num_class = 3
    num_samples = len(labels)

    # LightGBM multiclass feval preds are (num_class, num_samples) flattened
    preds = preds.reshape(num_class, num_samples).T

    # Long (Class 1)
    p_long = preds[:, 1]
    is_long = labels == 1
    is_not_long = labels != 1

    soft_tp_long = np.sum(p_long * is_long)
    soft_fp_long = np.sum(p_long * is_not_long)
    soft_precision_long = soft_tp_long / (soft_tp_long + soft_fp_long + 1e-8)
    soft_fpr_long = soft_fp_long / (np.sum(is_not_long) + 1e-8)

    # Short (Class 0)
    p_short = preds[:, 0]
    is_short = labels == 0
    is_not_short = labels != 0

    soft_tp_short = np.sum(p_short * is_short)
    soft_fp_short = np.sum(p_short * is_not_short)
    soft_precision_short = soft_tp_short / (soft_tp_short + soft_fp_short + 1e-8)
    soft_fpr_short = soft_fp_short / (np.sum(is_not_short) + 1e-8)

    score = (soft_precision_long + soft_precision_short) / 2.0 - (
        soft_fpr_long + soft_fpr_short
    ) / 2.0

    return "soft_trading_score", score, True


def trading_precision_fpr_metric(preds, train_data):
    """
    Custom LightGBM metric: (Precision_Long + Precision_Short) / 2 - (FPR_Long + FPR_Short) / 2.

    Classes: 0=Short, 1=Long, 2=Timeout

    Args:
        preds: Predicted probabilities (flat array of size N * num_classes)
        train_data: LightGBM Dataset object containing labels

    Returns:
        name, value, is_higher_better
    """
    labels = train_data.get_label()
    num_class = 3
    num_samples = len(labels)

    # Reshape predictions to (num_samples, num_class)
    # LightGBM multi-class preds are usually (num_class, num_samples) flattened in older versions
    # but in newer versions they might be (num_samples, num_class).
    # Actually for feval it's (num_samples * num_class, )
    # and it is filled class by class: [c0_s0, c0_s1, ..., c1_s0, c1_s1, ...]
    preds = preds.reshape(num_class, num_samples).T
    y_pred = np.argmax(preds, axis=1)

    # Precision = TP / (TP + FP)
    # FPR = FP / (FP + TN) = FP / (Total Negatives)

    # Long (Class 1)
    tp_long = np.sum((y_pred == 1) & (labels == 1))
    fp_long = np.sum((y_pred == 1) & (labels != 1))
    precision_long = tp_long / (tp_long + fp_long) if (tp_long + fp_long) > 0 else 0

    neg_long = np.sum(labels != 1)
    fpr_long = fp_long / neg_long if neg_long > 0 else 0

    # Short (Class 0)
    tp_short = np.sum((y_pred == 0) & (labels == 0))
    fp_short = np.sum((y_pred == 0) & (labels != 0))
    precision_short = (
        tp_short / (tp_short + fp_short) if (tp_short + fp_short) > 0 else 0
    )

    neg_short = np.sum(labels != 0)
    fpr_short = fp_short / neg_short if neg_short > 0 else 0

    # Combined score
    # We want to maximize precision and minimize FPR
    score = (precision_long + precision_short) / 2.0 - (fpr_long + fpr_short) / 2.0

    return "trading_score", score, True


def calculate_trading_metrics(y_true, y_pred_prob):
    """
    Calculate detailed trading metrics for multi-class classification.

    Args:
        y_true: True labels (0=Short, 1=Long, 2=Timeout)
        y_pred_prob: Predicted probabilities (N, 3)

    Returns:
        Dictionary of metrics
    """
    y_pred = np.argmax(y_pred_prob, axis=1)

    metrics = {}

    for cls_name, cls_idx in [("short", 0), ("long", 1), ("timeout", 2)]:
        tp = np.sum((y_pred == cls_idx) & (y_true == cls_idx))
        fp = np.sum((y_pred == cls_idx) & (y_true != cls_idx))
        fn = np.sum((y_pred != cls_idx) & (y_true == cls_idx))
        tn = np.sum((y_pred != cls_idx) & (y_true != cls_idx))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        metrics[f"{cls_name}_precision"] = precision
        metrics[f"{cls_name}_recall"] = recall
        metrics[f"{cls_name}_fpr"] = fpr

    metrics["trading_score"] = (
        metrics["long_precision"] + metrics["short_precision"]
    ) / 2.0 - (metrics["long_fpr"] + metrics["short_fpr"]) / 2.0

    return metrics
