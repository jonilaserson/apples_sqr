import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, precision_score, recall_score
from common import GT_COL, SCORE_COL
from metrics import compute_confusion_elements

def plot_roc_curve_for_model(y_true, y_score, threshold, color, model_name, ax=None, title=None, figsize=(4, 4)):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    model_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={model_auc:.2f})", color=color)

    y_pred = (y_score >= threshold).astype(int)
    tp, fp, fn, tn = compute_confusion_elements(y_true, y_pred)

    if (tp + fn) > 0 and (fp + tn) > 0:
        operation_fpr = fp / (fp + tn)
        operation_tpr = tp / (tp + fn)
        ax.plot(operation_fpr, operation_tpr, 'o', color=color)

    if fig is not None: # Created new fig
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title or "ROC Curve")
        ax.grid(True)
        plt.tight_layout()

    ax.legend(loc='lower right', fontsize=8)
    return ax.figure, ax

def plot_pr_curve_for_model(y_true, y_score, threshold, color, model_name, ax=None, title=None, figsize=(4, 4)):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    ax.plot(recall, precision, label=f"{model_name} (AP={ap:.2f})", color=color)

    y_pred = (y_score >= threshold).astype(int)
    operation_precision = precision_score(y_true, y_pred, zero_division=0)
    operation_recall = recall_score(y_true, y_pred, zero_division=0)

    ax.plot(operation_recall, operation_precision, 'o', color=color)

    if fig is not None: # Created new fig
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title or 'Precision-Recall Curve')
        ax.grid(True)
        plt.tight_layout()

    ax.legend(loc='lower left', fontsize=8)
    return ax.figure, ax

def plot_curves(subset_df, thresholds, colors, curve_type='roc', query_string=None, figsize=(5, 4)):
    """
    Plot either ROC or PR curves for all models on the same axis.

    Args:
        subset_df: Subset of the merged test and model dataframe.
        thresholds: List of thresholds, one per model.
        colors: List of colors to use.
        curve_type: 'roc' or 'pr'
        query_string: The query string to display as a subtitle
        figsize: Tuple of (width, height) for the figure size
    Returns:
        fig, ax: The figure and axis with all models plotted.
    """
    fig, ax = None, None

    model_names = get_meta_columns_in_order(subset_df)[1:]
    title = ':'.join([curve_type.upper(), query_string])
    for idx, model_name in enumerate(model_names):
        y_true = subset_df['test'][GT_COL]
        y_score = subset_df[model_name][SCORE_COL]
        threshold = thresholds[idx]
        color = colors[idx % len(colors)]

        if curve_type == 'roc':
            fig, ax = plot_roc_curve_for_model(y_true, y_score, threshold, color, model_name, ax=ax, title=title, figsize=figsize)
        elif curve_type == 'pr':
            fig, ax = plot_pr_curve_for_model(y_true, y_score, threshold, color, model_name, ax=ax, title=title, figsize=figsize)
        else:
            raise ValueError(f"Unknown curve_type: {curve_type}")

    return fig, ax 