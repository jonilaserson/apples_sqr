import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, average_precision_score


def plot_curves(raw_results: dict, thresholds: list[float], colors: list, curve_type: str = 'roc', query_string: str = None, figsize: tuple = (5, 4)):
    """
    Plot either ROC or PR curves for all models on the same axis.

    Args:
        raw_results: Dictionary of pre-computed metrics for each model
        thresholds: List of thresholds, one per model.
        colors: List of colors to use.
        curve_type: 'roc' or 'pr'
        query_string: The query string to display as a subtitle
        figsize: Tuple of (width, height) for the figure size
    Returns:
        fig, ax: The figure and axis with all models plotted.
    """
    fig, ax = None, None

    # Get the plots package data for the specific query
    model_names = list(raw_results["plots"].keys())
    #title = ':'.join([curve_type.upper(), query_string])
    
    for idx, model_name in enumerate(model_names):
        threshold = thresholds[idx]
        color = colors[idx % len(colors)]
        #plot_metrics = plots_data[model_name][query_string]  # Access metrics for this model and query

        if curve_type == 'roc':
            fig, ax = plot_roc_curve_for_model(raw_results, threshold, color, model_name, query=query_string, ax=ax, figsize=figsize)
        elif curve_type == 'pr':
            fig, ax = plot_pr_curve_for_model(raw_results, threshold, color, model_name, query=query_string, ax=ax, figsize=figsize)
        else:
            raise ValueError(f"Unknown curve_type: {curve_type}")

    return fig, ax

def plot_roc_curve_for_model(raw_results, threshold, color, model_name, query, ax=None, figsize=(4, 4)):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    plot_metrics = raw_results['plots'][model_name][query]  # Access metrics for this model and query
    fpr = plot_metrics['fpr']
    tpr = plot_metrics['tpr']
    model_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, label=f"{model_name} (AUC={model_auc:.2f})", color=color)

    thresh_metrics = raw_results['thresh'][model_name][query]
    # Find the closest threshold point using ROC thresholds
    operation_fpr = thresh_metrics['fpr']
    operation_tpr = thresh_metrics['recall']
    ax.plot(operation_fpr, operation_tpr, 'o', color=color)

    if fig is not None: # Created new fig
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title("ROC: %s" % query)
        ax.grid(True)
        plt.tight_layout()

    ax.legend(loc='lower right', fontsize=8)
    return ax.figure, ax

def plot_pr_curve_for_model(raw_results, threshold, color, model_name, query, ax=None, figsize=(4, 4)):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    plot_metrics = raw_results['plots'][model_name][query]  # Access metrics for this model and query
    precision = plot_metrics['precision']
    recall = plot_metrics['recall']
    # Calculate AP using precision and recall directly
    ap = np.trapz(precision, recall)  # Area under the PR curve

    ax.plot(recall, precision, label=f"{model_name} (AP={ap:.2f})", color=color)

    # Find the closest threshold point using ROC thresholds since they are sorted
    thresh_metrics = raw_results['thresh'][model_name][query]
    operation_precision = thresh_metrics['precision']
    operation_recall = thresh_metrics['recall']
    ax.plot(operation_recall, operation_precision, 'o', color=color)

    if fig is not None: # Created new fig
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title("PR: %s" % query)
        ax.grid(True)
        plt.tight_layout()

    ax.legend(loc='lower left', fontsize=8)
    return ax.figure, ax 