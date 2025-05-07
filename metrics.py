import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score, roc_curve, precision_recall_curve
from common import GT_COL, SCORE_COL, get_meta_columns_in_order, cache_data

# Type definitions for metric functions and results
MetricFunc = Callable[[np.ndarray, np.ndarray], float]
MetricPackage = Union[Dict[str, MetricFunc], Callable[[np.ndarray, np.ndarray], Dict[str, float]]]
RawResults = Dict[str, Dict[str, Dict[str, Dict[str, float]]]]  # package -> model -> query -> metric -> value

def compute_max_f1(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Compute maximum F1 score across all possible thresholds.
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted scores
        
    Returns:
        Maximum F1 score or None if no valid threshold found
    """
    thresholds = np.unique(y_score)
    return max((f1_score(y_true, (y_score >= t).astype(int)) for t in thresholds), default=None)


def compute_confusion_elements(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute confusion matrix elements and derived metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics including tp, fp, fn, tn, precision, recall, etc.
    """
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    fn = np.logical_and(y_true == 1, y_pred == 0).sum()
    tn = np.logical_and(y_true == 0, y_pred == 0).sum()
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'fpr': fp / (tn + fp) if (tn + fp) > 0 else 0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
        'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
    }

def compute_plot_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute metrics needed for plotting ROC and PR curves.
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted scores
        
    Returns:
        Dictionary containing:
            - fpr, tpr, roc_thresh: for ROC curve
            - precision, recall, pr_thresh: for PR curve
    """
    fpr, tpr, roc_thresh = roc_curve(y_true, y_score)
    precision, recall, pr_thresh = precision_recall_curve(y_true, y_score)
    return {
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision[::-1],
        'recall': recall[::-1],
    }

# Metric packages definition
METRIC_PACKAGES: Dict[str, MetricPackage] = {
    'raw': {
        'auc': roc_auc_score,
        'max_f1': compute_max_f1,
    },
    'thresh': compute_confusion_elements,
    'plots': compute_plot_metrics
}

@cache_data
def compute_model_metrics(
    models_df: pd.DataFrame,
    model_name: str,
    threshold: Optional[float] = None,
    package: str = 'raw',
    queries_bool_df: Optional[pd.DataFrame] = None
) -> Dict[str, Dict[str, float]]:
    """Compute metrics for a specific model across all queries.
    
    Args:
        models_df: DataFrame with model predictions and ground truth
        model_name: Name of the model column to extract data from
        threshold: Optional threshold to apply to predicted scores
        package: Metric package to use ('raw' or 'thresh')
        queries_bool_df: Optional boolean DataFrame (samples x queries). If None, computes metrics on entire dataset.
    
    Returns:
        Dictionary of metrics for each query: {query: {metric: value}}
    """
    # If no queries provided, treat entire dataset as one query
    if queries_bool_df is None:
        queries_bool_df = pd.DataFrame({'all': [True] * len(models_df)})
    
    query_results = {}
    for query_label, query_mask in queries_bool_df.items():
        # Create subset_df for this query using boolean mask
        subset_df = models_df[query_mask]
        
        y_true = subset_df["test"][GT_COL]
        y_pred = subset_df[model_name][SCORE_COL]
        
        if threshold is not None:
            y_pred_binary = (y_pred >= threshold).astype(int)
        else:
            y_pred_binary = y_pred
            
        metric_package = METRIC_PACKAGES[package]
        
        if callable(metric_package):
            # Package is a function that returns multiple metrics
            query_results[query_label] = metric_package(y_true, y_pred_binary)
        else:
            # Package is a dictionary of metric functions
            query_results[query_label] = {metric_name: func(y_true, y_pred) for metric_name, func in metric_package.items()}
    
    return query_results

@cache_data
def compute_metrics(
    models_df: pd.DataFrame,
    queries_bool_df: Optional[pd.DataFrame] = None,
    thresholds: Optional[List[float]] = None,
    packages: Tuple[str, ...] = ("thresh", "raw")
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Compute metrics for all models and queries.
    
    Args:
        models_df: DataFrame with model predictions and ground truth
        queries_bool_df: Optional boolean DataFrame (samples x queries). If None, computes metrics on entire dataset.
        thresholds: Optional list of thresholds for each model
        packages: Metric packages to compute
        
    Returns:
        Nested dictionary of results: {model: {package: {query: metrics}}}
    """
    raw_results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    model_names = get_meta_columns_in_order(models_df)[1:]

    for idx, model_name in enumerate(model_names):
        if model_name == "test":
            continue
            
        # Initialize model entry
        raw_results[model_name] = {}
            
        # Process each package
        for package_name in packages:
            # Determine threshold based on package name
            threshold = thresholds[idx] if thresholds is not None and package_name.startswith('thresh') else None
            
            # Compute metrics for all queries
            package_results = compute_model_metrics(
                models_df, model_name, threshold, package_name, queries_bool_df
            )
            
            # Store the entire package results
            raw_results[model_name][package_name] = package_results

    return raw_results

def raw_results_to_final_df(
    raw_results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    model_names: List[str],
    metrics: List[str],
    query_labels: pd.Index
) -> pd.DataFrame:
    """Convert raw results to a final DataFrame.
    
    Args:
        raw_results: Nested dictionary of results with structure {model: {package: {query: metrics}}}
        model_names: List of model names
        metrics: List of metric names to include
        query_labels: Index of query labels
        
    Returns:
        DataFrame with metrics for each model and query
    """
    columns = pd.MultiIndex.from_product([metrics, model_names], names=['Metric', 'Model'])
    final_df = pd.DataFrame(index=query_labels, columns=columns)

    for model_name, model_data in raw_results.items():
        for package_name, package_data in model_data.items():
            if package_name == 'plots':
                continue
            for query_label in query_labels:
                for metric_name, value in package_data[query_label].items():
                    if metric_name in metrics:  # Only include requested metrics
                        final_df.loc[query_label, (metric_name, model_name)] = value

    return final_df 