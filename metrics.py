import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
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

@cache_data
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
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
        'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
    }

# Metric packages definition
METRIC_PACKAGES: Dict[str, MetricPackage] = {
    'raw': {
        'auc': lambda y_true, y_score: roc_auc_score(y_true, y_score),
        'max_f1': lambda y_true, y_score: compute_max_f1(y_true, y_score),
    },
    'thresh': lambda y_true, y_pred: compute_confusion_elements(y_true, y_pred),
}

@cache_data
def compute_model_metrics(
    subset_df: pd.DataFrame,
    model_name: str,
    threshold: Optional[float] = None,
    package: str = 'raw'
) -> Dict[str, float]:
    """Compute metrics for a specific model on a subset of data.
    
    Args:
        subset_df: DataFrame containing the data for a specific query
        model_name: Name of the model column to extract data from
        threshold: Optional threshold to apply to predicted scores
        package: Metric package to use ('raw' or 'thresh')
    
    Returns:
        Dictionary of computed metrics
    """
    y_true = subset_df["test"][GT_COL]
    y_pred = subset_df[model_name][SCORE_COL]
    
    if threshold is not None:
        y_pred_binary = (y_pred >= threshold).astype(int)
    else:
        y_pred_binary = y_pred
        
    metric_package = METRIC_PACKAGES[package]
    
    if callable(metric_package):
        # Package is a function that returns multiple metrics
        return metric_package(y_true, y_pred_binary)
    else:
        # Package is a dictionary of metric functions
        return {metric_name: func(y_true, y_pred) for metric_name, func in metric_package.items()}

@cache_data
def compute_metrics(
    models_df: pd.DataFrame,
    queries_bool_df: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
    packages: Tuple[str, ...] = ("thresh", "raw")
) -> RawResults:
    """Compute metrics for all models and queries.
    
    Args:
        models_df: DataFrame with model predictions and ground truth
        queries_bool_df: Boolean DataFrame (samples x queries)
        thresholds: Optional list of thresholds for each model
        packages: Metric packages to compute
        
    Returns:
        Nested dictionary of results: {package: {model: {query: metrics}}}
    """
    raw_results: RawResults = {}
    model_names = get_meta_columns_in_order(models_df)[1:]

    for query_label, query_mask in queries_bool_df.items():
        # Create subset_df for this query using boolean mask
        subset_df = models_df[query_mask]
        
        for idx, model_name in enumerate(model_names):
            if model_name == "test":
                continue
                
            # Process each package
            for package_name in packages:
                # Determine threshold based on package name
                threshold = thresholds[idx] if thresholds is not None and package_name.startswith('thresh') else None
                
                # Compute all metrics for this package
                package_results = compute_model_metrics(
                    subset_df, model_name, threshold, package_name
                )
                
                # Store the entire package results
                raw_results.setdefault(package_name, {}).setdefault(model_name, {})[query_label] = package_results

    return raw_results

def apply_thresholds_and_evaluate(
    models_df: pd.DataFrame,
    queries_bool_df: pd.DataFrame,
    metrics: List[str],
    thresholds: Optional[List[float]]
) -> pd.DataFrame:
    """Apply thresholds and compute metrics for all models and queries.
    
    Args:
        models_df: DataFrame with model predictions and ground truth
        queries_bool_df: Boolean DataFrame (samples x queries)
        metrics: List of metric names to compute
        thresholds: Optional list of thresholds for each model
        
    Returns:
        DataFrame with metrics for each model and query
    """
    raw_results = compute_metrics(models_df, queries_bool_df, thresholds=thresholds)
    model_names = get_meta_columns_in_order(models_df)[1:]
    final_df = raw_results_to_final_df(raw_results, model_names, metrics, queries_bool_df.columns)
    return final_df

def raw_results_to_final_df(
    raw_results: RawResults,
    model_names: List[str],
    metrics: List[str],
    query_labels: pd.Index
) -> pd.DataFrame:
    """Convert raw results to a final DataFrame.
    
    Args:
        raw_results: Nested dictionary of results
        model_names: List of model names
        metrics: List of metric names to include
        query_labels: Index of query labels
        
    Returns:
        DataFrame with metrics for each model and query
    """
    columns = pd.MultiIndex.from_product([metrics, model_names], names=['Metric', 'Model'])
    final_df = pd.DataFrame(index=query_labels, columns=columns)

    for package_name, package_data in raw_results.items():
        for model_name, model_data in package_data.items():
            for query_label in query_labels:
                if query_label in model_data:
                    for metric_name, value in model_data[query_label].items():
                        if metric_name in metrics:  # Only include requested metrics
                            final_df.loc[query_label, (metric_name, model_name)] = value

    return final_df 