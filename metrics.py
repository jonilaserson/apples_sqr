import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
from common import GT_COL, SCORE_COL, get_meta_columns_in_order, cache_data

def compute_max_f1(y_true, y_score):
    thresholds = np.unique(y_score)
    return max((f1_score(y_true, (y_score >= t).astype(int)) for t in thresholds), default=None)

@cache_data
def compute_confusion_elements(y_true, y_pred):
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
METRIC_PACKAGES = {
    'raw': {
        'auc': lambda y_true, y_score: roc_auc_score(y_true, y_score),
        'max_f1': lambda y_true, y_score: compute_max_f1(y_true, y_score),
    },
    'thresh': lambda y_true, y_pred: compute_confusion_elements(y_true, y_pred),
}

# Function to compute metrics for a single model with caching
@cache_data
def compute_model_metrics(subset_df, model_name, threshold=None, package='raw'):
    """
    Compute metrics for a specific model on a subset of data.
    
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
def compute_metrics(models_df, queries, metrics, thresholds=None):
    raw_results = {}
    model_names = get_meta_columns_in_order(models_df)[1:]

    for query in queries:
        # Create subset_df for this query
        subset_df = models_df if query == 'all' else models_df.loc[models_df["test"].query(query).index]
        
        for idx, model_name in enumerate(model_names):
            if model_name == "test":
                continue
                
            threshold = None if thresholds is None else thresholds[idx]
            
            # Process each requested metric
            for metric_name in metrics:
                # Determine package based on metric and threshold
                if metric_name in ['auc', 'max_f1']:
                    package = 'raw'
                else:
                    package = 'thresh' if thresholds is not None else 'raw'
                
                # Compute all metrics for this package
                package_results = compute_model_metrics(
                    subset_df, model_name, threshold, package
                )
                
                # Store only the requested metric
                if metric_name in package_results:
                    raw_results.setdefault(metric_name, {}).setdefault(model_name, {})[query] = package_results[metric_name]

    return raw_results

def apply_thresholds_and_evaluate(models_df, queries_dict, metrics, thresholds):
    # With the new design, we don't need to separate calculations with and without thresholds
    raw_results = compute_metrics(models_df, list(queries_dict.keys()), metrics, thresholds)
    model_names = get_meta_columns_in_order(models_df)[1:]
    final_df = raw_results_to_final_df(raw_results, model_names, metrics, queries_dict)
    return final_df

def raw_results_to_final_df(raw_results, model_names, metrics, queries: dict):
    columns = pd.MultiIndex.from_product([metrics, model_names], names=['Metric', 'Model'])
    final_df = pd.DataFrame(index=list(queries.values()), columns=columns)

    for metric_name in metrics:
        for model_name in model_names:
            for query, query_label in queries.items():
                value = raw_results.get(metric_name, {}).get(model_name, {}).get(query, None)
                final_df.loc[query_label, (metric_name, model_name)] = value

    return final_df 