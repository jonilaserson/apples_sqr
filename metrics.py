import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score, roc_curve, precision_recall_curve
from common import GT_COL, SCORE_COL, get_meta_columns_in_order, cache_data
from dataclasses import dataclass

# Metric packages definition
from enum import Enum
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

@dataclass
class DatasetInfo:
    """Information about the dataset being evaluated."""
    test_file: str
    total_samples: int
    filtered_samples: int
    filter_query: Optional[str]
    gt_column: str
    model_paths: Dict[str, str]
    available_classes: List[str]
    pos_classes: List[str]
    score_column: Optional[str] = None

class MetricType(Enum):
    SCALAR = "scalar"  # Metrics that go into final_df as scalar columns
    MATRIX = "matrix"  # Metrics that produce their own matrix/table
    PLOT = "plot"     # Metrics used for plotting

@dataclass
class MetricPackage:
    func: Callable
    type: MetricType
    needs_threshold: bool
    needs_binary: bool = True  # Whether it expects binary y_true/y_pred

# Type definitions for metric functions and results
RawResults = Dict[str, Dict[str, Dict[str, Dict[str, float]]]]  # package -> model -> query -> metric -> value

def compute_max_f1(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Compute maximum F1 score across all possible thresholds."""
    thresholds = np.unique(y_score)
    return max((f1_score(y_true, (y_score >= t).astype(int)) for t in thresholds), default=None)


def compute_confusion_elements(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute confusion matrix elements and derived metrics."""
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
    """Compute metrics needed for plotting ROC and PR curves."""
    fpr, tpr, roc_thresh = roc_curve(y_true, y_score)
    precision, recall, pr_thresh = precision_recall_curve(y_true, y_score)
    return {
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision[::-1],
        'recall': recall[::-1],
    }


def compute_multiclass_confusion_matrix(
    y_true_classes: pd.Series,
    y_pred_scores: pd.DataFrame,
    threshold: float = 0.5
) -> pd.DataFrame:
    """Compute multi-class confusion matrix using vectorized operations.
    
    Args:
        y_true_classes: Series of true class labels
        y_pred_scores: DataFrame where columns are class names and values are scores
        threshold: Threshold for considering a class prediction as positive
        
    Returns:
        DataFrame with confusion matrix and metrics, where:
        - Index is named 'Ground Truth' and contains true class labels
        - Columns are named 'Predicted' and contain predicted class labels
        - Last columns are metrics: recall, precision, prevalence, coverage, and specificity
    """
    # Get winning class and score in one go
    winning_scores = y_pred_scores.max(axis=1)
    winning_classes = y_pred_scores.idxmax(axis=1)
    
    # Create prediction column - winning class if score passes threshold, else 'dont_know'
    pred_classes = winning_classes.where(winning_scores >= threshold, 'dont_know')
    
    # Create DataFrame with GT and predictions
    df = pd.DataFrame({
        'GT_class': y_true_classes,
        'pred_class': pred_classes
    })
    
    # Get confusion matrix using value_counts and stack
    conf_matrix = df.value_counts(['GT_class', 'pred_class']).sort_index().unstack(fill_value=0)
    
    # Get all classes from y_pred_scores columns
    all_classes = list(y_pred_scores.columns)
    
    # Ensure all classes are present in both index and columns
    conf_matrix = conf_matrix.reindex(index=all_classes, columns=all_classes + ['dont_know'], fill_value=0)
    
    # Add clear names to the confusion matrix
    conf_matrix.index.name = 'Ground Truth'
    conf_matrix.columns.name = 'Predicted'
    
    # Calculate metrics
    stats = conf_matrix[[]].copy()
    
    # Recall: true positives / total actual positives
    stats['recall'] = np.diagonal(conf_matrix.values) / conf_matrix.sum(axis=1)
    
    # Precision: true positives / total predicted positives
    stats['precision'] = np.diagonal(conf_matrix.values) / conf_matrix.sum(axis=0)[:-1]
    
    # Prevalence: percentage of samples in each class
    total_samples = conf_matrix.sum().sum()
    stats['prevalence'] = conf_matrix.sum(axis=1) / total_samples
    
    # Coverage: percentage of samples that were confidently predicted (not "don't know")
    dont_know_percentage = conf_matrix['dont_know'] / total_samples
    stats['coverage'] = 1 - dont_know_percentage
    
    # Specificity: true negatives / (true negatives + false positives)
    # For each class, true negatives are all correct predictions of other classes
    # False positives are all incorrect predictions of this class
    for cls in all_classes:
        # True negatives: sum of diagonal elements excluding current class
        tn = sum(conf_matrix.loc[c, c] for c in all_classes if c != cls)
        # False positives: sum of column excluding diagonal
        fp = conf_matrix[cls].sum() - conf_matrix.loc[cls, cls]
        stats.loc[cls, 'specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Format all float values to 3 decimal places
    conf_matrix = conf_matrix
    stats = stats.round(3)
    
    return pd.concat([conf_matrix, stats], axis=1)

#
# Metric packages definition
METRIC_PACKAGES: Dict[str, MetricPackage] = {
    'raw': MetricPackage(
        func={
            'auc': roc_auc_score,
            'max_f1': compute_max_f1,
        },
        type=MetricType.SCALAR,
        needs_threshold=False
    ),
    'thresh': MetricPackage(
        func=compute_confusion_elements,
        type=MetricType.SCALAR,
        needs_threshold=True
    ),
    'plots': MetricPackage(
        func=compute_plot_metrics,
        type=MetricType.PLOT,
        needs_threshold=False
    ),
    'multiclass': MetricPackage(
        func=compute_multiclass_confusion_matrix,
        type=MetricType.MATRIX,
        needs_threshold=True,
        needs_binary=False
    )
}
@cache_data
def compute_model_metrics(
    models_df: pd.DataFrame,
    model_name: str,
    info: DatasetInfo,
    threshold: Optional[float] = None,
    package: str = 'raw',
    queries_bool_df: Optional[pd.DataFrame] = None
) -> Dict[str, Dict[str, float]]:
    """Compute metrics for a specific model across all queries.
    
    Args:
        models_df: DataFrame with model predictions and ground truth
        model_name: Name of the model column to extract data from
        info: Information about the dataset being evaluated
        threshold: Optional threshold to apply to predicted scores
        package: Metric package to use ('raw' or 'thresh')
        queries_bool_df: Optional boolean DataFrame (samples x queries). If None, computes metrics on entire dataset.
    
    Returns:
        Dictionary of metrics for each query: {query: {metric: value}}
    """
    # If no queries provided, treat entire dataset as one query
    if queries_bool_df is None:
        queries_bool_df = pd.DataFrame({'all': [True] * len(models_df)})

    metric_package = METRIC_PACKAGES[package]
    
    query_results = {}
    for query_label, query_mask in queries_bool_df.items():
        # Create subset_df for this query using boolean mask
        subset_df = models_df[query_mask]
        
        if metric_package.needs_binary:
            y_true = subset_df["test"][GT_COL]
            y_pred = subset_df[model_name][SCORE_COL]
        else:
            y_true = subset_df["test"][info.gt_column]
            y_pred = subset_df[model_name].drop(columns=[SCORE_COL])
        
        if threshold is not None:
            y_pred = (y_pred >= threshold).astype(int)
                    
        if callable(metric_package.func):
            # Package is a function that returns multiple metrics
            query_results[query_label] = metric_package.func(y_true, y_pred)
        else:
            # Package is a dictionary of metric functions
            query_results[query_label] = {metric_name: func(y_true, y_pred) for metric_name, func in metric_package.func.items()}
    
    return query_results

@cache_data
def compute_metrics(
    models_df: pd.DataFrame,
    info: DatasetInfo,
    queries_bool_df: Optional[pd.DataFrame] = None,
    thresholds: Optional[List[float]] = None,
    packages: Tuple[str, ...] = ("thresh", "raw")
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Compute metrics for all models and queries.
    
    Args:
        models_df: DataFrame with model predictions and ground truth
        info: Information about the dataset being evaluated
        queries_bool_df: Optional boolean DataFrame (samples x queries). If None, computes metrics on entire dataset.
        thresholds: Optional list of thresholds for each model
        packages: Metric packages to compute
        
    Returns:
        Nested dictionary of results: {model: {package: {query: metrics}}}
    """
    raw_results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    model_names = list(info.model_paths.keys())

    for idx, model_name in enumerate(model_names):
        # Initialize model entry
        raw_results[model_name] = {}
            
        # Process each package
        for package_name in packages:    
            metric_package = METRIC_PACKAGES[package_name]

            # Determine threshold based on package name
            threshold = thresholds[idx] if metric_package.needs_threshold else None
            
            # Compute metrics for all queries
            package_results = compute_model_metrics(
                models_df, model_name, info, threshold, package_name, queries_bool_df
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
            metric_package = METRIC_PACKAGES[package_name]
            if metric_package.type != MetricType.SCALAR:
                continue
            for query_label in query_labels:
                for metric_name, value in package_data[query_label].items():
                    if metric_name in metrics:  # Only include requested metrics
                        final_df.loc[query_label, (metric_name, model_name)] = value

    return final_df
